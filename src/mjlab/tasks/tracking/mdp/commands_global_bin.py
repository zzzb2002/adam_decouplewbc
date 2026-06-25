from __future__ import annotations

import copy
import glob
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))

def _resolve_motion_files(
    motion_file: str, motion_files: tuple[str, ...], motion_dir: str
) -> list[str]:
  """Resolve motion files from various inputs."""
  if motion_dir:
    pattern = str(Path(motion_dir) / "*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
      raise ValueError(f"No .npz files found in {motion_dir}")
    return files
  elif motion_files:
    return list(motion_files)
  elif motion_file:
    return [motion_file]
  else:
    raise ValueError("Must specify one of: motion_file, motion_files, or motion_dir")

class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = self.joint_pos.shape[0]

class MultiMotionLoader:
  """Loader for multiple motion clips."""
  
  def __init__(
    self, motion_files: list[str], body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    self._body_indexes = body_indexes
    self._device = device
    
    # 加载所有文件并拼接
    all_joint_pos = []
    all_joint_vel = []
    all_body_pos_w = []
    all_body_quat_w = []
    all_body_lin_vel_w = []
    all_body_ang_vel_w = []
    
    for f in motion_files:
      data = np.load(f)
      all_joint_pos.append(torch.tensor(data["joint_pos"], dtype=torch.float32, device=device))
      all_joint_vel.append(torch.tensor(data["joint_vel"], dtype=torch.float32, device=device))
      all_body_pos_w.append(torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device))
      all_body_quat_w.append(torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device))
      all_body_lin_vel_w.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device))
      all_body_ang_vel_w.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device))
    
    # 沿时间维度拼接
    self.joint_pos = torch.cat(all_joint_pos, dim=0)
    self.joint_vel = torch.cat(all_joint_vel, dim=0)
    self._body_pos_w = torch.cat(all_body_pos_w, dim=0)
    self._body_quat_w = torch.cat(all_body_quat_w, dim=0)
    self._body_lin_vel_w = torch.cat(all_body_lin_vel_w, dim=0)
    self._body_ang_vel_w = torch.cat(all_body_ang_vel_w, dim=0)
    
    # 筛选指定的body
    self.body_pos_w = self._body_pos_w[:, body_indexes]
    self.body_quat_w = self._body_quat_w[:, body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, body_indexes]
    
    self.time_step_total = self.joint_pos.shape[0]
    
    # 记录每个clip的边界用于分bin
    clip_sizes = [p.shape[0] for p in all_joint_pos]
    self.clip_boundaries = [0] + list(np.cumsum(clip_sizes))
    self.num_clips = len(motion_files)
    self.motion_files = motion_files

class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    motion_files = _resolve_motion_files(
      self.cfg.motion_file, self.cfg.motion_files, self.cfg.motion_dir
    )

    self.motion = MultiMotionLoader(
      motion_files, self.body_indexes, device=self.device
    )

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self._use_multi_clip = (
        hasattr(self.motion, "num_clips") and self.motion.num_clips > 1
    )
    if self._use_multi_clip:
        # 每个clip独立的bin统计
        self._clip_bin_counts = []  # 每个clip的时间步对应bin数
        
        boundaries = self.motion.clip_boundaries
        for i in range(self.motion.num_clips):
            clip_steps = boundaries[i+1] - boundaries[i]
            bin_count = int(clip_steps // (1 / env.step_dt)) + 1
            self._clip_bin_counts.append(bin_count)
            kernel = torch.tensor(
                [self.cfg.adaptive_lambda**j for j in range(self.cfg.adaptive_kernel_size)],
                device=self.device,
            )
        
        # 当前env所在的clip
        self._env_clips = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._max_clip_time_steps = max(
            boundaries[i+1] - boundaries[i] for i in range(self.motion.num_clips)
        )
    
        # 新增：全局 bin buffer
        self._global_bin_boundaries = torch.tensor(
            [0] + list(np.cumsum(self._clip_bin_counts)),
            dtype=torch.long, device=self.device
        )
        self._global_bin_count = self._global_bin_boundaries[-1].item()
        self._global_bin_failed_count = torch.zeros(
            self._global_bin_count, dtype=torch.float, device=self.device
        )
        self._global_current_bin_failed = torch.zeros(
            self._global_bin_count, dtype=torch.float, device=self.device
        )
        self._global_kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self._global_kernel = self._global_kernel / self._global_kernel.sum()
        
    else:
        # 原有单clip逻辑保持不变
        self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
        self.bin_failed_count = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
      if not self._use_multi_clip:
          # === 单clip原有逻辑（保持不变） ===
          episode_failed = self._env.termination_manager.terminated[env_ids]
          if torch.any(episode_failed):
              current_bin_index = torch.clamp(
                  (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
                  0, self.bin_count - 1,
              )
              fail_bins = current_bin_index[env_ids][episode_failed]
              self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

          sampling_probabilities = (
              self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
          )
          sampling_probabilities = torch.nn.functional.pad(
              sampling_probabilities.unsqueeze(0).unsqueeze(0),
              (0, self.cfg.adaptive_kernel_size - 1),
              mode="replicate",
          )
          sampling_probabilities = torch.nn.functional.conv1d(
              sampling_probabilities, self.kernel.view(1, 1, -1)
          ).view(-1)
          sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

          sampled_bins = torch.multinomial(
              sampling_probabilities, len(env_ids), replacement=True
          )
          self.time_steps[env_ids] = (
              (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
              / self.bin_count * (self.motion.time_step_total - 1)
          ).long()

          # Update metrics
          H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
          H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
          pmax, imax = sampling_probabilities.max(dim=0)
          self.metrics["sampling_entropy"][:] = H_norm
          self.metrics["sampling_top1_prob"][:] = pmax
          self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
          return
      
      # === 多clip adaptive sampling ===
      boundaries_t = torch.tensor(self.motion.clip_boundaries, device=self.device)
      
      # 1. 更新全局失败统计
      episode_failed = self._env.termination_manager.terminated[env_ids]
      if torch.any(episode_failed):
          # 计算每个 env 的全局 bin index
          clip_starts = boundaries_t[self._env_clips[env_ids]]
          clips = self._env_clips[env_ids]
          time_in_clip = self.time_steps[env_ids] - clip_starts
          
          # clip 内的 bin index
          clip_bin_counts_t = torch.tensor(self._clip_bin_counts, device=self.device)
          clip_steps_t = boundaries_t[clips + 1] - boundaries_t[clips]
          clip_bin_count = clip_bin_counts_t[clips]
          
          current_bins = torch.clamp(
              (time_in_clip * clip_bin_count) // torch.clamp(clip_steps_t, min=1),
              0, int(clip_bin_count.max().item())
          )
          # 全局 bin index = clip 的全局起始 bin + clip 内 bin
          global_bins = self._global_bin_boundaries[clips] + current_bins
          fail_bins = global_bins[episode_failed]
          self._global_current_bin_failed[:] = torch.bincount(
              fail_bins, minlength=self._global_bin_count
          ).float()

      # 2. 全局 adaptive sampling
      sampling_prob = (
          self._global_bin_failed_count 
          + self.cfg.adaptive_uniform_ratio / float(self._global_bin_count)
      )
      sampling_prob = torch.nn.functional.pad(
          sampling_prob.unsqueeze(0).unsqueeze(0),
          (0, self.cfg.adaptive_kernel_size - 1),
          mode="replicate",
      )
      sampling_prob = torch.nn.functional.conv1d(
          sampling_prob, self._global_kernel.view(1, 1, -1)
      ).view(-1)
      sampling_prob = sampling_prob / sampling_prob.sum()

      sampled_global_bins = torch.multinomial(
          sampling_prob, len(env_ids), replacement=True
      )
      
      # 3. 全局 bin → (clip, clip_offset)
      clip_indices = torch.searchsorted(
          self._global_bin_boundaries, sampled_global_bins, right=True
      ) - 1
      self._env_clips[env_ids] = clip_indices
      
      clip_starts = boundaries_t[clip_indices]
      clip_ends = boundaries_t[clip_indices + 1]
      clip_steps = clip_ends - clip_starts
      
      clip_bin_counts_t = torch.tensor(self._clip_bin_counts, device=self.device)
      clip_bin_count = clip_bin_counts_t[clip_indices]
      
      # bin 在 clip 内的偏移
      bin_in_clip = sampled_global_bins - self._global_bin_boundaries[clip_indices]
      
      self.time_steps[env_ids] = clip_starts + (
          (bin_in_clip.float() + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
          / clip_bin_count * (clip_steps - 1)
      ).long()
      
      # metrics
      H = -(sampling_prob * (sampling_prob + 1e-12).log()).sum()
      H_norm = H / math.log(self._global_bin_count) if self._global_bin_count > 1 else 1.0
      pmax, imax = sampling_prob.max(dim=0)
      self.metrics["sampling_entropy"][:] = H_norm
      self.metrics["sampling_top1_prob"][:] = pmax
      self.metrics["sampling_top1_bin"][:] = imax.float() / self._global_bin_count


  def _uniform_sampling(self, env_ids: torch.Tensor):
      if not self._use_multi_clip:
          self.time_steps[env_ids] = torch.randint(
              0, self.motion.time_step_total, (len(env_ids),), device=self.device
          )
          self.metrics["sampling_entropy"] = 1.0
          self.metrics["sampling_top1_prob"] = 1.0 / self.bin_count
          self.metrics["sampling_top1_bin"] = 0.5
      else:
          boundaries_t = torch.tensor(self.motion.clip_boundaries, device=self.device)
          # 1. 每个 env 等概率随机选 clip
          clip_indices = torch.randint(
              0, self.motion.num_clips, (len(env_ids),), device=self.device
          )
          self._env_clips[env_ids] = clip_indices

          # 2. 在选中的 clip 内随机选时间步
          clip_starts = boundaries_t[clip_indices]
          clip_ends = boundaries_t[clip_indices + 1]
          clip_lengths = clip_ends - clip_starts
          random_offsets = torch.rand(len(env_ids), device=self.device)
          self.time_steps[env_ids] = clip_starts + (random_offsets * clip_lengths).long()
          self.metrics["sampling_entropy"] = 1.0
          self.metrics["sampling_top1_prob"] = 1.0 / self.motion.num_clips
          self.metrics["sampling_top1_bin"] = 0.5

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self.robot.reset(env_ids=env_ids)

  def _update_command(self):
    self.time_steps += 1
    if self._use_multi_clip:
        boundaries_t = torch.tensor(self.motion.clip_boundaries, device=self.device)
        clip_ends = boundaries_t[self._env_clips + 1]
        env_ids = torch.where(self.time_steps >= clip_ends)[0]
        if env_ids.numel() > 0:
            self._resample_command(env_ids)
    else:
        # 单clip模式：原有逻辑
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        if env_ids.numel() > 0:
            self._resample_command(env_ids)

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    if self.cfg.sampling_mode == "adaptive":
        if self._use_multi_clip:
            self._global_bin_failed_count = (
                self.cfg.adaptive_alpha * self._global_current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self._global_bin_failed_count
            )
            self._global_current_bin_failed.zero_()
        else:
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str = ""  # 单个文件（兼容旧用法）
  motion_files: tuple[str, ...] = ()  # 多个文件
  motion_dir: str = ""  # 目录：加载目录下所有npz
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)