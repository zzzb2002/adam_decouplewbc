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

    # === sonic 风格 adaptive sampling 初始化 ===
    if self._use_multi_clip:
        self._env_clips = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        num_clips = self.motion.num_clips
        boundaries = self.motion.clip_boundaries
    else:
        num_clips = 1
        boundaries = [0, self.motion.time_step_total]

    self.adp_samp_num_frames = torch.zeros(num_clips, dtype=torch.long, device=self.device)
    for i in range(num_clips):
        self.adp_samp_num_frames[i] = boundaries[i + 1] - boundaries[i]

    lengths_shifted = self.adp_samp_num_frames.roll(1)
    lengths_shifted[0] = 0
    self.adp_samp_length_starts = lengths_shifted.cumsum(0)
    self.adp_samp_total_frames = self.adp_samp_num_frames.sum().item()

    bin_size = self.cfg.adaptive_bin_size
    all_bins = []
    all_bin_lengths = []
    all_bin_new_motion_masks = []
    all_num_peer_bins = []
    all_motion_to_bins = []

    self.adp_samp_frame_to_bin = torch.zeros(
        self.adp_samp_total_frames, dtype=torch.long, device=self.device
    )
    cur_bin_idx = 0
    for clip_id in range(num_clips):
        num_frames = self.adp_samp_num_frames[clip_id].item()
        frame_start = self.adp_samp_length_starts[clip_id].item()
        frame_end = frame_start + num_frames

        bin_starts = torch.arange(
            0, num_frames, bin_size, device=self.device, dtype=torch.long
        )
        bin_ends = torch.minimum(
            bin_starts + bin_size,
            torch.tensor(num_frames, device=self.device, dtype=torch.long),
        )
        num_bins = len(bin_starts)
        clip_ids = torch.full(
            (num_bins,), clip_id, device=self.device, dtype=torch.long
        )
        motion_bins = torch.stack([clip_ids, bin_starts, bin_ends], dim=1)
        all_bins.append(motion_bins)

        all_bin_lengths.append(bin_ends - bin_starts)

        new_motion_mask = torch.zeros(num_bins, device=self.device, dtype=torch.bool)
        new_motion_mask[0] = True
        all_bin_new_motion_masks.append(new_motion_mask)

        peer_bins = torch.full(
            (num_bins,), num_bins, device=self.device, dtype=torch.long
        )
        all_num_peer_bins.append(peer_bins)

        bin_ids = torch.zeros(num_frames, device=self.device, dtype=torch.long)
        if num_bins > 1:
            switch_points = bin_starts[1:].long()
            bin_ids[switch_points] = 1
            bin_ids = bin_ids.cumsum(0) + cur_bin_idx
        else:
            bin_ids[:] = cur_bin_idx
        self.adp_samp_frame_to_bin[frame_start:frame_end] = bin_ids

        motion_bin_indices = torch.arange(
            cur_bin_idx, cur_bin_idx + num_bins, device=self.device, dtype=torch.long
        )
        all_motion_to_bins.append(motion_bin_indices)

        cur_bin_idx += num_bins

    self.adp_samp_bins = torch.cat(all_bins, dim=0)
    self.adp_samp_bin_motion_length = torch.cat(all_bin_lengths, dim=0).float()
    self.adp_samp_bin_new_motion_mask = torch.cat(all_bin_new_motion_masks, dim=0)
    self.adp_samp_num_peer_bins = torch.cat(all_num_peer_bins, dim=0)
    self.adp_samp_num_bins = len(self.adp_samp_bins)
    self.orig_motion_id_to_bins = all_motion_to_bins

    self.adp_samp_bin_weights = (
        self.adp_samp_bin_motion_length / self.adp_samp_bin_motion_length.mean()
    )
    if self.cfg.adaptive_sequence_length_agnostic:
        self.adp_samp_bin_weights = (
            self.adp_samp_bin_weights / self.adp_samp_num_peer_bins.float()
        )

    init_num = self.cfg.adaptive_init_num_failures
    self.adp_samp_num_failures = (
        torch.ones(self.adp_samp_num_bins, device=self.device, dtype=torch.float32) * init_num
    )
    self.adp_samp_num_episodes = (
        torch.ones(self.adp_samp_num_bins, device=self.device, dtype=torch.float32) * init_num
    )
    self.adp_samp_failure_rate = torch.ones(
        self.adp_samp_num_bins, device=self.device, dtype=torch.float32
    )
    self.adp_samp_failure_rate_raw = torch.ones(
        self.adp_samp_num_bins, device=self.device, dtype=torch.float32
    )

    # mjlab 加载了所有 clips，active bins = all bins
    self.adp_samp_active_motion_bins = torch.arange(
        self.adp_samp_num_bins, device=self.device, dtype=torch.long
    )

    self._compute_adaptive_sampling_probabilities()

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
    n = len(env_ids)
    sampled_bin_ids = torch.multinomial(
        self.adp_sampling_active_prob, num_samples=n, replacement=True
    )
    bin_ids = self.adp_samp_active_motion_bins[sampled_bin_ids]
    bins = self.adp_samp_bins[bin_ids]
    clip_ids, bin_start, bin_end = bins[:, 0], bins[:, 1], bins[:, 2]

    offsets = (torch.rand(n, device=self.device) * (bin_end - bin_start)).floor().long()
    time_steps_in_clip = bin_start + offsets

    if self.cfg.adaptive_pre_failure_sample_window > 0:
        offset = torch.randint(
            self.cfg.adaptive_pre_failure_sample_window, (n,), device=self.device
        )
        time_steps_in_clip = torch.clamp(time_steps_in_clip - offset, min=0)

    if self._use_multi_clip:
        boundaries_t = torch.tensor(self.motion.clip_boundaries, device=self.device)
        clip_starts = boundaries_t[clip_ids]
        self.time_steps[env_ids] = clip_starts + time_steps_in_clip
        self._env_clips[env_ids] = clip_ids
    else:
        self.time_steps[env_ids] = time_steps_in_clip

  def _uniform_sampling(self, env_ids: torch.Tensor):
      if not self._use_multi_clip:
          self.time_steps[env_ids] = torch.randint(
              0, self.motion.time_step_total, (len(env_ids),), device=self.device
          )
          num_bins = self.adp_samp_num_bins
          self.metrics["sampling_entropy"][:] = 1.0
          self.metrics["sampling_top1_prob"][:] = 1.0 / num_bins
          self.metrics["sampling_top1_bin"][:] = 0.5
      else:
          boundaries_t = torch.tensor(self.motion.clip_boundaries, device=self.device)
          clip_indices = torch.randint(
              0, self.motion.num_clips, (len(env_ids),), device=self.device
          )
          self._env_clips[env_ids] = clip_indices
          clip_starts = boundaries_t[clip_indices]
          clip_ends = boundaries_t[clip_indices + 1]
          clip_lengths = clip_ends - clip_starts
          random_offsets = torch.rand(len(env_ids), device=self.device)
          self.time_steps[env_ids] = clip_starts + (random_offsets * clip_lengths).long()
          num_bins = self.adp_samp_num_bins
          self.metrics["sampling_entropy"][:] = 1.0
          self.metrics["sampling_top1_prob"][:] = 1.0 / num_bins
          self.metrics["sampling_top1_bin"][:] = 0.5

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

  def _update_adaptive_sampling_stats(self):
    if self._use_multi_clip:
        boundaries_t = torch.tensor(self.motion.clip_boundaries, device=self.device)
        clip_starts = boundaries_t[self._env_clips]
        time_in_clip = self.time_steps - clip_starts
        global_frames = self.adp_samp_length_starts[self._env_clips] + time_in_clip
    else:
        global_frames = self.time_steps

    bin_ids = self.adp_samp_frame_to_bin[global_frames]
    counts = torch.bincount(bin_ids, minlength=self.adp_samp_num_bins).float()
    counts = counts / self.adp_samp_bin_motion_length
    self.adp_samp_num_episodes += counts

    terminated = self._env.termination_manager.terminated
    if terminated.any():
        failed_bins = bin_ids[terminated]
        failure_counts = torch.bincount(failed_bins, minlength=self.adp_samp_num_bins).float()
        self.adp_samp_num_failures += (
            failure_counts * self.cfg.adaptive_failure_counts_multiplier
        )

  def _compute_adaptive_sampling_probabilities(self):
    if self.cfg.adaptive_use_failure_rate_decay:
        gamma = self.cfg.adaptive_decay_gamma
        num_steps = self.adp_samp_num_bins
        failure_rate = self.adp_samp_num_failures / self.adp_samp_num_episodes
        failure_rate_w_decay = torch.zeros_like(failure_rate)
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_failure_rate = 0
                next_is_not_terminal = 0.0
            else:
                next_failure_rate = failure_rate_w_decay[step + 1]
                next_is_not_terminal = (
                    1.0 - self.adp_samp_bin_new_motion_mask[step + 1].float()
                )
            failure_rate_w_decay[step] = (
                failure_rate[step] + next_is_not_terminal * gamma * next_failure_rate
            )
        self.adp_samp_failure_rate = failure_rate_w_decay
    else:
        self.adp_samp_failure_rate = self.adp_samp_num_failures / self.adp_samp_num_episodes

    self.adp_samp_failure_rate_raw = self.adp_samp_failure_rate.clone()

    active_rate = self.adp_samp_failure_rate[self.adp_samp_active_motion_bins].double()
    upper_bound = active_rate.mean() * self.cfg.adaptive_failure_rate_max_over_mean
    active_rate_clipped = torch.clamp(active_rate, 0.0, upper_bound)

    failure_prob = active_rate_clipped / active_rate_clipped.sum()
    uniform_prob = torch.ones_like(failure_prob) / len(failure_prob)
    prob = (
        failure_prob * (1 - self.cfg.adaptive_uniform_sampling_rate)
        + uniform_prob * self.cfg.adaptive_uniform_sampling_rate
    )

    prob *= self.adp_samp_bin_weights[self.adp_samp_active_motion_bins]
    prob = prob / prob.sum()

    # max_prob_per_bin
    max_prob_per_bin_cfg = self.cfg.adaptive_max_prob_per_bin
    if max_prob_per_bin_cfg is not None:
        num_active_bins = len(self.adp_samp_active_motion_bins)
        if max_prob_per_bin_cfg == "auto":
            multiplier = self.cfg.adaptive_failure_rate_max_over_mean
            max_prob_per_bin = multiplier / num_active_bins if num_active_bins > 0 else 1.0
        else:
            max_prob_per_bin = float(max_prob_per_bin_cfg) if max_prob_per_bin_cfg else 0.0

        if max_prob_per_bin > 0 and num_active_bins > 1.0 / max_prob_per_bin:
            prob = torch.clamp(prob, max=max_prob_per_bin)
            prob = prob / prob.sum()

    # max_prob_per_motion (clip)
    max_prob_per_motion_cfg = self.cfg.adaptive_max_prob_per_motion
    if max_prob_per_motion_cfg is not None:
        active_clip_ids = self.adp_samp_bins[self.adp_samp_active_motion_bins, 0]
        num_active_clips = len(active_clip_ids.unique())
        if max_prob_per_motion_cfg == "auto":
            multiplier = self.cfg.adaptive_failure_rate_max_over_mean
            max_prob_per_motion = (
                multiplier / num_active_clips if num_active_clips > 0 else 1.0
            )
        else:
            max_prob_per_motion = float(max_prob_per_motion_cfg) if max_prob_per_motion_cfg else 0.0

        if max_prob_per_motion > 0 and num_active_clips > 1.0 / max_prob_per_motion:
            unique_clips = active_clip_ids.unique()
            for clip_id in unique_clips:
                clip_mask = active_clip_ids == clip_id
                clip_total_prob = prob[clip_mask].sum()
                if clip_total_prob > max_prob_per_motion:
                    scale = max_prob_per_motion / clip_total_prob
                    prob[clip_mask] *= scale
            prob = prob / prob.sum()

    self.adp_sampling_active_prob = prob.float()

    # === metrics 更新 ===
    num_bins = len(self.adp_samp_active_motion_bins)
    H = -(prob * (prob + 1e-12).log()).sum()
    H_norm = H / math.log(num_bins) if num_bins > 1 else 1.0
    pmax, imax = prob.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / num_bins

    self.adp_sampling_active_prob = prob.float()
    assert (self.adp_sampling_active_prob >= 0).all()
    
  def _update_command(self):

    if self.cfg.sampling_mode == "adaptive":
        self._update_adaptive_sampling_stats()
        self._compute_adaptive_sampling_probabilities()

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
  adaptive_bin_size: int = 50
  adaptive_sequence_length_agnostic: bool = True
  adaptive_init_num_failures: float = 1.0
  adaptive_failure_rate_max_over_mean: float = 50.0
  adaptive_uniform_sampling_rate: float = 0.1
  adaptive_failure_counts_multiplier: int = 1
  adaptive_pre_failure_sample_window: int = 0
  adaptive_max_prob_per_bin: str | float | None = None
  adaptive_max_prob_per_motion: str | float | None = None
  adaptive_use_failure_rate_decay: bool = False
  adaptive_decay_gamma: float = 0.99
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)