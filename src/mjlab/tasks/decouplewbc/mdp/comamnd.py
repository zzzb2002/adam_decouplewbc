from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformDecouplewbcCommand(CommandTerm):
  cfg: UniformDecouplewbcCommandCfg

  def __init__(self, cfg: UniformDecouplewbcCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)
    self.vel_command_w = torch.zeros(self.num_envs, 4, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_world_env = torch.zeros_like(self.is_heading_env)
    self.is_forward_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )
    self.metrics["error_height"] += (
      torch.abs(self.vel_command_b[:, 3] - self.robot.data.root_link_pos_w[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # if self.cfg.keyboard_control:
    #   self.is_heading_env[env_ids] = False
    #   self.is_standing_env[env_ids] = False
    #   self.is_world_env[env_ids] = False
    #   self.is_forward_env[env_ids] = False
    #   self.vel_command_b[env_ids, :3] = 0.0
    #   self.vel_command_w[env_ids, :3] = 0.0
    #   height_low, height_high = self._height_range_abs()
    #   initial_height = self.cfg.keyboard_initial_height
    #   if initial_height is None:
    #     initial_height = self.cfg.default_base_height
    #   initial_height = min(max(initial_height, height_low), height_high)
    #   self.vel_command_b[env_ids, 3] = initial_height
    #   self.vel_command_w[env_ids, 3] = initial_height
    #   return

    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    height_low, height_high = self._height_range_abs()
    if height_low == height_high:
      self.vel_command_b[env_ids, 3] = height_low
    else:
      self.vel_command_b[env_ids, 3] = r.uniform_(height_low, height_high)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # Randomly assign world-frame envs.
    self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
    # Copy sampled velocities as world-frame reference for world envs.
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    # Forward-only envs: positive lin_vel_x, zero lateral and angular.
    self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    fwd_ids = env_ids[self.is_forward_env[env_ids]]
    if len(fwd_ids) > 0:
      min_forward = max(0.0, self.cfg.ranges.lin_vel_x[0])
      max_forward = max(0.0, self.cfg.ranges.lin_vel_x[1])
      if max_forward > 0.0:
        min_forward = min(max(0.3, min_forward), max_forward)
        self.vel_command_b[fwd_ids, 0] = self.vel_command_b[fwd_ids, 0].abs().clamp(
          min=min_forward,
          max=max_forward,
        )
      else:
        self.vel_command_b[fwd_ids, 0] = 0.0
      self.vel_command_b[fwd_ids, 1] = 0.0
      self.vel_command_b[fwd_ids, 2] = 0.0

    self._zero_velocity_below_min_height(env_ids)

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    # World-frame envs: rotate world-frame linear vel into body frame.
    if self.is_world_env.any():
      w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
      heading = self.robot.data.heading_w[w_ids]
      cos_h = torch.cos(heading)
      sin_h = torch.sin(heading)
      vx_w = self.vel_command_w[w_ids, 0]
      vy_w = self.vel_command_w[w_ids, 1]
      self.vel_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
      self.vel_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :3] = 0.0
    self.vel_command_w[standing_env_ids, :3] = 0.0
    self._zero_velocity_below_min_height()

  def _zero_velocity_below_min_height(self, env_ids: torch.Tensor | None = None) -> None:
    if self.cfg.min_height_for_velocity is None:
      return
    if env_ids is None:
      low_height_env_ids = (
        self.vel_command_b[:, 3] < self.cfg.min_height_for_velocity
      ).nonzero(as_tuple=False).flatten()
    else:
      low_height_env_ids = env_ids[
        self.vel_command_b[env_ids, 3] < self.cfg.min_height_for_velocity
      ]
    if len(low_height_env_ids) == 0:
      return
    self.vel_command_b[low_height_env_ids, :3] = 0.0
    self.vel_command_w[low_height_env_ids, :3] = 0.0

  def _height_range_abs(self) -> tuple[float, float]:
    low, high = self.cfg.ranges.target_height
    return (
      min(low, high) + self.cfg.default_base_height,
      max(low, high) + self.cfg.default_base_height,
    )

  def _velocity_range(self, axis: int) -> tuple[float, float]:
    ranges = (
      self.cfg.ranges.lin_vel_x,
      self.cfg.ranges.lin_vel_y,
      self.cfg.ranges.ang_vel_z,
    )
    low, high = ranges[axis]
    return min(low, high), max(low, high)

  def _set_keyboard_command(self, env_idx: int, axis: int, value: float) -> None:
    if axis < 3:
      low, high = self._velocity_range(axis)
    else:
      low, high = self._height_range_abs()
    value = min(max(value, low), high)
    self.vel_command_b[env_idx, axis] = value
    self.vel_command_w[env_idx, axis] = value
    self._zero_velocity_below_min_height(
      torch.tensor([env_idx], device=self.device, dtype=torch.long)
    )

  def handle_keyboard_command(self, key: int, env_idx: int) -> bool:
    """Handle native viewer keyboard control for play mode."""
    if not self.cfg.keyboard_control:
      return False

    from mjlab.viewer.native.keys import (
      KEY_E,
      KEY_J,
      KEY_L,
      KEY_O,
      KEY_Q,
      KEY_S,
      KEY_U,
      KEY_W,
      KEY_X,
    )

    if env_idx < 0 or env_idx >= self.num_envs:
      return False

    if key == KEY_W:
      self._set_keyboard_command(
        env_idx,
        0,
        float(self.vel_command_b[env_idx, 0]) + self.cfg.keyboard_lin_vel_step,
      )
    elif key == KEY_S:
      self._set_keyboard_command(
        env_idx,
        0,
        float(self.vel_command_b[env_idx, 0]) - self.cfg.keyboard_lin_vel_step,
      )
    elif key == KEY_J:
      self._set_keyboard_command(
        env_idx,
        1,
        float(self.vel_command_b[env_idx, 1]) + self.cfg.keyboard_lin_vel_step,
      )
    elif key == KEY_L:
      self._set_keyboard_command(
        env_idx,
        1,
        float(self.vel_command_b[env_idx, 1]) - self.cfg.keyboard_lin_vel_step,
      )
    elif key == KEY_Q:
      self._set_keyboard_command(
        env_idx,
        2,
        float(self.vel_command_b[env_idx, 2]) + self.cfg.keyboard_ang_vel_step,
      )
    elif key == KEY_E:
      self._set_keyboard_command(
        env_idx,
        2,
        float(self.vel_command_b[env_idx, 2]) - self.cfg.keyboard_ang_vel_step,
      )
    elif key == KEY_U:
      self._set_keyboard_command(
        env_idx,
        3,
        float(self.vel_command_b[env_idx, 3]) + self.cfg.keyboard_height_step,
      )
    elif key == KEY_O:
      self._set_keyboard_command(
        env_idx,
        3,
        float(self.vel_command_b[env_idx, 3]) - self.cfg.keyboard_height_step,
      )
    elif key == KEY_X:
      self._set_keyboard_command(env_idx, 0, 0.0)
      self._set_keyboard_command(env_idx, 1, 0.0)
      self._set_keyboard_command(env_idx, 2, 0.0)
    else:
      return False
    return True

  # GUI.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    def _slider_bounds(min_val: float, max_val: float) -> tuple[float, float]:
      if min_val == max_val:
        return min_val - 0.05, max_val + 0.05
      return min_val, max_val

    def _clamp(value: float, min_val: float, max_val: float) -> float:
      return min(max(value, min_val), max_val)

    lin_vel_x_min, lin_vel_x_max = _slider_bounds(-ranges.lin_vel_x[1], ranges.lin_vel_x[1])
    lin_vel_y_min, lin_vel_y_max = _slider_bounds(-ranges.lin_vel_y[1], ranges.lin_vel_y[1])
    ang_vel_z_min, ang_vel_z_max = _slider_bounds(-ranges.ang_vel_z[1], ranges.ang_vel_z[1])
    target_height_min, target_height_max = _slider_bounds(
      ranges.target_height[0] + self.cfg.default_base_height,
      ranges.target_height[1] + self.cfg.default_base_height,
    )
    target_height_initial = self.cfg.keyboard_initial_height
    if target_height_initial is None:
      target_height_initial = self.cfg.default_base_height
    target_height_initial = _clamp(
      target_height_initial,
      target_height_min,
      target_height_max,
    )

    axes = [
      ("lin_vel_x", 0.0, lin_vel_x_min, lin_vel_x_max),
      ("lin_vel_y", 0.0, lin_vel_y_min, lin_vel_y_max),
      ("ang_vel_z", 0.0, ang_vel_z_min, ang_vel_z_max),
      (
        "target_height",
        target_height_initial,
        target_height_min,
        target_height_max,
      ),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, initial_value, min_val, max_val in axes:
        slider = server.gui.add_slider(
          label,
          min=min_val,
          max=max_val,
          step=0.05,
          initial_value=initial_value,
        )

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders[:3]:
          s.value = 0.0
        sliders[3].value = target_height_initial

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformDecouplewbcCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  rel_world_envs: float = 0.0
  default_base_height: float = 0.9
  min_height_for_velocity: float | None = None
  """If set, commands below this absolute base-height target get zero xy/yaw velocity."""
  keyboard_control: bool = False
  keyboard_initial_height: float | None = None
  keyboard_lin_vel_step: float = 0.1
  keyboard_ang_vel_step: float = 0.1
  keyboard_height_step: float = 0.02
  """Fraction of environments that use world-frame velocity commands.
  World-frame envs sample linear velocity in world frame and rotate to body
  frame each step, so the command direction stays fixed in the world."""
  rel_forward_envs: float = 0.0
  """Fraction of environments that receive forward-only commands (positive
  lin_vel_x, zero lin_vel_y and ang_vel_z). Increases training coverage for
  straight-line walking, which is important for stair climbing."""
  init_velocity_prob: float = 0.0

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    target_height :tuple[float,float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformDecouplewbcCommand:
    return UniformDecouplewbcCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
