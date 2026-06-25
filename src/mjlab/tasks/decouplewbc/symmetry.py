"""Symmetry augmentation helpers for DecoupleWBC observations and actions."""

from __future__ import annotations

from math import prod
from typing import Any

import torch

from mjlab.sensor import GridPatternCfg


def get_symmetric_states(
  obs: torch.Tensor | None = None,
  actions: torch.Tensor | None = None,
  env: Any | None = None,
  obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
  """Return original plus mirrored observations/actions for PPO augmentation.

  The local RSL-RL PPO expects the first half of the returned batch to be the
  original samples and the second half to be their mirrored counterparts.
  """
  if env is None:
    raise ValueError("DecoupleWBC symmetry augmentation requires env.")

  if obs is not None:
    mirrored_obs = _mirror_observations(obs, env, obs_type)
    aug_obs = torch.cat((obs, mirrored_obs), dim=0)
  else:
    aug_obs = None

  if actions is not None:
    mirrored_actions = _mirror_actions(actions, env)
    aug_actions = torch.cat((actions, mirrored_actions), dim=0)
  else:
    aug_actions = None
  return aug_obs, aug_actions


def _mirror_observations(
  obs: torch.Tensor,
  env: Any,
  obs_type: str,
) -> torch.Tensor:
  env = _unwrap_env(env)
  group_name = "actor" if obs_type == "policy" else obs_type
  slices = _observation_slices(env, group_name)
  expected_width = max((slc.stop for slc in slices.values()), default=0)
  if obs.shape[1] != expected_width:
    raise ValueError(
      f"Observation width mismatch for group '{group_name}': expected"
      f" {expected_width}, got {obs.shape[1]}."
    )
  mirrored = obs.clone()

  if "base_ang_vel" in slices:
    slc = slices["base_ang_vel"]
    mirrored[:, slc] = _mirror_history_vectors(obs[:, slc], vector_dim=3, kind="axial")

  if "projected_gravity" in slices:
    slc = slices["projected_gravity"]
    mirrored[:, slc] = _mirror_history_vectors(obs[:, slc], vector_dim=3, kind="polar")

  if "joint_pos" in slices:
    slc = slices["joint_pos"]
    names = _joint_names_for_width(env, slc.stop - slc.start)
    mirrored[:, slc] = _mirror_named_values(obs[:, slc], names)

  if "joint_vel" in slices:
    slc = slices["joint_vel"]
    names = _joint_names_for_width(env, slc.stop - slc.start)
    mirrored[:, slc] = _mirror_named_values(obs[:, slc], names)

  if "actions" in slices:
    slc = slices["actions"]
    names = _action_names(env, slc.stop - slc.start)
    mirrored[:, slc] = _mirror_named_values(obs[:, slc], names)

  if "command" in slices:
    slc = slices["command"]
    mirrored[:, slc] = _mirror_command(obs[:, slc])

  if "height_scan" in slices:
    slc = slices["height_scan"]
    mirrored[:, slc] = _mirror_height_scan(obs[:, slc], env)

  for term_name in ("foot_height", "foot_air_time", "foot_contact"):
    if term_name in slices:
      slc = slices[term_name]
      mirrored[:, slc] = _swap_left_right_blocks(obs[:, slc], block_width=1)

  if "foot_contact_forces" in slices:
    slc = slices["foot_contact_forces"]
    mirrored[:, slc] = _mirror_foot_forces(obs[:, slc])

  return mirrored


def _mirror_actions(actions: torch.Tensor, env: Any) -> torch.Tensor:
  env = _unwrap_env(env)
  names = _action_names(env, actions.shape[1])
  return _mirror_named_values(actions, names)


def _observation_slices(env: Any, group_name: str) -> dict[str, slice]:
  manager = env.observation_manager
  if group_name not in manager.active_terms:
    raise ValueError(
      f"Observation group '{group_name}' not found. Available groups:"
      f" {list(manager.active_terms.keys())}."
    )
  if not manager.group_obs_concatenate[group_name]:
    raise ValueError("Symmetry augmentation requires concatenated observations.")

  slices: dict[str, slice] = {}
  start = 0
  for term_name, term_dim in zip(
    manager.active_terms[group_name],
    manager.group_obs_term_dim[group_name],
    strict=True,
  ):
    width = int(prod(term_dim))
    slices[term_name] = slice(start, start + width)
    start += width
  return slices


def _unwrap_env(env: Any) -> Any:
  return env.unwrapped if hasattr(env, "unwrapped") else env


def _joint_names_for_width(env: Any, width: int) -> list[str]:
  robot = env.scene["robot"]
  joint_names = list(robot.joint_names)
  if len(joint_names) > 0 and width % len(joint_names) == 0:
    return joint_names

  action_names = _action_names(env)
  if len(action_names) > 0 and width % len(action_names) == 0:
    return action_names

  raise ValueError(
    f"Cannot infer joint names for observation width {width}. Robot has"
    f" {len(joint_names)} joints and action space has {len(action_names)} targets."
  )


def _action_names(env: Any, width: int | None = None) -> list[str]:
  names: list[str] = []
  for term_name in env.action_manager.active_terms:
    term = env.action_manager.get_term(term_name)
    if hasattr(term, "target_names"):
      names.extend(list(term.target_names))
    else:
      names.extend(f"{term_name}_{i}" for i in range(term.action_dim))

  if width is not None and (len(names) == 0 or width % len(names) != 0):
    raise ValueError(
      f"Cannot mirror action width {width}; action manager exposes"
      f" {len(names)} targets."
    )
  return names


def _mirror_named_values(values: torch.Tensor, names: list[str]) -> torch.Tensor:
  if len(names) == 0:
    raise ValueError("Cannot mirror named values without names.")
  if values.shape[1] % len(names) != 0:
    raise ValueError(
      f"Value width {values.shape[1]} is not a multiple of named columns"
      f" ({len(names)})."
    )

  history = values.shape[1] // len(names)
  if history > 1:
    values_view = values.reshape(values.shape[0], history, len(names))
    mirrored = _mirror_named_frame(values_view, names)
    return mirrored.reshape_as(values)
  return _mirror_named_frame(values.unsqueeze(1), names).squeeze(1)


def _mirror_named_frame(values: torch.Tensor, names: list[str]) -> torch.Tensor:
  name_to_idx = {name: idx for idx, name in enumerate(names)}
  mirrored = values.clone()
  for target_idx, target_name in enumerate(names):
    source_name = _opposite_side_name(target_name)
    source_idx = name_to_idx.get(source_name, target_idx)
    mirrored[..., target_idx] = values[..., source_idx] * _joint_sign(target_name)
  return mirrored


def _opposite_side_name(name: str) -> str:
  replacements = (
    ("left_", "right_"),
    ("right_", "left_"),
    ("_left_", "_right_"),
    ("_right_", "_left_"),
    ("_left", "_right"),
    ("_right", "_left"),
    ("left", "right"),
    ("right", "left"),
    ("Left", "Right"),
    ("Right", "Left"),
  )
  for old, new in replacements:
    if old in name:
      return name.replace(old, new, 1)
  return name


def _joint_sign(name: str) -> float:
  lower = name.lower()
  if "roll" in lower or "yaw" in lower:
    return -1.0
  return 1.0


def _mirror_history_vectors(
  values: torch.Tensor,
  vector_dim: int,
  kind: str,
) -> torch.Tensor:
  if values.shape[1] % vector_dim != 0:
    raise ValueError(
      f"Vector observation width {values.shape[1]} is not divisible by"
      f" {vector_dim}."
    )
  vectors = values.reshape(values.shape[0], -1, vector_dim).clone()
  if kind == "polar":
    vectors[..., 1] *= -1.0
  elif kind == "axial":
    vectors[..., 0] *= -1.0
    vectors[..., 2] *= -1.0
  else:
    raise ValueError(f"Unknown mirror vector kind: {kind}")
  return vectors.reshape_as(values)


def _mirror_command(command: torch.Tensor) -> torch.Tensor:
  if command.shape[1] % 4 != 0:
    raise ValueError(
      f"DecoupleWBC command width should be a multiple of 4, got"
      f" {command.shape[1]}."
    )
  command_view = command.reshape(command.shape[0], -1, 4).clone()
  command_view[..., 1] *= -1.0
  command_view[..., 2] *= -1.0
  return command_view.reshape_as(command)


def _mirror_height_scan(height_scan: torch.Tensor, env: Any) -> torch.Tensor:
  sensor = env.scene["terrain_scan"]
  pattern = sensor.cfg.pattern
  if not isinstance(pattern, GridPatternCfg):
    return height_scan

  num_rays = int(sensor.num_rays_per_frame)
  num_frames = int(sensor.num_frames)
  if num_rays <= 0 or num_frames <= 0:
    raise ValueError("Terrain scan sensor exposes no rays or frames.")
  if height_scan.shape[1] % (num_frames * num_rays) != 0:
    raise ValueError(
      f"Height scan width {height_scan.shape[1]} is not divisible by"
      f" {num_frames * num_rays} sensor columns."
    )

  size_x, size_y = pattern.size
  resolution = pattern.resolution
  num_x = int(round(size_x / resolution)) + 1
  num_y = int(round(size_y / resolution)) + 1
  if num_x * num_y != num_rays:
    raise ValueError(
      f"Terrain scan grid shape {num_y}x{num_x} does not match sensor rays"
      f" ({num_rays})."
    )

  scan_view = height_scan.reshape(
    height_scan.shape[0],
    -1,
    num_frames,
    num_y,
    num_x,
  )
  return scan_view.flip(dims=(-2,)).reshape_as(height_scan)


def _swap_left_right_blocks(values: torch.Tensor, block_width: int) -> torch.Tensor:
  if values.shape[1] % (2 * block_width) != 0:
    raise ValueError(
      f"Cannot left/right swap width {values.shape[1]} with block width"
      f" {block_width}."
    )
  blocks = values.reshape(values.shape[0], -1, 2, block_width)
  swapped = blocks[:, :, [1, 0], :].reshape_as(values)
  return swapped


def _mirror_foot_forces(forces: torch.Tensor) -> torch.Tensor:
  if forces.shape[1] % 6 != 0:
    raise ValueError(
      f"Foot contact force width should be a multiple of 6, got"
      f" {forces.shape[1]}."
    )
  force_view = forces.reshape(forces.shape[0], -1, 2, 3)
  mirrored = force_view[:, :, [1, 0], :].clone()
  mirrored[..., 1] *= -1.0
  return mirrored.reshape_as(forces)
