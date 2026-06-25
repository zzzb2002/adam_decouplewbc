"""Domain randomization functions for actuators and special entity-level DR."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from mjlab.actuator import BuiltinPositionActuator, IdealPdActuator, XmlPositionActuator
from mjlab.actuator.actuator import Actuator
from mjlab.actuator.delayed_actuator import DelayedActuator
from mjlab.entity import Entity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ._core import _DEFAULT_ASSET_CFG
from ._types import resolve_distribution

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _resolve_actuator_groups(
  asset: Entity, actuator_ids: list[int] | slice | int
) -> list[Actuator]:
  """Resolve SceneEntityCfg actuator IDs to unique Actuator groups.

  ``SceneEntityCfg.actuator_ids`` are indices into ``entity.actuator_names``
  (derived from ``spec.actuators``), which lists every individual MjsActuator.
  ``entity.actuators`` is a shorter list of high-level Actuator groups, where
  each group owns one or more MjsActuators. This helper bridges the two
  namespaces, returning the groups that own the selected MjsActuators in spec
  order with duplicates removed.
  """
  mjs_to_group: dict[int, Actuator] = {}
  for group in asset.actuators:
    for mjs_act in group._mjs_actuators:
      mjs_to_group[mjs_act.id] = group

  if isinstance(actuator_ids, list):
    selected_mjs = [asset.spec.actuators[i] for i in actuator_ids]
  elif isinstance(actuator_ids, slice):
    selected_mjs = asset.spec.actuators[actuator_ids]
  else:
    selected_mjs = [asset.spec.actuators[actuator_ids]]

  seen: set[int] = set()
  groups: list[Actuator] = []
  for mjs_act in selected_mjs:
    group = mjs_to_group.get(mjs_act.id)
    if group is not None and id(group) not in seen:
      seen.add(id(group))
      groups.append(group)
  return groups


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def pd_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Randomize PD stiffness and damping gains.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all.
    kp_range: (min, max) for proportional gain randomization.
    kd_range: (min, max) for derivative gain randomization.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
    operation: "scale" multiplies default gains by sampled values, "abs" sets
      absolute values.
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  actuators = _resolve_actuator_groups(asset, asset_cfg.actuator_ids)

  actuators = [
    a.base_actuator if isinstance(a, DelayedActuator) else a for a in actuators
  ]

  for actuator in actuators:
    ctrl_ids = actuator.global_ctrl_ids

    dist = resolve_distribution(distribution)
    kp_samples = dist.sample(
      torch.tensor(kp_range[0], device=env.device),
      torch.tensor(kp_range[1], device=env.device),
      (len(env_ids), len(ctrl_ids)),
      env.device,
    )
    kd_samples = dist.sample(
      torch.tensor(kd_range[0], device=env.device),
      torch.tensor(kd_range[1], device=env.device),
      (len(env_ids), len(ctrl_ids)),
      env.device,
    )

    if isinstance(actuator, (BuiltinPositionActuator, XmlPositionActuator)):
      if operation == "scale":
        default_gainprm = env.sim.get_default_field("actuator_gainprm")
        default_biasprm = env.sim.get_default_field("actuator_biasprm")
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
          default_gainprm[ctrl_ids, 0] * kp_samples
        )
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = (
          default_biasprm[ctrl_ids, 1] * kp_samples
        )
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = (
          default_biasprm[ctrl_ids, 2] * kd_samples
        )
      elif operation == "abs":
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = kp_samples
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = -kp_samples
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = -kd_samples

    elif isinstance(actuator, IdealPdActuator):
      assert actuator.stiffness is not None
      assert actuator.damping is not None
      if operation == "scale":
        assert actuator.default_stiffness is not None
        assert actuator.default_damping is not None
        actuator.set_gains(
          env_ids,
          kp=actuator.default_stiffness[env_ids] * kp_samples,
          kd=actuator.default_damping[env_ids] * kd_samples,
        )
      elif operation == "abs":
        actuator.set_gains(env_ids, kp=kp_samples, kd=kd_samples)

    else:
      raise TypeError(
        f"pd_gains only supports BuiltinPositionActuator, "
        f"XmlPositionActuator, and IdealPdActuator (optionally wrapped "
        f"with DelayedActuator), got {type(actuator).__name__}"
      )


@requires_model_fields("actuator_forcerange")
def effort_limits(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  effort_limit_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Randomize actuator effort limits.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all.
    effort_limit_range: (min, max) for effort limit randomization.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
    operation: "scale" multiplies default limits, "abs" sets absolute values.
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  actuators = _resolve_actuator_groups(asset, asset_cfg.actuator_ids)

  for actuator in actuators:
    ctrl_ids = actuator.global_ctrl_ids
    num_actuators = len(ctrl_ids)

    dist = resolve_distribution(distribution)
    effort_samples = dist.sample(
      torch.tensor(effort_limit_range[0], device=env.device),
      torch.tensor(effort_limit_range[1], device=env.device),
      (len(env_ids), num_actuators),
      env.device,
    )

    if isinstance(actuator, (BuiltinPositionActuator, XmlPositionActuator)):
      if operation == "scale":
        default_forcerange = env.sim.get_default_field("actuator_forcerange")
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] = (
          default_forcerange[ctrl_ids, 0] * effort_samples
        )
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
          default_forcerange[ctrl_ids, 1] * effort_samples
        )
      elif operation == "abs":
        env.sim.model.actuator_forcerange[
          env_ids[:, None], ctrl_ids, 0
        ] = -effort_samples
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
          effort_samples
        )

    elif isinstance(actuator, IdealPdActuator):
      assert actuator.force_limit is not None
      if operation == "scale":
        assert actuator.default_force_limit is not None
        actuator.set_effort_limit(
          env_ids,
          effort_limit=actuator.default_force_limit[env_ids] * effort_samples,
        )
      elif operation == "abs":
        actuator.set_effort_limit(env_ids, effort_limit=effort_samples)

    else:
      raise TypeError(
        f"effort_limits only supports BuiltinPositionActuator, "
        f"XmlPositionActuator, and IdealPdActuator, "
        f"got {type(actuator).__name__}"
      )


def sync_actuator_delays(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  lag_range: tuple[int, int],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Synchronize delay lags across all delayed actuators.

  Samples a single lag value per environment and applies it to all delayed
  actuators.

  Args:
    env: The environment.
    env_ids: Environment IDs to set. If None, sets all environments.
    lag_range: (min_lag, max_lag) range for sampling lag values.
    asset_cfg: Asset configuration specifying which entity and actuators.
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)

  actuators = _resolve_actuator_groups(asset, asset_cfg.actuator_ids)

  delayed_actuators = [a for a in actuators if isinstance(a, DelayedActuator)]

  if not delayed_actuators:
    return

  lags = torch.randint(
    lag_range[0],
    lag_range[1] + 1,
    (len(env_ids),),
    device=env.device,
    dtype=torch.long,
  )

  for actuator in delayed_actuators:
    actuator.set_lags(lags, env_ids)


@requires_model_fields("actuator_gear")
def motor_efficiency(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  efficiency_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomize motor efficiency by scaling actuator gear ratio.

  Scales actuator_gear[:, 0] by a per-env, per-actuator factor sampled
  uniformly from efficiency_range. This models the loss between commanded
  and actual torque: actual_torque = efficiency * commanded_torque.
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  actuators = _resolve_actuator_groups(asset, asset_cfg.actuator_ids)
  actuators = [
    a.base_actuator if isinstance(a, DelayedActuator) else a for a in actuators
  ]

  default_gear = env.sim.get_default_field("actuator_gear")  # (nu, 6)

  for actuator in actuators:
    ctrl_ids = actuator.global_ctrl_ids
    n_envs = len(env_ids)
    n_ctrls = len(ctrl_ids)

    eff_samples = (
      torch.rand(n_envs, n_ctrls, device=env.device)
      * (efficiency_range[1] - efficiency_range[0])
      + efficiency_range[0]
    )

    env.sim.model.actuator_gear[env_ids[:, None], ctrl_ids, 0] = (
      default_gear[ctrl_ids, 0] * eff_samples
    )
