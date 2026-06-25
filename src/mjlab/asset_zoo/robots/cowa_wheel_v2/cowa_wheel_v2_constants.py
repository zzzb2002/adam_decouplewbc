"""Cowa_wheel_v2 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

Cowa_wheel_v2_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "cowa_wheel_v2"
  / "xmls"
  / "wheel_v2_10dof_wo_arm_simplify_cylinder.xml"
)
assert Cowa_wheel_v2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, Cowa_wheel_v2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(Cowa_wheel_v2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Motor specs (from Cowa).
ROTOR_INERTIAS_hip = (
  0.0959e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_hip = (
  1,
  100,
  1,
)
ARMATURE_hip = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_hip, GEARS_hip
)

ROTOR_INERTIAS_knee = (
  0.489e-4,
  0.017e-4,
  0.533e-4,
)
GEARS_knee = (
  1,
  1,
  1,
)
ARMATURE_knee = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_knee, GEARS_knee
)

ROTOR_INERTIAS_foot = (
  0.215e-4,
  0.017e-4,
  0.738e-4,
)
GEARS_foot = (
  1,
  51,
  1,
)
ARMATURE_foot = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_foot, GEARS_foot
)

ROTOR_INERTIAS_wheel = (
  0.068e-4,
  0.068e-4,
  0.068e-4,
)
GEARS_wheel = (
  1,
  10,
  1,
)
ARMATURE_wheel = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_wheel, GEARS_wheel
)

ACTUATOR_hip = ElectricActuator(
  reflected_inertia=ARMATURE_hip,
  velocity_limit=5.0,
  effort_limit=105.0,
)
ACTUATOR_knee = ElectricActuator(
  reflected_inertia=ARMATURE_knee,
  velocity_limit=8.8,
  effort_limit=230.0,
)
ACTUATOR_foot = ElectricActuator(
  reflected_inertia=ARMATURE_foot,
  velocity_limit=9.0,
  effort_limit=58.0,
)
ACTUATOR_wheel = ElectricActuator(
  reflected_inertia=ARMATURE_wheel,
  velocity_limit=38.0,
  effort_limit=40.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_hip = ARMATURE_hip * NATURAL_FREQ**2
STIFFNESS_knee = ARMATURE_knee * NATURAL_FREQ**2
STIFFNESS_foot = ARMATURE_foot * NATURAL_FREQ**2
STIFFNESS_wheel = ARMATURE_wheel * NATURAL_FREQ**2

DAMPING_hip = 2.0 * DAMPING_RATIO * ARMATURE_hip * NATURAL_FREQ
DAMPING_knee = 2.0 * DAMPING_RATIO * ARMATURE_knee * NATURAL_FREQ
DAMPING_foot = 2.0 * DAMPING_RATIO * ARMATURE_foot * NATURAL_FREQ
DAMPING_wheel = 2.0 * DAMPING_RATIO * ARMATURE_wheel * NATURAL_FREQ

COWA_ACTUATOR_Hip = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
  ),
  stiffness=STIFFNESS_hip,
  damping=DAMPING_hip,
  effort_limit=ACTUATOR_hip.effort_limit,
  armature=ACTUATOR_hip.reflected_inertia,
)

COWA_ACTUATOR_Knee = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_knee_pitch_joint",
  ),
  stiffness=STIFFNESS_knee,
  damping=DAMPING_knee,
  effort_limit=ACTUATOR_knee.effort_limit,
  armature=ACTUATOR_knee.reflected_inertia,
)

COWA_ACTUATOR_FOOT = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_foot_joint",
  ),
  stiffness=STIFFNESS_foot,
  damping=DAMPING_foot,
  effort_limit=ACTUATOR_foot.effort_limit,
  armature=ACTUATOR_foot.reflected_inertia,
)

COWA_ACTUATOR_WHEEL = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wheel_joint",
  ),
  stiffness=0,  # D-only control: Kp=0 for foot walking mode
  damping=DAMPING_wheel,
  effort_limit=ACTUATOR_wheel.effort_limit,
  armature=ACTUATOR_wheel.reflected_inertia,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.40),
  joint_pos={
    ".*_hip_roll_joint": 0.0,
    ".*_hip_pitch_joint": 0.0,
    ".*_knee_pitch_joint": 0.0,
    ".*_wheel_joint": 0.0,
    ".*_foot_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-6]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-6]_collision$": 1},
  friction={r"^(left|right)_foot[1-6]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-6]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-6]_collision$": 1},
  friction={r"^(left|right)_foot[1-6]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-6]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)


##
# Final config.
##

COWA_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    COWA_ACTUATOR_Hip,
    COWA_ACTUATOR_Knee,
    COWA_ACTUATOR_WHEEL,
    COWA_ACTUATOR_FOOT,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_cowa_wheel_v2_robot_cfg() -> EntityCfg:
  """Get a fresh Cowa wheel v2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION_WITHOUT_SELF,),
    spec_fn=get_spec,
    articulation=COWA_ARTICULATION,
  )


COWA_ACTION_SCALE: dict[str, float] = {}
for a in COWA_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    if s == 0:
        # For D-only control (Kp=0), use damping-based scale
        d = a.damping
        COWA_ACTION_SCALE[n] = 0.25 * e / d if d > 0 else 0.1
    else:
        COWA_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_cowa_wheel_v2_robot_cfg())

  viewer.launch(robot.spec.compile())