"""Cowa Wheel V2 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  COWA_ACTION_SCALE,
  get_cowa_wheel_v2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def cowa_wheel_v2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Cowa Wheel V2 flat terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  # Use Cowa robot
  cfg.scene.entities = {"robot": get_cowa_wheel_v2_robot_cfg()}

  # Set raycast sensor frame to Cowa base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  # Cowa robot site names (defined in XML)
  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
      f"{side}_foot{i}_collision" 
      for side in ("left", "right") 
      for i in range(1, 7)
  )+ tuple(
      f"{side}_foot_actuator_collision" 
      for side in ("left", "right")
  )
  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  # Foot contact sensor - use left_foot_link and right_foot_link
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_foot_link|right_foot_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True
  # Set action scale
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = COWA_ACTION_SCALE

  # Viewer settings
  cfg.viewer.body_name = "base_link"

  # Command viz offset
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.4  # Adjust to Cowa height

  # Event configs
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # Reward configs - adjust for Cowa robot
  # Cowa joints: hip_roll, hip_pitch, knee_pitch, wheel, foot
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.1}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_roll.*": 0.1,
    r".*hip_pitch.*": 0.2,
    r".*knee_pitch.*": 0.3,
    r".*wheel_joint": 0.5,
    r".*foot_joint": 0.2,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_roll.*": 0.15,
    r".*hip_pitch.*": 0.3,
    r".*knee_pitch.*": 0.5,
    r".*wheel_joint": 1.0,
    r".*foot_joint": 0.3,
  }

  # Upright - use base_link
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Adjust reward weights for Cowa
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.5 #0.0

  # Penalize wheel velocity to encourage foot walking instead of wheel rolling
  cfg.rewards["wheel_velocity"] = RewardTermCfg(
    func=envs_mdp.joint_vel_l2,
    weight=-0.5,
    params={
      "asset_cfg": SceneEntityCfg(
        name="robot",
        joint_names=[".*_wheel_joint"],
      ),
    },
  )

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def cowa_wheel_v2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Cowa_wheel_v2 flat terrain velocity configuration."""
  cfg = cowa_wheel_v2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  # cfg.observations["actor"].terms.pop("foot_height", None)
  # cfg.observations["critic"].terms.pop("foot_height", None)
  # cfg.rewards.pop("feet_swing_height", None)

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
