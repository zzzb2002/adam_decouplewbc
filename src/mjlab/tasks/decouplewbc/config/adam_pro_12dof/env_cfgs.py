"""ADAM robot velocity environment configurations."""
from mjlab.asset_zoo.robots import (
  ADAM_PRO_12DOF_ACTION_SCALE,
  get_adam_pro_12dof_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.envs.mdp import dr
from mjlab.tasks.decouplewbc import mdp
from mjlab.tasks.decouplewbc.mdp import UniformDecouplewbcCommandCfg
from mjlab.tasks.decouplewbc.decouplewbc_env_cfg import make_decouplewbc_env_cfg


def _set_play_command(
  cmd: UniformDecouplewbcCommandCfg,
  *,
  lin_vel_x: tuple[float, float] = (-0.8, 0.8),
  lin_vel_y: tuple[float, float] = (-0.8, 0.8),
  ang_vel_z: tuple[float, float] = (-0.5, 0.5),
  base_height: tuple[float, float] = (0.6, 0.9),
  initial_height: float | None = None,
) -> None:
  if base_height is None:
    base_height = (cmd.default_base_height, cmd.default_base_height)
  if initial_height is None:
    initial_height = cmd.default_base_height

  cmd.resampling_time_range = (1e9, 1e9)
  cmd.ranges.lin_vel_x = lin_vel_x
  cmd.ranges.lin_vel_y = lin_vel_y
  cmd.ranges.ang_vel_z = ang_vel_z
  # base_height is an absolute world-z range; command ranges store offsets from
  # default_base_height.
  cmd.ranges.target_height = (
    base_height[0] - cmd.default_base_height,
    base_height[1] - cmd.default_base_height,
  )
  cmd.heading_command = False
  cmd.ranges.heading = None
  cmd.rel_heading_envs = 0.0
  cmd.rel_forward_envs = 0.0
  cmd.rel_standing_envs = 0.0
  cmd.rel_world_envs = 0.0
  cmd.keyboard_control = True
  cmd.keyboard_initial_height = initial_height


def adam_pro_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Adam rough terrain velocity configuration."""
  cfg = make_decouplewbc_env_cfg()


  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  # Register Adam robot as the scene entity.
  cfg.scene.entities = {"robot": get_adam_pro_12dof_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  feet_parallel_site_names = (
    "parallel_marker_left_rear",
    "parallel_marker_left_mid",
    "parallel_marker_left_front",
    "parallel_marker_right_rear",
    "parallel_marker_right_mid",
    "parallel_marker_right_front",
  )
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
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
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
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

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ADAM_PRO_12DOF_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["decouplewbc"]
  assert isinstance(twist_cmd, UniformDecouplewbcCommandCfg)
  twist_cmd.viz.z_offset = 2.0
  cfg.observations["actor"].history_length = 5
  cfg.observations["critic"].history_length = 5

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  # Randomize all body COM offsets (torso_link keeps original ranges; others +/-0.02m).
  all_body_cfg = SceneEntityCfg("robot", body_names=(".*",))
  cfg.events["base_com"].params["asset_cfg"] = all_body_cfg
  cfg.events["base_com"].params["ranges"] = {
    "torso_link": {0: (-0.08, 0.08), 1: (-0.08, 0.08), 2: (-0.1, 0.1)},
    "(?!torso_link).*": (-0.02, 0.02),
  }
  cfg.events["foot_friction"].mode = "reset"
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["base_com"].mode = "reset"
  cfg.events["encoder_bias"].mode = "reset"

  cfg.events["body_mass"] = EventTermCfg(
    mode="reset",
    func=dr.body_mass,
    params={
      "asset_cfg": all_body_cfg,
      "operation": "scale",
      "ranges": (0.8, 1.2),
    },
  )
  cfg.events["body_inertia"] = EventTermCfg(
    mode="reset",
    func=dr.body_inertia,
    params={
      "asset_cfg": all_body_cfg,
      "operation": "scale",
      "ranges": (0.8, 1.2),
    },
  )
  # Joint parameter randomization.
  all_joint_cfg = SceneEntityCfg("robot")
  cfg.events["joint_friction"] = EventTermCfg(
    mode="reset",
    func=dr.joint_friction,
    params={
      "asset_cfg": all_joint_cfg,
      "ranges": (0.8, 1.2),
      "operation": "scale",
    },
  )
  cfg.events["joint_damping"] = EventTermCfg(
    mode="reset",
    func=dr.joint_damping,
    params={
      "asset_cfg": all_joint_cfg,
      "ranges": (0.8, 1.2),
      "operation": "scale",
    },
  )
  cfg.events["joint_armature"] = EventTermCfg(
    mode="reset",
    func=dr.joint_armature,
    params={
      "asset_cfg": all_joint_cfg,
      "ranges": (0.8, 1.2),
      "operation": "scale",
    },
  )

  cfg.events["pd_gains"] = EventTermCfg(
    mode="reset",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
      "operation": "scale",
    },
  )
  cfg.events["actuator_delay"] = EventTermCfg(
    mode="reset",
    func=dr.sync_actuator_delays,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
      "lag_range": (0, 2),
    },
  )
  cfg.events["motor_efficiency"] = EventTermCfg(
    mode="reset",
    func=dr.motor_efficiency,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "efficiency_range": (0.8, 1.0),
    },
  )
  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.05,
    r".*hip_yaw.*": 0.1,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.05,
    # # Waist.
    # r".*waist_yaw.*": 0.2,
    # r".*waist_roll.*": 0.08,
    # r".*waist_pitch.*": 0.1,
    # # Arms.
    # r".*shoulder_pitch.*": 0.15,
    # r".*shoulder_roll.*": 0.15,
    # r".*shoulder_yaw.*": 0.1,
    # r".*elbow.*": 0.15,
    # r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # # Waist.
    # r".*waist_yaw.*": 0.3,
    # r".*waist_roll.*": 0.08,
    # r".*waist_pitch.*": 0.2,
    # # Arms.
    # r".*shoulder_pitch.*": 0.5,
    # r".*shoulder_roll.*": 0.2,
    # r".*shoulder_yaw.*": 0.15,
    # r".*elbow.*": 0.35,
    # r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.05
  cfg.rewards["hip_yaw_default"] = RewardTermCfg(
    func=mdp.default_joint_position_l2_when_moving,
    weight=-0.2,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(r".*hip_yaw_joint",),
      ),
      "command_name": "decouplewbc",
      "command_threshold": 0.05,
    },
  )
  cfg.rewards["ankle_roll_default"] = RewardTermCfg(
    func=mdp.default_joint_position_l2_when_moving,
    weight=-0.5,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(r".*ankle_roll_joint",),
      ),
      "command_name": "decouplewbc",
      "command_threshold": 0.05,
    },
  )

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["feet_parallel"] = RewardTermCfg(
    func=mdp.feet_parallel,
    weight=-3.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        site_names=feet_parallel_site_names,
        preserve_order=True,
      ),
      "command_name": "decouplewbc",
      "height_threshold": twist_cmd.min_height_for_velocity
      if twist_cmd.min_height_for_velocity is not None
      else twist_cmd.default_base_height,
    },
  )
  cfg.rewards["foot_parallel_ground"] = RewardTermCfg(
    func=mdp.foot_parallel_ground,
    weight=-2.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "asset_cfg": SceneEntityCfg(
        "robot",
        site_names=feet_parallel_site_names,
        preserve_order=True,
      ),
      "command_name": "decouplewbc",
      "command_threshold": 0.05,
      "min_air_time": 0.05,
      "min_contact_steps": 3,
    },
  )
  cfg.rewards["knee_distance_lateral"] = RewardTermCfg(
    func=mdp.knee_distance_lateral,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=(
          "left_hip_yaw_link",
          "left_knee_link",
          "right_hip_yaw_link",
          "right_knee_link",
        ),
        preserve_order=True,
      ),
      "command_name": "decouplewbc",
      "least_knee_distance_lateral": 0.2,
      "most_knee_distance_lateral": 0.35,
      "height_threshold": twist_cmd.default_base_height - 0.005,
    },
  )

  # Apply play mode overrides.
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

    _set_play_command(twist_cmd)

  return cfg


def adam_pro_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Adam flat terrain velocity configuration."""
  cfg = adam_pro_rough_env_cfg(play=play)
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

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["decouplewbc"]
    assert isinstance(twist_cmd, UniformDecouplewbcCommandCfg)
    # _set_play_command(twist_cmd)

  return cfg
