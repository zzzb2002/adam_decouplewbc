from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  cowa_wheel_v2_flat_env_cfg,
)
from .rl_cfg import cowa_wheel_v2_ppo_runner_cfg


register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Cowa-Wheel-V2",
  env_cfg=cowa_wheel_v2_flat_env_cfg(),
  play_env_cfg=cowa_wheel_v2_flat_env_cfg(play=True),
  rl_cfg=cowa_wheel_v2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
