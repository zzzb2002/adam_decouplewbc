from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.decouplewbc.rl import DecouplewbcOnPolicyRunner

from .env_cfgs import adam_pro_flat_env_cfg, adam_pro_rough_env_cfg
from .rl_cfg import adam_pro_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Decouplewbc-Rough-Adam-Pro-12DOF",
  env_cfg=adam_pro_rough_env_cfg(),
  play_env_cfg=adam_pro_rough_env_cfg(play=True),
  rl_cfg=adam_pro_ppo_runner_cfg(),
  runner_cls=DecouplewbcOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Decouplewbc-Flat-Adam-Pro-12DOF",
  env_cfg=adam_pro_flat_env_cfg(),
  play_env_cfg=adam_pro_flat_env_cfg(play=True),
  rl_cfg=adam_pro_ppo_runner_cfg(),
  runner_cls=DecouplewbcOnPolicyRunner,
)