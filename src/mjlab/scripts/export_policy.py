"""Script to export a trained policy to a TorchScript .pt file.

Exports the actor policy as a TorchScript model that can be loaded with
``torch.jit.load()`` for deployment, along with a ``policy.json`` containing
observation/action metadata.
"""

import json
import copy
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from torch import nn
import torch
import tyro

# When this file is executed directly as
# `python src/mjlab/scripts/export_policy.py`, Python puts `src/mjlab/scripts`
# on sys.path but not the repository root. Add it so the in-tree `rsl_rl/`
# package is preferred over the site-packages dependency.
if __package__ in (None, ""):
  repo_root = Path(__file__).resolve().parents[3]
  for path in (str(repo_root / "src"), str(repo_root)):
    if path not in sys.path:
      sys.path.insert(0, path)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path


class _LegacyJitPolicy(nn.Module):
  """TorchScript wrapper for legacy actor-critic policies."""

  def __init__(self, actor: nn.Module, obs_normalizer: nn.Module):
    super().__init__()
    self.actor = copy.deepcopy(actor).to("cpu")
    self.obs_normalizer = copy.deepcopy(obs_normalizer).to("cpu")

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.actor(self.obs_normalizer(obs))


@dataclass(frozen=True)
class ExportConfig:
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  """Local checkpoint file to export from."""
  output_file: str | None = None
  """Output .pt file path. Defaults to the experiment's export/ directory."""
  device: str | None = None


def run_export(task_id: str, cfg: ExportConfig):
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  # Resolve checkpoint path.
  log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    print(f"[INFO]: Loading checkpoint: {resume_path.name}")
  else:
    if cfg.wandb_run_path is None:
      raise ValueError(
        "Either `--checkpoint-file` or `--wandb-run-path` is required."
      )
    resume_path, was_cached = get_wandb_checkpoint_path(
      log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
    )
    run_id = resume_path.parent.name
    checkpoint_name = resume_path.name
    cached_str = "cached" if was_cached else "downloaded"
    print(
      f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
    )

  # Determine output path.
  if cfg.output_file is not None:
    output_path = Path(cfg.output_file)
  else:
    output_path = log_root_path / "exported" / "policy_mjlab.pt"
  output_path.parent.mkdir(parents=True, exist_ok=True)

  # Build env and runner.
  env_cfg.scene.num_envs = 1
  env_cfg.sim.nconmax = max(env_cfg.sim.nconmax or 0, 512)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
  )

  # Get the actor policy and export via rsl-rl's built-in JIT export when
  # available. Older ActorCritic policies do not implement as_jit(), so wrap
  # the actor and observation normalizer directly.
  policy = runner.alg.get_policy()

  if hasattr(policy, "as_jit"):
    # as_jit() returns a scriptable nn.Module with distribution stripped out.
    jit_ready = policy.as_jit()
  else:
    actor = getattr(policy, "actor", None)
    if actor is None:
      raise AttributeError("Policy does not expose as_jit() or an actor module.")
    jit_ready = _LegacyJitPolicy(actor, runner.obs_normalizer)
  jit_ready.to("cpu")

  # Script to TorchScript.
  scripted_policy = torch.jit.script(jit_ready)

  # Save TorchScript model.
  torch.jit.save(scripted_policy, str(output_path))

  # Save metadata alongside the model.
  unwrapped = env.unwrapped
  metadata = {
    "observation_terms": unwrapped.observation_manager.active_terms.get("actor", []),
    "action_terms": list(unwrapped.action_manager.active_terms),
  }
  json_path = output_path.with_suffix(".json")
  with open(json_path, "w") as f:
    json.dump(metadata, f, indent=2)

  num_params = sum(p.numel() for p in policy.parameters())
  file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
  print(f"[INFO]: Exported TorchScript policy to {output_path}")
  print(f"[INFO]: Metadata saved to {json_path}")
  print(f"[INFO]: Parameters: {num_params:,}, File size: {file_size_mb:.2f} MB")

  env.close()


def main():
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  from mjlab.tasks.registry import list_tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    ExportConfig,
    args=remaining_args,
    default=ExportConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  run_export(chosen_task, args)


if __name__ == "__main__":
  main()
