import copy
import os

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from torch import nn

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class _LegacyOnnxPolicy(nn.Module):
  """ONNX wrapper for legacy RSL-RL ActorCritic policies."""

  input_names = ["obs"]
  output_names = ["actions"]

  def __init__(self, policy: nn.Module, obs_normalizer: nn.Module, input_size: int):
    super().__init__()
    self.policy = copy.deepcopy(policy).to("cpu")
    self.obs_normalizer = copy.deepcopy(obs_normalizer).to("cpu")
    self.input_size = input_size

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.policy.act_inference(self.obs_normalizer(obs))

  def get_dummy_inputs(self) -> torch.Tensor:
    return torch.zeros(1, self.input_size)


def _infer_legacy_policy_input_size(policy: nn.Module) -> int:
  actor = getattr(policy, "actor", None)
  if actor is not None:
    for module in actor.modules():
      if isinstance(module, nn.Linear):
        return module.in_features
  raise AttributeError("Cannot infer ONNX input size for legacy policy.")


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Strip None-valued optional configs so MLPModel doesn't receive them.
    for key in ("actor", "critic"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Overrides the base implementation to set dynamo=False, avoiding warnings about
    dynamic_axes being deprecated with the new TorchDynamo export path
    (torch>=2.9 default).
    """
    policy = self.alg.get_policy()
    if hasattr(policy, "as_onnx"):
      onnx_model = policy.as_onnx(verbose=verbose)
    else:
      onnx_model = _LegacyOnnxPolicy(
        policy,
        self.obs_normalizer,
        _infer_legacy_policy_input_size(policy),
      )
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint.

    Extends the base implementation to persist the environment's
    common_step_counter and to respect the ``upload_model`` config flag.
    """
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    if self.empirical_normalization:
      saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
      saved_dict["privileged_obs_norm_state_dict"] = (
        self.privileged_obs_normalizer.state_dict()
      )
    torch.save(saved_dict, path)
    if not self.cfg["upload_model"]:
      return
    if hasattr(self, "logger"):
      self.logger.save_model(path, self.current_learning_iteration)
    elif getattr(self, "logger_type", None) in ["neptune", "wandb"] and not getattr(
      self, "disable_logs", False
    ):
      self.writer.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint.

    Extends the base implementation to:
    1. Restore common_step_counter to preserve curricula state.
    2. Migrate legacy checkpoints (actor.* -> mlp.*, actor_obs_normalizer.*
      -> obs_normalizer.*) to the current format (rsl-rl>=4.0).
    """
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    if "model_state_dict" in loaded_dict:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = loaded_dict.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():
        # Migrate actor keys.
        if key.startswith("actor."):
          new_key = key.replace("actor.", "mlp.")
          actor_state_dict[new_key] = value
        elif key.startswith("actor_obs_normalizer."):
          new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
          actor_state_dict[new_key] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        # Migrate critic keys.
        if key.startswith("critic."):
          new_key = key.replace("critic.", "mlp.")
          critic_state_dict[new_key] = value
        elif key.startswith("critic_obs_normalizer."):
          new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
          critic_state_dict[new_key] = value

      loaded_dict["actor_state_dict"] = actor_state_dict
      loaded_dict["critic_state_dict"] = critic_state_dict

    # Migrate rsl-rl 4.x actor keys to 5.x distribution keys.
    actor_sd = loaded_dict.get("actor_state_dict", {})
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

    if self.empirical_normalization:
      if "obs_norm_state_dict" in loaded_dict:
        self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        privileged_obs_norm_state_dict = loaded_dict.get(
          "privileged_obs_norm_state_dict",
          loaded_dict["obs_norm_state_dict"],
        )
        self.privileged_obs_normalizer.load_state_dict(privileged_obs_norm_state_dict)
      else:
        print(
          f"[WARN] Checkpoint {path} has no obs_norm_state_dict; "
          "using a fresh observation normalizer."
        )

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos
