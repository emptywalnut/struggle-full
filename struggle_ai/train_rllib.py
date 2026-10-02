from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

from ray.rllib.algorithms.ppo import PPOConfig as _PPOConfig


_IMPL_NAME = "struggle_ai._train_rllib_recovered_20260902"
_IMPL_PATH = Path(__file__).with_name("_train_rllib_recovered_20260902.pyc")
_SPEC = importlib.util.spec_from_file_location(_IMPL_NAME, _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load recovered trainer from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_IMPL_NAME] = _IMPL
_SPEC.loader.exec_module(_IMPL)

for _name, _value in vars(_IMPL).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


def linear_annealed_value(start: float, end: float, elapsed: float, horizon: float) -> float:
    if horizon <= 0:
        return end
    progress = min(1.0, max(0.0, elapsed / horizon))
    return start + progress * (end - start)


def _set_exploration(policy: Any, temperature: float, entropy_coeff: float) -> None:
    policy.entropy_coeff = entropy_coeff
    models = [getattr(policy, "model", None), *getattr(policy, "model_gpu_towers", [])]
    for model in models:
        if model is not None and hasattr(model, "policy_temperature"):
            model.policy_temperature = temperature


def _install_wall_clock_exploration_anneal() -> None:
    horizon = float(os.environ.get("STRUGGLE_EXPLORATION_ANNEAL_SECONDS", "0"))
    if horizon <= 0:
        return
    final_temperature = float(os.environ["STRUGGLE_POLICY_TEMPERATURE_FINAL"])
    final_entropy = float(os.environ["STRUGGLE_ENTROPY_COEFF_FINAL"])
    original_build = _PPOConfig.build

    def build(config: Any, *args: Any, **kwargs: Any) -> Any:
        algo = original_build(config, *args, **kwargs)
        original_train = algo.train
        started_at = time.monotonic()
        last_temperature_bucket: int | None = None

        def train() -> dict[str, Any]:
            nonlocal last_temperature_bucket
            elapsed = time.monotonic() - started_at
            policy = algo.get_policy("shared_policy")
            start_temperature = float(getattr(policy.model, "policy_temperature", 1.0)) if last_temperature_bucket is None else float(config.model["custom_model_config"]["policy_temperature"])
            start_entropy = float(config.entropy_coeff)
            temperature = linear_annealed_value(start_temperature, final_temperature, elapsed, horizon)
            entropy = linear_annealed_value(start_entropy, final_entropy, elapsed, horizon)
            _set_exploration(policy, temperature, entropy)
            bucket = int(temperature * 100)
            if bucket != last_temperature_bucket:
                workers = getattr(algo, "env_runner_group", None)
                workers = workers() if callable(workers) else workers
                if workers is not None:
                    workers.foreach_policy(
                        lambda remote_policy, policy_id: _set_exploration(remote_policy, temperature, entropy)
                        if policy_id == "shared_policy"
                        else None
                    )
                last_temperature_bucket = bucket
                print(
                    {
                        "event": "exploration_anneal_set",
                        "elapsed_seconds": elapsed,
                        "policy_temperature": temperature,
                        "entropy_coeff": entropy,
                    },
                    flush=True,
                )
            return original_train()

        algo.train = train
        return algo

    _PPOConfig.build = build


if __name__ == "__main__":
    _install_wall_clock_exploration_anneal()
    _IMPL.main()
