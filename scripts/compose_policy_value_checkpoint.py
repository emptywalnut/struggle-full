#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from struggle_ai.train_mcts import build_algorithm
from struggle_ai.train_rllib import SHARED_POLICY_ID, load_policy_weights, save_checkpoint


VALUE_PREFIXES = ("value.", "value_state_encoder.")


def is_value_tensor(name: str) -> bool:
    return name.startswith(VALUE_PREFIXES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose an actor checkpoint with the value head from another checkpoint."
    )
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="composed")
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--card-history-layers", type=int, default=2)
    parser.add_argument("--history-attention-heads", type=int, default=4)
    parser.add_argument("--history-dropout", type=float, default=0.05)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-neighbor-hops", type=int, default=2)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument(
        "--value-state-tower",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Build the composed checkpoint with the critic-specific state tower and "
            "copy both value_state_encoder.* and value.* from the value checkpoint."
        ),
    )
    return parser.parse_args()


def digest(weights: dict[str, object]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(weights):
        array = np.ascontiguousarray(np.asarray(weights[name]))
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(str(array.shape).encode("ascii"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def main() -> None:
    args = parse_args()
    actor_checkpoint = args.actor_checkpoint.expanduser().resolve()
    value_checkpoint = args.value_checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    algo_args = SimpleNamespace(
        restore_from=actor_checkpoint,
        num_gpus=args.num_gpus,
        hidden=args.hidden,
        history_layers=args.history_layers,
        card_history_layers=args.card_history_layers,
        history_attention_heads=args.history_attention_heads,
        history_dropout=args.history_dropout,
        graph_layers=args.graph_layers,
        graph_neighbor_hops=args.graph_neighbor_hops,
        policy_temperature=args.policy_temperature,
        value_state_tower=args.value_state_tower,
        ray_num_cpus=1,
        ray_object_store_memory=150_000_000,
    )
    algo = build_algorithm(algo_args)
    try:
        policy = algo.get_policy(SHARED_POLICY_ID)
        actor_weights = {
            name: np.array(value, copy=True) for name, value in policy.get_weights().items()
        }
        value_weights = {
            name: np.array(value, copy=True)
            for name, value in load_policy_weights(value_checkpoint, SHARED_POLICY_ID).items()
        }
        composed = {name: np.array(value, copy=True) for name, value in actor_weights.items()}
        replaced: list[dict[str, object]] = []
        for name in sorted(actor_weights):
            if not is_value_tensor(name):
                continue
            if name not in value_weights:
                raise KeyError(f"value checkpoint is missing {name}")
            actor_array = np.asarray(actor_weights[name])
            value_array = np.asarray(value_weights[name])
            if actor_array.shape != value_array.shape:
                raise ValueError(
                    f"value tensor shape mismatch for {name}: "
                    f"actor={actor_array.shape}, value={value_array.shape}"
                )
            composed[name] = np.array(value_weights[name], copy=True)
            replaced.append(
                {
                    "name": name,
                    "shape": list(actor_array.shape),
                    "actor_max_delta": float(np.max(np.abs(actor_array - value_array))),
                }
            )
        expected = {name for name in actor_weights if is_value_tensor(name)}
        if not expected or {entry["name"] for entry in replaced} != expected:
            raise RuntimeError("did not replace the complete value head")
        source_tower = any(name.startswith("value_state_encoder.") for name in value_weights)
        composed_tower = any(name.startswith("value_state_encoder.") for name in expected)
        if source_tower != composed_tower:
            raise RuntimeError(
                "value-state-tower architecture mismatch: "
                f"source={source_tower}, composed={composed_tower}; "
                "pass --value-state-tower when composing a tower checkpoint"
            )
        policy.set_weights(composed)
        checkpoint = Path(save_checkpoint(algo, args.output_dir, args.label)).resolve()
    finally:
        algo.stop()
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass

    saved_weights = load_policy_weights(checkpoint, SHARED_POLICY_ID)
    for name in expected:
        if not np.array_equal(np.asarray(saved_weights[name]), np.asarray(value_weights[name])):
            raise RuntimeError(f"saved value tensor differs from source: {name}")
    for name in actor_weights:
        if name not in expected and not np.array_equal(
            np.asarray(saved_weights[name]), np.asarray(actor_weights[name])
        ):
            raise RuntimeError(f"saved actor tensor changed unexpectedly: {name}")

    manifest = {
        "actor_checkpoint": str(actor_checkpoint),
        "value_checkpoint": str(value_checkpoint),
        "checkpoint": str(checkpoint),
        "actor_weights_sha256": digest(actor_weights),
        "value_weights_sha256": digest(value_weights),
        "composed_weights_sha256": digest(saved_weights),
        "value_state_tower": bool(args.value_state_tower),
        "replaced": replaced,
        "verified": True,
    }
    path = args.output_dir / "composition_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
