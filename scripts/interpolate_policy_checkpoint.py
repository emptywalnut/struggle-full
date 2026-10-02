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


SCOPES = {
    "action_head": (
        "action_encoder",
        "policy_query",
        "policy_query_us",
        "policy_query_ussr",
        "action_bias",
        "action_bias_us",
        "action_bias_ussr",
    ),
    "policy_heads": (
        "policy_query",
        "policy_query_us",
        "policy_query_ussr",
        "action_bias",
        "action_bias_us",
        "action_bias_ussr",
    ),
    "policy_value_heads": (
        "policy_query",
        "policy_query_us",
        "policy_query_ussr",
        "action_bias",
        "action_bias_us",
        "action_bias_ussr",
        "value",
    ),
    "value_tower": ("value_state_encoder",),
    "value_tower_head": ("value_state_encoder", "value"),
}


def interpolate_weights(
    parent: dict[str, object],
    student: dict[str, object],
    *,
    alpha: float,
    prefixes: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if parent.keys() != student.keys():
        missing = sorted(parent.keys() - student.keys())
        extra = sorted(student.keys() - parent.keys())
        raise ValueError(f"checkpoint keys differ: missing={missing}, extra={extra}")

    result: dict[str, np.ndarray] = {}
    changed: list[dict[str, object]] = []
    for name, parent_value in parent.items():
        parent_array = np.asarray(parent_value)
        student_array = np.asarray(student[name])
        if parent_array.shape != student_array.shape:
            raise ValueError(
                f"checkpoint tensor shape mismatch for {name}: "
                f"parent={parent_array.shape}, student={student_array.shape}"
            )
        selected = name.startswith(prefixes)
        if selected:
            blended = parent_array + alpha * (student_array - parent_array)
            result[name] = blended.astype(parent_array.dtype, copy=False)
            changed.append(
                {
                    "name": name,
                    "shape": list(parent_array.shape),
                    "parent_student_max_delta": float(
                        np.max(np.abs(parent_array - student_array), initial=0.0)
                    ),
                    "parent_blended_max_delta": float(
                        np.max(np.abs(parent_array - result[name]), initial=0.0)
                    ),
                }
            )
        else:
            result[name] = np.array(parent_array, copy=True)
    if not changed:
        raise ValueError(f"no tensors matched prefixes {prefixes}")
    return result, changed


def digest(weights: dict[str, object]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(weights):
        array = np.ascontiguousarray(np.asarray(weights[name]))
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(str(array.shape).encode("ascii"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate selected student tensors from a parent RLlib checkpoint."
    )
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="interpolated")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--scope", choices=tuple(SCOPES), default="policy_value_heads")
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
        help="Migrate the parent into the critic-specific state-tower architecture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parent_checkpoint = args.parent_checkpoint.expanduser().resolve()
    student_checkpoint = args.student_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    algo_args = SimpleNamespace(
        restore_from=parent_checkpoint,
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
        parent_weights = {
            name: np.array(value, copy=True) for name, value in policy.get_weights().items()
        }
        student_weights = {
            name: np.array(value, copy=True)
            for name, value in load_policy_weights(student_checkpoint, SHARED_POLICY_ID).items()
        }
        blended, changed = interpolate_weights(
            parent_weights,
            student_weights,
            alpha=float(args.alpha),
            prefixes=SCOPES[args.scope],
        )
        policy.set_weights(blended)
        checkpoint = Path(save_checkpoint(algo, output_dir, args.label)).resolve()
    finally:
        algo.stop()
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass

    saved = load_policy_weights(checkpoint, SHARED_POLICY_ID)
    for name, expected in blended.items():
        if not np.array_equal(np.asarray(saved[name]), np.asarray(expected)):
            raise RuntimeError(f"saved tensor differs from interpolated value: {name}")

    manifest = {
        "parent_checkpoint": str(parent_checkpoint),
        "student_checkpoint": str(student_checkpoint),
        "checkpoint": str(checkpoint),
        "alpha": float(args.alpha),
        "scope": args.scope,
        "prefixes": SCOPES[args.scope],
        "changed": changed,
        "parent_weights_sha256": digest(parent_weights),
        "student_weights_sha256": digest(student_weights),
        "interpolated_weights_sha256": digest(saved),
        "verified": True,
    }
    manifest_path = output_dir / f"{args.label}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
