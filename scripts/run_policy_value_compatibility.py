#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


COMBINATIONS = {
    "parent_policy_parent_value": ("parent", "parent"),
    "child_policy_parent_value": ("child", "parent"),
    "parent_policy_child_value": ("parent", "child"),
    "child_policy_child_value": ("child", "child"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-root D2/32 policy/value compatibility ablations."
    )
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--child-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--game-log", type=Path, default=None)
    parser.add_argument(
        "--positions",
        default=(
            "olympics_t1,olympics_t3,olympics_t5,destalinization_t1,"
            "europe_scoring_t2,mexico_t5_04,mexico_t5_05,iran_coup_t7,"
            "final_scoring_t10"
        ),
    )
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--seed", type=int, default=8_808_800)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def root_record(path: Path, budget: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload["puct_depth2"]["budgets"][str(int(budget))])
    rows.sort(key=lambda row: (-float(row["N"]), -float(row["Q"]), int(row["action_index"])))
    raw = payload["raw"]["top10"][0]
    selected = rows[0]
    return {
        "raw_action": raw["action"],
        "raw_action_index": int(raw["action_index"]),
        "search_action": selected["action"],
        "search_action_index": int(selected["action_index"]),
        "root_actions": [
            {
                "action": row["action"],
                "action_index": int(row["action_index"]),
                "N": float(row["N"]),
                "Q": float(row["Q"]),
                "P": float(row["P"]),
            }
            for row in rows
        ],
        "raw_critic": float(payload["raw"]["raw_critic"]),
        "calibrated_critic": float(payload["raw"]["calibrated_critic"]),
        "information_mask": payload["root"]["information_mask"],
    }


def summarize(output_dir: Path, positions: list[str], budget: int) -> dict[str, Any]:
    combinations: dict[str, Any] = {}
    for name in COMBINATIONS:
        combinations[name] = {
            position: root_record(output_dir / name / f"{position}.json", budget)
            for position in positions
        }
    baseline = combinations["parent_policy_parent_value"]
    for name, records in combinations.items():
        for position, record in records.items():
            record["agrees_with_parent_search"] = (
                record["search_action_index"] == baseline[position]["search_action_index"]
            )
    agreement = {
        name: sum(record["agrees_with_parent_search"] for record in records.values())
        / max(1, len(records))
        for name, records in combinations.items()
    }
    changed = {
        name: [
            position
            for position, record in records.items()
            if not record["agrees_with_parent_search"]
        ]
        for name, records in combinations.items()
    }
    policy_only = agreement["child_policy_parent_value"]
    value_only = agreement["parent_policy_child_value"]
    joint = agreement["child_policy_child_value"]
    strong = 0.78
    degraded = 0.67
    if joint >= strong:
        diagnosis = "compatible"
    elif policy_only >= strong and value_only >= strong:
        diagnosis = "policy_value_interaction"
    elif policy_only <= degraded and value_only >= policy_only + 0.15:
        diagnosis = "policy_prior_degradation"
    elif value_only <= degraded and policy_only >= value_only + 0.15:
        diagnosis = "value_degradation"
    else:
        diagnosis = "mixed_or_shared_trunk"
    return {
        "budget": int(budget),
        "positions": positions,
        "agreement_with_parent_search": agreement,
        "changed_positions": changed,
        "diagnosis": diagnosis,
        "diagnosis_inputs": {
            "child_policy_parent_value_agreement": policy_only,
            "parent_policy_child_value_agreement": value_only,
            "child_policy_child_value_agreement": joint,
            "strong_threshold": strong,
            "degraded_threshold": degraded,
        },
        "combinations": combinations,
    }


def main() -> None:
    args = parse_args()
    parent = args.parent_checkpoint.expanduser().resolve()
    child = args.child_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = [item.strip() for item in args.positions.split(",") if item.strip()]
    checkpoints = {"parent": parent, "child": child}
    identity = {
        "parent_checkpoint": str(parent),
        "child_checkpoint": str(child),
        "game_log": str(args.game_log.expanduser().resolve()) if args.game_log else None,
        "positions": positions,
        "belief_particles": int(args.belief_particles),
        "budget": int(args.budget),
        "seed": int(args.seed),
    }
    identity_path = output_dir / "config.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise SystemExit(f"compatibility configuration mismatch: {identity_path}")
    else:
        atomic_json(identity_path, identity)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for name, (policy_source, value_source) in COMBINATIONS.items():
        combination_dir = output_dir / name
        manifest = combination_dir / "manifest.json"
        if manifest.is_file() and all(
            (combination_dir / f"{position}.json").is_file() for position in positions
        ):
            continue
        command = [
            sys.executable,
            "scripts/run_frozen_diagnostics.py",
            "--checkpoint", str(checkpoints[policy_source]),
            "--value-checkpoint", str(checkpoints[value_source]),
            "--output-dir", str(combination_dir),
            "--positions", ",".join(positions),
            "--belief-particles", str(args.belief_particles),
            "--budgets", str(args.budget),
            "--depth1-budget", str(args.budget),
            "--depth3-budget", str(args.budget),
            "--root-search-only",
            "--num-gpus", "1.0",
            "--seed", str(args.seed),
        ]
        if args.game_log is not None:
            command.extend(["--game-log", str(args.game_log.expanduser().resolve())])
        log_path = output_dir / f"{name}.log"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "compatibility_start", "command": command}) + "\n")
            handle.flush()
            subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    report = {
        **identity,
        **summarize(output_dir, positions, args.budget),
    }
    atomic_json(output_dir / "compatibility_report.json", report)
    print(json.dumps({
        "event": "policy_value_compatibility_complete",
        "agreement_with_parent_search": report["agreement_with_parent_search"],
        "changed_positions": report["changed_positions"],
        "diagnosis": report["diagnosis"],
        "report": str(output_dir / "compatibility_report.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
