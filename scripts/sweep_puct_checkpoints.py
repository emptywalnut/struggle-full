#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


UPDATE_RE = re.compile(r"(?:updates-|^u)(?P<update>\d+)")
METRIC_KEYS = (
    "search_distill/policy_cross_entropy",
    "search_distill/policy_entropy",
    "search_distill/teacher_accuracy",
    "search_distill/raw_teacher_agreement",
    "search_distill/value_mse",
    "search_distill/value_sign_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched raw/D2 sweep over saved distillation checkpoints.")
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Candidate in LABEL=CHECKPOINT form; repeat for every saved checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distillation-report", type=Path, default=None)
    parser.add_argument("--distill-log", type=Path, default=None)
    parser.add_argument("--final-label", default="u200")
    parser.add_argument("--seed-pairs", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=94_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--intermediate-margin", type=float, default=None)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--checkpoint must be LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or label in result:
            raise SystemExit(f"invalid or duplicate checkpoint label: {label!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"checkpoint does not exist: {path}")
        result[label] = path
    if not result:
        raise SystemExit("provide at least one --checkpoint LABEL=PATH")
    return result


def checkpoint_update(label: str, path: Path, report: dict[str, Any] | None) -> int | None:
    for text in (label, path.name):
        match = UPDATE_RE.search(text)
        if match:
            return int(match.group("update"))
    if report and path == Path(str(report.get("checkpoint") or "")).expanduser().resolve():
        return int(report["updates"])
    return None


def metric_history(report_path: Path | None, log_path: Path | None) -> tuple[dict[int, dict[str, float]], dict[str, Any] | None]:
    report = None
    history: dict[int, dict[str, float]] = {}
    if report_path and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for metrics in report.get("history", []):
            history[int(metrics["update"])] = {
                key: float(metrics[key]) for key in METRIC_KEYS if key in metrics
            }
        if report.get("last_metrics"):
            metrics = report["last_metrics"]
            history[int(metrics["update"])] = {
                key: float(metrics[key]) for key in METRIC_KEYS if key in metrics
            }
    if log_path and log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "'event': 'search_distill_progress'" not in line:
                continue
            try:
                metrics = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            history[int(metrics["update"])] = {
                key: float(metrics[key]) for key in METRIC_KEYS if key in metrics
            }
    return history, report


def evaluation_command(
    checkpoint: Path,
    parent: Path,
    output: Path,
    args: argparse.Namespace,
    *,
    search: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/eval_puct_agent_pair_parallel.py",
        "--candidate-checkpoint", str(checkpoint),
        "--opponent-checkpoint", str(parent),
        "--candidate-label", checkpoint.name,
        "--opponent-label", "accepted-parent",
        "--output-dir", str(output),
        "--seed-pairs", str(args.seed_pairs),
        "--seed-base", str(args.seed_base),
        "--workers", str(min(args.workers, args.seed_pairs * 2)),
        "--gpus", str(args.gpus),
        "--replicas-per-checkpoint", "1",
        "--meaningful-depth", "2",
        "--belief-particles", "32",
        "--initial-simulations", "32",
        "--medium-simulations", "32",
        "--max-simulations", "32",
        "--resume",
        "--search" if search else "--no-search",
    ]
    return command


def run_evaluation(command: list[str], output: Path, log_path: Path) -> dict[str, Any]:
    results_path = output / "results.json"
    if not results_path.is_file():
        output.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "checkpoint_sweep_eval_start", "command": command}) + "\n")
            handle.flush()
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return json.loads(results_path.read_text(encoding="utf-8"))["summary"]


def side_floor(summary: dict[str, Any]) -> float:
    return min(float(summary["by_candidate_side"][side]["score"]) for side in ("us", "ussr"))


def rank_key(record: dict[str, Any]) -> tuple[float, float, float]:
    raw = record["raw"]
    d2 = record["d2"]
    return (
        min(float(raw["score"]), float(d2["score"]), side_floor(raw), side_floor(d2)),
        min(float(raw["score"]), float(d2["score"])),
        (float(raw["score"]) + float(d2["score"])) / 2.0,
    )


def choose(records: dict[str, dict[str, Any]], final_label: str, margin: float) -> tuple[str, dict[str, Any]]:
    if final_label not in records:
        raise SystemExit(f"--final-label {final_label!r} is not among checkpoint labels")
    final_key = rank_key(records[final_label])
    best_label = max(records, key=lambda label: rank_key(records[label]))
    best_key = rank_key(records[best_label])
    selected = final_label
    clearly_dominates = False
    if best_label != final_label:
        no_material_side_regression = best_key[0] + 1e-12 >= final_key[0]
        clearly_dominates = no_material_side_regression and (
            best_key[1] >= final_key[1] + margin
            or best_key[2] >= final_key[2] + margin
        )
        if clearly_dominates:
            selected = best_label
    return selected, {
        "best_ranked_label": best_label,
        "best_rank_key": list(best_key),
        "final_rank_key": list(final_key),
        "intermediate_clearly_dominates_final": clearly_dominates,
        "margin": margin,
    }


def main() -> None:
    args = parse_args()
    parent = args.parent_checkpoint.expanduser().resolve()
    checkpoints = parse_checkpoints(args.checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    margin = (
        float(args.intermediate_margin)
        if args.intermediate_margin is not None
        else 1.0 / max(2.0, float(args.seed_pairs) * 2.0)
    )
    identity = {
        "parent_checkpoint": str(parent),
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "seed_pairs": int(args.seed_pairs),
        "seed_base": int(args.seed_base),
        "gpus": str(args.gpus),
        "final_label": str(args.final_label),
        "intermediate_margin": margin,
    }
    identity_path = output_dir / "config.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise SystemExit(f"checkpoint sweep configuration mismatch: {identity_path}")
    else:
        atomic_json(identity_path, identity)
    history, distill_report = metric_history(args.distillation_report, args.distill_log)
    records: dict[str, dict[str, Any]] = {}
    for label, checkpoint in checkpoints.items():
        candidate_dir = output_dir / label
        raw = run_evaluation(
            evaluation_command(checkpoint, parent, candidate_dir / "raw", args, search=False),
            candidate_dir / "raw",
            candidate_dir / "raw.log",
        )
        d2 = run_evaluation(
            evaluation_command(checkpoint, parent, candidate_dir / "d2", args, search=True),
            candidate_dir / "d2",
            candidate_dir / "d2.log",
        )
        update = checkpoint_update(label, checkpoint, distill_report)
        records[label] = {
            "checkpoint": str(checkpoint),
            "update": update,
            "training_metrics": history.get(update) if update is not None else None,
            "raw": raw,
            "d2": d2,
        }
        atomic_json(output_dir / "progress.json", {"identity": identity, "checkpoints": records})
    selected, selection = choose(records, args.final_label, margin)
    report = {
        "identity": identity,
        "checkpoints": records,
        "selection": {
            **selection,
            "selected_label": selected,
            "selected_checkpoint": records[selected]["checkpoint"],
            "selected_update": records[selected]["update"],
        },
    }
    atomic_json(output_dir / "checkpoint_sweep.json", report)
    print(json.dumps({"event": "checkpoint_sweep_complete", **report["selection"]}, sort_keys=True))


if __name__ == "__main__":
    main()
