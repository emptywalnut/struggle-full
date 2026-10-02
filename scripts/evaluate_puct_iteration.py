#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate raw, cheap-search, and deep-search curves for one distilled checkpoint."
    )
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-pairs", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=82_650_000)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--cheap-depth", type=int, default=2)
    parser.add_argument("--cheap-simulations", type=int, default=32)
    parser.add_argument("--deep-depth", type=int, default=4)
    parser.add_argument("--deep-simulations", type=int, default=128)
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--max-atomic-steps", type=int, default=512)
    parser.add_argument(
        "--max-expanded-nodes",
        type=int,
        default=8192,
        help="Per-search safety cap. Any reported cutoff invalidates that search for teacher use.",
    )
    parser.add_argument("--num-gpus-per-checkpoint", type=float, default=0.24)
    parser.add_argument("--capture-search-replay", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(
        command,
        cwd=cwd,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        text=True,
        check=True,
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_root.mkdir(parents=True, exist_ok=True)
    student = args.student_checkpoint.expanduser().resolve()
    parent = args.parent_checkpoint.expanduser().resolve()
    curves: list[tuple[str, int | None, int | None]] = [
        ("raw", None, None),
        ("cheap", args.cheap_depth, args.cheap_simulations),
        ("deep", args.deep_depth, args.deep_simulations),
    ]
    summaries: dict[str, Any] = {}
    for name, depth, simulations in curves:
        output_dir = args.output_root / name
        launcher = [
            sys.executable,
            "scripts/run_puct_pair_shards.py",
            "--output-root",
            str(output_dir),
            "--shards",
            str(args.seed_pairs),
            "--pairs-per-shard",
            "1",
            "--seed-base",
            str(args.seed_base),
            "--gpus",
            args.gpus,
            "--workers-per-gpu",
            str(args.workers_per_gpu),
        ]
        if args.capture_search_replay and depth is not None:
            launcher.append("--capture-search-replay")
        candidate_label = f"student-{name}"
        launcher.extend(
            [
                "--",
                "--candidate-checkpoint",
                str(student),
                "--opponent-checkpoint",
                str(parent),
                "--candidate-label",
                candidate_label,
                "--opponent-label",
                "parent-raw",
                "--belief-particles",
                str(args.belief_particles),
                "--max-atomic-steps",
                str(args.max_atomic_steps),
                "--max-expanded-nodes",
                str(args.max_expanded_nodes),
                "--num-gpus-per-checkpoint",
                str(args.num_gpus_per_checkpoint),
            ]
        )
        if depth is not None and simulations is not None:
            launcher.extend(
                [
                    "--candidate-search",
                    "--meaningful-depth",
                    str(depth),
                    "--forced-response-depth",
                    str(depth),
                    "--initial-simulations",
                    str(simulations),
                    "--medium-simulations",
                    str(simulations),
                    "--max-simulations",
                    str(simulations),
                ]
            )
        run(launcher, cwd=root)
        summary_path = output_dir / "summary.json"
        run(
            [
                sys.executable,
                "scripts/summarize_puct_tournament.py",
                "--result-dir",
                str(output_dir),
                "--baseline",
                "parent-raw",
                "--output",
                str(summary_path),
            ],
            cwd=root,
        )
        summaries[name] = json.loads(summary_path.read_text(encoding="utf-8"))
        (args.output_root / "iteration_evaluation.json").write_text(
            json.dumps(
                {
                    "student_checkpoint": str(student),
                    "parent_checkpoint": str(parent),
                    "seed_base": args.seed_base,
                    "seed_pairs": args.seed_pairs,
                    "curves": summaries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
