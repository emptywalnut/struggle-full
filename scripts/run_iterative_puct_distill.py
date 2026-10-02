#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from struggle_ai.search_replay import load_search_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate deep PUCT teacher generation, hierarchical distillation, and optional PPO stabilization."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--position-manifest", type=Path, default=None)
    parser.add_argument("--game-log", type=Path, action="append", default=[])
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-searches", type=int, default=None)
    parser.add_argument("--teacher-workers", type=int, default=1)
    parser.add_argument(
        "--teacher-gpus",
        default="",
        help="Optional comma-separated physical GPU IDs assigned round-robin to parallel teacher workers.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--meaningful-depth", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=2048)
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--max-atomic-steps", type=int, default=512)
    parser.add_argument("--max-expanded-nodes", type=int, default=16384)
    parser.add_argument("--distill-updates", type=int, default=100)
    parser.add_argument("--distill-batch-size", type=int, default=32)
    parser.add_argument("--distill-lr", type=float, default=1e-5)
    parser.add_argument(
        "--search-policy-mix",
        type=float,
        default=0.5,
        help="PUCT share of the in-loop PPO search-distillation target.",
    )
    parser.add_argument("--lambda-anchor", type=float, default=1.0)
    parser.add_argument("--anchor-replay-batch-size", type=int, default=64)
    parser.add_argument("--balance-distillation-sides", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-q-advantage", type=float, default=0.003)
    parser.add_argument("--min-teacher-visits", type=float, default=4.0)
    parser.add_argument("--min-teacher-visit-share", type=float, default=0.5)
    parser.add_argument("--q-weight-scale", type=float, default=0.05)
    parser.add_argument("--max-q-weight", type=float, default=4.0)
    parser.add_argument("--hierarchy-weight", type=float, default=1.25)
    parser.add_argument(
        "--max-examples-per-source-root",
        type=int,
        default=8,
        help="Cap qualified hierarchical labels contributed by one searched root.",
    )
    parser.add_argument(
        "--parameter-scope",
        choices=(
            "actor",
            "head",
            "action_head",
            "actor_value",
            "actor_value_heads",
            "policy_value_heads",
        ),
        default="policy_value_heads",
    )
    parser.add_argument("--ppo-stabilization-episodes", type=int, default=0)
    parser.add_argument("--ppo-num-env-runners", type=int, default=1)
    parser.add_argument("--ppo-train-batch-size", type=int, default=2048)
    parser.add_argument(
        "--ppo-student-selection",
        choices=("distilled", "ppo"),
        default="distilled",
        help=(
            "Checkpoint propagated to the next search iteration after PPO stabilization. "
            "Select ppo only after paired evaluation shows that it does not regress."
        ),
    )
    parser.add_argument("--evaluation-seed-pairs", type=int, default=0)
    parser.add_argument("--evaluation-gpus", default="4,5")
    parser.add_argument("--evaluation-workers-per-gpu", type=int, default=2)
    parser.add_argument("--evaluation-cheap-depth", type=int, default=2)
    parser.add_argument("--evaluation-cheap-simulations", type=int, default=32)
    parser.add_argument("--evaluation-deep-depth", type=int, default=4)
    parser.add_argument("--evaluation-deep-simulations", type=int, default=128)
    parser.add_argument("--evaluation-max-expanded-nodes", type=int, default=8192)
    parser.add_argument("--evaluation-min-raw-score", type=float, default=0.50)
    parser.add_argument("--evaluation-min-raw-side-score", type=float, default=0.40)
    parser.add_argument(
        "--stop-on-rejected-student",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=8_260_001)
    return parser.parse_args()


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "command", "argv": command}) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def run_parallel(commands: list[tuple[list[str], Path, str | None]]) -> None:
    processes: list[tuple[subprocess.Popen[str], Any]] = []
    try:
        for command, log_path, gpu in commands:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            handle.write(json.dumps({"event": "command", "argv": command}) + "\n")
            handle.flush()
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((process, handle))
        failures: list[tuple[int, int]] = []
        for index, (process, _handle) in enumerate(processes):
            return_code = process.wait()
            if return_code != 0:
                failures.append((index, return_code))
        if failures:
            raise RuntimeError(f"parallel teacher workers failed: {failures}")
    finally:
        for process, handle in processes:
            if process.poll() is None:
                process.terminate()
        for process, handle in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            handle.close()


def newest_checkpoint(root: Path) -> Path:
    candidates = [path.parent for path in root.rglob("rllib_checkpoint.json")]
    if not candidates:
        raise RuntimeError(f"no RLlib checkpoint found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def replay_cutoff_counts(replay_path: Path) -> dict[str, int]:
    """Count safe and incomplete labels without discarding unrelated roots."""

    examples = load_search_replay((replay_path / "manifest.json",))
    safe = 0
    atomic = 0
    node = 0
    for example in examples:
        metadata = example.metadata
        has_atomic_cutoff = int(metadata.atomic_horizon_cutoffs) > 0
        has_node_cutoff = int(metadata.node_budget_cutoffs) > 0
        atomic += int(has_atomic_cutoff)
        node += int(has_node_cutoff)
        safe += int(not has_atomic_cutoff and not has_node_cutoff)
    return {
        "total_examples": len(examples),
        "safe_examples": safe,
        "atomic_cutoff_examples": atomic,
        "node_cutoff_examples": node,
    }


def student_acceptance_from_evaluation(
    evaluation: dict[str, Any],
    *,
    min_score: float,
    min_side_score: float,
) -> dict[str, Any]:
    raw = dict((evaluation.get("curves") or {}).get("raw") or {})
    ratings = list(raw.get("ratings") or [])
    student = next(
        (rating for rating in ratings if str(rating.get("agent")) == "student-raw"),
        None,
    )
    if student is None:
        raise RuntimeError("iteration evaluation has no student-raw rating")
    score = float(student.get("score", 0.0))
    us_score = float(student.get("us_score", 0.0))
    ussr_score = float(student.get("ussr_score", 0.0))
    accepted = score >= float(min_score) and min(us_score, ussr_score) >= float(min_side_score)
    return {
        "accepted": accepted,
        "score": score,
        "us_score": us_score,
        "ussr_score": ussr_score,
        "min_score": float(min_score),
        "min_side_score": float(min_side_score),
    }


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.teacher_workers <= 0:
        raise SystemExit("--teacher-workers must be positive")
    teacher_gpus = [gpu.strip() for gpu in args.teacher_gpus.split(",") if gpu.strip()]
    if args.position_manifest is None and not args.game_log:
        raise SystemExit("provide --position-manifest or at least one --game-log")
    args.output_root.mkdir(parents=True, exist_ok=True)
    current = args.checkpoint.expanduser().resolve()
    manifest: dict[str, Any] = {
        "started_at": time.time(),
        "initial_checkpoint": str(current),
        "position_manifest": (
            str(args.position_manifest.expanduser().resolve())
            if args.position_manifest is not None
            else None
        ),
        "game_logs": [str(path.expanduser().resolve()) for path in args.game_log],
        "iterations": [],
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    manifest_path = args.output_root / "iterative_manifest.json"
    write_manifest(manifest_path, manifest)

    for iteration in range(args.iterations):
        iteration_dir = args.output_root / f"iteration-{iteration:02d}"
        replay_dir = iteration_dir / "search_replay"
        distill_dir = iteration_dir / "distill"
        entry: dict[str, Any] = {
            "iteration": iteration,
            "parent_checkpoint": str(current),
            "status": "generating_teacher",
        }
        manifest["iterations"].append(entry)
        write_manifest(manifest_path, manifest)

        generate_base = [
            sys.executable,
            "scripts/generate_search_replay.py",
            "--checkpoint",
            str(current),
            "--initial-simulations",
            str(args.simulations),
            "--medium-simulations",
            str(args.simulations),
            "--max-simulations",
            str(args.simulations),
            "--meaningful-depth",
            str(args.meaningful_depth),
            "--forced-response-depth",
            str(args.meaningful_depth),
            "--belief-particles",
            str(args.belief_particles),
            "--max-atomic-steps",
            str(args.max_atomic_steps),
            "--max-expanded-nodes",
            str(args.max_expanded_nodes),
            "--seed",
            str(args.seed + iteration * 100_003),
            "--num-gpus",
            "1",
        ]
        replay_paths: list[Path] = []
        if args.position_manifest is not None and args.teacher_workers > 1:
            source_manifest = json.loads(
                args.position_manifest.expanduser().resolve().read_text(encoding="utf-8")
            )
            positions = list(source_manifest.get("positions") or [])
            worker_count = min(args.teacher_workers, len(positions))
            if worker_count <= 0:
                raise RuntimeError("position manifest contains no positions")
            shard_commands: list[tuple[list[str], Path, str | None]] = []
            for worker in range(worker_count):
                shard_manifest = {
                    **source_manifest,
                    "positions": positions[worker::worker_count],
                }
                shard_manifest_path = iteration_dir / "teacher_manifests" / f"positions-{worker:03d}.json"
                write_manifest(shard_manifest_path, shard_manifest)
                shard_replay = replay_dir / f"shard-{worker:03d}"
                replay_paths.append(shard_replay)
                command = [
                    *generate_base,
                    "--output-dir",
                    str(shard_replay),
                    "--position-manifest",
                    str(shard_manifest_path),
                ]
                gpu = teacher_gpus[worker % len(teacher_gpus)] if teacher_gpus else None
                shard_commands.append((command, iteration_dir / f"teacher-{worker:03d}.log", gpu))
            run_parallel(shard_commands)
        else:
            generate_command = [*generate_base, "--output-dir", str(replay_dir)]
            if args.position_manifest is not None:
                generate_command.extend(["--position-manifest", str(args.position_manifest)])
            for game_log in args.game_log:
                generate_command.extend(["--game-log", str(game_log)])
            if args.max_games is not None:
                generate_command.extend(["--max-games", str(args.max_games)])
            if args.max_examples is not None:
                generate_command.extend(["--max-examples", str(args.max_examples)])
            if args.max_searches is not None:
                generate_command.extend(["--max-searches", str(args.max_searches)])
            run(generate_command, iteration_dir / "teacher.log")
            replay_paths.append(replay_dir)
        accepted_replay_paths: list[Path] = []
        rejected_replay_paths: list[dict[str, Any]] = []
        replay_quality: list[dict[str, Any]] = []
        for replay_path in replay_paths:
            quality = replay_cutoff_counts(replay_path)
            quality["path"] = str((replay_path / "manifest.json").resolve())
            replay_quality.append(quality)
            if quality["safe_examples"] > 0:
                accepted_replay_paths.append(replay_path)
            else:
                rejected_replay_paths.append(quality)
        replay_paths = accepted_replay_paths
        if not replay_paths:
            raise RuntimeError("every generated teacher example was rejected by cutoff validation")
        entry["search_replay"] = [str((path / "manifest.json").resolve()) for path in replay_paths]
        entry["search_replay_quality"] = replay_quality
        entry["rejected_search_replay"] = rejected_replay_paths
        entry["status"] = "distilling"
        write_manifest(manifest_path, manifest)

        distill_command = [
            sys.executable,
            "scripts/distill_search_replay.py",
            "--restore-from",
            str(current),
            "--output-dir",
            str(distill_dir),
            "--updates",
            str(args.distill_updates),
            "--batch-size",
            str(args.distill_batch_size),
            "--lr",
            str(args.distill_lr),
            "--lambda-anchor",
            str(args.lambda_anchor),
            "--anchor-from-all-safe-replay",
            "--anchor-replay-batch-size",
            str(args.anchor_replay_batch_size),
            "--balance-sides" if args.balance_distillation_sides else "--no-balance-sides",
            "--parameter-scope",
            args.parameter_scope,
            "--min-q-advantage",
            str(args.min_q_advantage),
            "--min-teacher-visits",
            str(args.min_teacher_visits),
            "--min-teacher-visit-share",
            str(args.min_teacher_visit_share),
            "--q-weight-scale",
            str(args.q_weight_scale),
            "--max-q-weight",
            str(args.max_q_weight),
            "--hierarchy-weight",
            str(args.hierarchy_weight),
            "--max-examples-per-source-root",
            str(args.max_examples_per_source_root),
            "--seed",
            str(args.seed + iteration * 100_003 + 1),
            "--num-gpus",
            "1",
        ]
        for replay_path in replay_paths:
            distill_command.extend(["--search-replay", str(replay_path)])
        run(distill_command, iteration_dir / "distill.log")
        distill_report = json.loads((distill_dir / "distillation_report.json").read_text(encoding="utf-8"))
        current = Path(str(distill_report["checkpoint"])).expanduser().resolve()
        entry["distilled_checkpoint"] = str(current)
        distilled_checkpoint = current

        if args.ppo_stabilization_episodes > 0:
            entry["status"] = "ppo_stabilization"
            write_manifest(manifest_path, manifest)
            ppo_dir = iteration_dir / "ppo_stabilization"
            ppo_command = [
                sys.executable,
                "-m",
                "struggle_ai.train_rllib",
                "--restore-from",
                str(current),
                "--model-arch",
                "transformer_history",
                "--policy-sharing",
                "unified",
                "--multi-agent",
                "--side-specific-heads",
                "--stop-iters",
                "1000000",
                "--stop-episodes",
                str(args.ppo_stabilization_episodes),
                "--num-env-runners",
                str(args.ppo_num_env_runners),
                "--num-gpus",
                "1",
                "--train-batch-size",
                str(args.ppo_train_batch_size),
                "--batch-mode",
                "complete_episodes",
                "--search-distill-lambda",
                "0.05",
                "--search-distill-policy-mix",
                str(args.search_policy_mix),
                "--search-distill-lambda-value",
                "0.1",
                "--search-distill-lambda-search-q",
                "0.02",
                "--search-distill-confidence-weight-floor",
                "0.1",
                "--search-distill-min-value-examples-per-batch",
                str(min(32, args.distill_batch_size)),
                "--search-distill-updates-per-iter",
                "1",
                "--search-distill-batch-size",
                str(args.distill_batch_size),
                "--search-distill-lr",
                str(args.distill_lr),
                "--search-distill-min-q-advantage",
                str(args.min_q_advantage),
                "--search-distill-min-teacher-visits",
                str(args.min_teacher_visits),
                "--search-distill-min-teacher-visit-share",
                str(args.min_teacher_visit_share),
                "--search-distill-max-examples-per-source-root",
                str(args.max_examples_per_source_root),
                "--search-distill-require-flip",
                "--search-distill-parameter-scope",
                args.parameter_scope,
                "--checkpoint-dir",
                str(ppo_dir / "checkpoints"),
                "--metrics-dir",
                str(ppo_dir / "metrics"),
                "--log-games-dir",
                str(ppo_dir / "game_logs"),
                "--checkpoint-eval-every-steps",
                "0",
                "--checkpoint-eval-every-episodes",
                "0",
                "--checkpoint-eval-every-seconds",
                "0",
                "--no-final-eval",
                "--wandb-mode",
                "disabled",
                "--no-force-setup-heuristic",
                "--terminal-reward-scale",
                "2.0",
                "--reward-shaping-scale",
                "0.0",
                "--nuke-death-penalty",
                "1.0",
                "--defcon-suicide-mode",
                "hard_filter",
            ]
            for replay_path in replay_paths:
                ppo_command.extend(["--search-replay", str(replay_path)])
            run(ppo_command, iteration_dir / "ppo_stabilization.log")
            ppo_checkpoint = newest_checkpoint(ppo_dir / "checkpoints")
            entry["ppo_checkpoint"] = str(ppo_checkpoint)
            current = (
                ppo_checkpoint
                if args.ppo_student_selection == "ppo"
                else distilled_checkpoint
            )
            entry["ppo_student_selection"] = args.ppo_student_selection

        entry["student_checkpoint"] = str(current)
        if args.evaluation_seed_pairs > 0:
            entry["status"] = "evaluation"
            write_manifest(manifest_path, manifest)
            evaluation_dir = iteration_dir / "evaluation"
            evaluation_command = [
                sys.executable,
                "scripts/evaluate_puct_iteration.py",
                "--student-checkpoint",
                str(current),
                "--parent-checkpoint",
                str(entry["parent_checkpoint"]),
                "--output-root",
                str(evaluation_dir),
                "--seed-pairs",
                str(args.evaluation_seed_pairs),
                "--seed-base",
                str(args.seed + iteration * 100_003 + 20_000),
                "--gpus",
                args.evaluation_gpus,
                "--workers-per-gpu",
                str(args.evaluation_workers_per_gpu),
                "--cheap-depth",
                str(args.evaluation_cheap_depth),
                "--cheap-simulations",
                str(args.evaluation_cheap_simulations),
                "--deep-depth",
                str(args.evaluation_deep_depth),
                "--deep-simulations",
                str(args.evaluation_deep_simulations),
                "--belief-particles",
                str(args.belief_particles),
                "--max-atomic-steps",
                str(args.max_atomic_steps),
                "--max-expanded-nodes",
                str(args.evaluation_max_expanded_nodes),
                "--capture-search-replay",
            ]
            run(evaluation_command, iteration_dir / "evaluation.log")
            evaluation_path = evaluation_dir / "iteration_evaluation.json"
            entry["evaluation"] = str(evaluation_path.resolve())
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            acceptance = student_acceptance_from_evaluation(
                evaluation,
                min_score=args.evaluation_min_raw_score,
                min_side_score=args.evaluation_min_raw_side_score,
            )
            entry["acceptance"] = acceptance
            if not acceptance["accepted"]:
                rejected_checkpoint = current
                current = Path(str(entry["parent_checkpoint"])).expanduser().resolve()
                entry["student_checkpoint"] = str(current)
                entry["rejected_checkpoint"] = str(rejected_checkpoint)
                entry["status"] = "rejected"
                write_manifest(manifest_path, manifest)
                if args.stop_on_rejected_student:
                    break
                continue
        entry["status"] = "complete"
        write_manifest(manifest_path, manifest)

    manifest["completed_at"] = time.time()
    manifest["final_checkpoint"] = str(current)
    write_manifest(manifest_path, manifest)
    print(json.dumps({"event": "iterative_puct_distill_complete", **manifest}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
