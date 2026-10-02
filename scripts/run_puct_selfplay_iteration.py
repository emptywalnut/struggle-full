#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One resumable PUCT self-play policy-improvement iteration.")
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-games", type=int, default=20)
    parser.add_argument("--search-fraction", type=float, choices=(0.5, 0.75, 1.0), default=0.75)
    parser.add_argument("--seed-base", type=int, default=85_000_000)
    parser.add_argument("--gpu", default="4")
    parser.add_argument(
        "--eval-gpu",
        default="5",
        help="Deprecated compatibility option; paired evaluation uses --parallel-inference-gpus.",
    )
    parser.add_argument("--parallel-inference-gpus", default="4,5")
    parser.add_argument("--parallel-search-workers", type=int, default=32)
    parser.add_argument("--inference-replicas-per-gpu", type=int, default=2)
    parser.add_argument(
        "--eval-inference-replicas-per-checkpoint",
        type=int,
        default=1,
        help="Evaluation servers per distinct checkpoint; one avoids fragmenting sparse paired-eval batches.",
    )
    parser.add_argument("--max-inference-batch-size", type=int, default=32)
    parser.add_argument("--max-inference-delay-ms", type=float, default=3.0)
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--search-simulations", type=int, default=32)
    parser.add_argument("--behavior-temperature", type=float, default=0.75)
    parser.add_argument("--distill-updates", type=int, default=200)
    parser.add_argument("--distill-batch-size", type=int, default=128)
    parser.add_argument("--distill-lr", type=float, default=2e-6)
    parser.add_argument(
        "--distill-parameter-scope",
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
    parser.add_argument("--lambda-policy", type=float, default=1.0)
    parser.add_argument(
        "--distill-search-policy-mix",
        type=float,
        default=0.5,
        help="Blend PUCT visits with the parent's raw policy at each supervised state.",
    )
    parser.add_argument("--lambda-value", type=float, default=0.1)
    parser.add_argument("--lambda-search-q", type=float, default=0.02)
    parser.add_argument("--lambda-anchor", type=float, default=0.3)
    parser.add_argument(
        "--min-value-examples-per-batch",
        type=int,
        default=32,
        help="Minimum complete-game terminal targets sampled in each distillation batch.",
    )
    parser.add_argument(
        "--separate-value-batch-size",
        type=int,
        default=64,
        help="Independent complete-root value batch used with targeted policy labels.",
    )
    parser.add_argument(
        "--distill-require-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict actor supervision to decisions where PUCT changed the raw action.",
    )
    parser.add_argument(
        "--distill-min-q-advantage",
        type=float,
        default=0.02,
        help="Minimum PUCT Q improvement over the raw action for actor supervision.",
    )
    parser.add_argument(
        "--distill-min-teacher-visit-share",
        type=float,
        default=0.20,
        help="Minimum root visit share assigned to the PUCT teacher action.",
    )
    parser.add_argument(
        "--distill-us-policy-weight",
        type=float,
        default=1.0,
        help="Relative search-policy loss weight for US replay examples.",
    )
    parser.add_argument(
        "--distill-ussr-policy-weight",
        type=float,
        default=1.0,
        help="Relative search-policy loss weight for USSR replay examples.",
    )
    parser.add_argument(
        "--lambda-value-anchor",
        type=float,
        default=1.0,
        help="Regularize critic outputs against the frozen parent during joint search training.",
    )
    parser.add_argument("--d4-adjudication-states", type=int, default=16)
    parser.add_argument("--d4-adjudication-workers", type=int, default=16)
    parser.add_argument("--d4-adjudication-simulations", type=int, default=128)
    parser.add_argument("--deep-adjudication-states", type=int, default=2)
    parser.add_argument("--deep-adjudication-depth", type=int, default=8)
    parser.add_argument("--deep-adjudication-simulations", type=int, default=2048)
    parser.add_argument("--ppo-env-runners", type=int, default=2)
    parser.add_argument("--ppo-train-batch-size", type=int, default=2048)
    parser.add_argument("--ppo-minibatch-size", type=int, default=256)
    parser.add_argument("--eval-seed-pairs", type=int, default=10)
    parser.add_argument("--run-practical-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-raw-score", type=float, default=0.45)
    parser.add_argument("--minimum-side-score", type=float, default=0.30)
    parser.add_argument("--minimum-cheap-search-score", type=float, default=0.50)
    parser.add_argument(
        "--diagnostic-pass-file",
        type=Path,
        help="JSON approval artifact with passed=true and child_checkpoint matching the evaluated child.",
    )
    return parser.parse_args()


def run_stage(name: str, command: list[str], output_dir: Path, gpu: str) -> None:
    marker = output_dir / f".{name}.complete"
    if marker.exists():
        return
    log_path = output_dir / f"{name}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": "stage_start", "name": name, "command": command}) + "\n")
        log.flush()
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    marker.write_text("complete\n", encoding="utf-8")


def checkpoint_from_report(path: Path) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Path(str(payload["checkpoint"])).expanduser().resolve()


def checkpoint_from_log(path: Path) -> Path:
    matches = re.findall(r"^checkpoint=(.+)$", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if not matches:
        raise RuntimeError(f"could not find final checkpoint in {path}")
    return Path(matches[-1]).expanduser().resolve()


def eval_command(
    args: argparse.Namespace,
    child: Path,
    output: Path,
    *,
    search: bool,
    depth: int,
    simulations: int,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/eval_puct_agent_pair_parallel.py",
        "--candidate-checkpoint", str(child),
        "--opponent-checkpoint", str(args.parent_checkpoint.expanduser().resolve()),
        "--candidate-label", "child",
        "--opponent-label", "parent",
        "--output-dir", str(output),
        "--seed-pairs", str(args.eval_seed_pairs),
        "--seed-base", str(args.seed_base + 500_000),
        "--workers", str(min(args.parallel_search_workers, args.eval_seed_pairs * 2)),
        "--gpus", str(args.parallel_inference_gpus),
        "--replicas-per-checkpoint", str(args.eval_inference_replicas_per_checkpoint),
        "--max-inference-batch-size", str(args.max_inference_batch_size),
        "--max-inference-delay-ms", str(args.max_inference_delay_ms),
        "--meaningful-depth", str(depth),
        "--initial-simulations", str(simulations),
        "--medium-simulations", str(simulations),
        "--max-simulations", str(simulations),
    ]
    command.append("--search" if search else "--no-search")
    return command


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "results.json").read_text(encoding="utf-8"))["summary"]


def diagnostic_gate(path: Path | None, child: Path) -> tuple[bool, dict[str, Any]]:
    if path is None:
        return False, {"reason": "no diagnostic pass artifact supplied"}
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    recorded_child = Path(str(payload.get("child_checkpoint") or "")).expanduser().resolve()
    passed = bool(payload.get("passed")) and recorded_child == child.resolve()
    return passed, {
        "artifact": str(resolved),
        "passed": bool(payload.get("passed")),
        "recorded_child_checkpoint": str(recorded_child),
        "checkpoint_matches": recorded_child == child.resolve(),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.search_fraction <= 1.0:
        raise SystemExit("--search-fraction must be in (0, 1]")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    search_games = max(1, round(args.total_games * args.search_fraction))
    ppo_games = max(0, args.total_games - search_games)
    config = {**vars(args), "parent_checkpoint": str(args.parent_checkpoint), "output_dir": str(args.output_dir), "search_games": search_games, "ppo_games": ppo_games}
    (args.output_dir / "iteration_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    collection = args.output_dir / "puct_selfplay"
    run_stage(
        "collect",
        [
            sys.executable, "scripts/collect_puct_selfplay_parallel.py",
            "--checkpoint", str(args.parent_checkpoint),
            "--output-dir", str(collection),
            "--games", str(search_games),
            "--seed-base", str(args.seed_base),
            "--gpus", str(args.parallel_inference_gpus),
            "--workers", str(args.parallel_search_workers),
            "--inference-replicas-per-gpu", str(args.inference_replicas_per_gpu),
            "--max-inference-batch-size", str(args.max_inference_batch_size),
            "--max-inference-delay-ms", str(args.max_inference_delay_ms),
            "--meaningful-depth", str(args.search_depth),
            "--initial-simulations", str(args.search_simulations),
            "--medium-simulations", str(args.search_simulations),
            "--max-simulations", str(args.search_simulations),
            "--behavior-temperature", str(args.behavior_temperature),
        ],
        args.output_dir,
        args.parallel_inference_gpus,
    )

    replay_arguments = ["--search-replay", str(collection / "search_replay")]
    d4_output = args.output_dir / "d4_adjudication"
    if args.d4_adjudication_states > 0:
        all_d4_positions = args.output_dir / "d4_candidate_roots.json"
        run_stage(
            "build_d4_candidates",
            [
                sys.executable, "scripts/build_d4_candidate_manifest.py",
                "--collection-dir", str(collection),
                "--output", str(all_d4_positions),
            ],
            args.output_dir,
            args.gpu,
        )
        d4_positions = args.output_dir / "d4_adjudication_candidates.json"
        run_stage(
            "select_d4_adjudication",
            [
                sys.executable, "scripts/select_d4_adjudication.py",
                "--d2-position-manifest", str(all_d4_positions),
                "--output", str(d4_positions),
                "--max-positions", str(args.d4_adjudication_states),
            ],
            args.output_dir,
            args.gpu,
        )
        run_stage(
            "d4_adjudication",
            [
                sys.executable, "scripts/generate_search_replay_parallel.py",
                "--checkpoint", str(args.parent_checkpoint),
                "--position-manifest", str(d4_positions),
                "--output-dir", str(d4_output),
                "--max-searches", str(args.d4_adjudication_states),
                "--gpus", str(args.parallel_inference_gpus),
                "--workers", str(args.d4_adjudication_workers),
                "--inference-replicas-per-gpu", str(args.inference_replicas_per_gpu),
                "--max-inference-batch-size", str(args.max_inference_batch_size),
                "--max-inference-delay-ms", str(args.max_inference_delay_ms),
                "--meaningful-depth", "4",
                "--initial-simulations", str(args.d4_adjudication_simulations),
                "--medium-simulations", str(args.d4_adjudication_simulations),
                "--max-simulations", str(args.d4_adjudication_simulations),
            ],
            args.output_dir,
            args.parallel_inference_gpus,
        )
        replay_arguments.extend(("--search-replay", str(d4_output / "search_replay")))
    if args.deep_adjudication_states > 0:
        if args.d4_adjudication_states <= 0:
            raise SystemExit("D8 adjudication requires a preceding D4 stage")
        d8_positions = args.output_dir / "d8_adjudication_candidates.json"
        run_stage(
            "select_d8_adjudication",
            [
                sys.executable, "scripts/select_d8_adjudication.py",
                "--d2-position-manifest", str(d4_positions),
                "--d4-search-replay", str(d4_output / "search_replay"),
                "--output", str(d8_positions),
                "--max-positions", str(args.deep_adjudication_states),
            ],
            args.output_dir,
            args.gpu,
        )
        selected_d8 = json.loads(d8_positions.read_text(encoding="utf-8")).get("positions", [])
        if selected_d8:
            deep_output = args.output_dir / "deep_adjudication"
            run_stage(
                "deep_adjudication",
                [
                    sys.executable, "scripts/generate_search_replay_parallel.py",
                    "--checkpoint", str(args.parent_checkpoint),
                    "--position-manifest", str(d8_positions),
                    "--output-dir", str(deep_output),
                    "--max-searches", str(args.deep_adjudication_states),
                    "--gpus", str(args.parallel_inference_gpus),
                    "--workers", str(min(args.d4_adjudication_workers, len(selected_d8))),
                    "--inference-replicas-per-gpu", str(args.inference_replicas_per_gpu),
                    "--max-inference-batch-size", str(args.max_inference_batch_size),
                    "--max-inference-delay-ms", str(args.max_inference_delay_ms),
                    "--meaningful-depth", str(args.deep_adjudication_depth),
                    "--initial-simulations", str(args.deep_adjudication_simulations),
                    "--medium-simulations", str(args.deep_adjudication_simulations),
                    "--max-simulations", str(args.deep_adjudication_simulations),
                    "--max-atomic-steps", "2048", "--max-expanded-nodes", "8192",
                    "--search-temperature", "1.0",
                ],
                args.output_dir,
                args.parallel_inference_gpus,
            )
            replay_arguments.extend(("--search-replay", str(deep_output / "search_replay")))

    distilled = args.output_dir / "search_student"
    run_stage(
        "distill",
        [
            sys.executable, "scripts/distill_search_replay.py",
            "--restore-from", str(args.parent_checkpoint),
            *replay_arguments,
            "--output-dir", str(distilled),
            "--updates", str(args.distill_updates),
            "--batch-size", str(args.distill_batch_size),
            "--lr", str(args.distill_lr),
            "--lambda-search", str(args.lambda_policy),
            "--search-policy-mix", str(args.distill_search_policy_mix),
            "--lambda-value", str(args.lambda_value),
            "--lambda-search-q", str(args.lambda_search_q),
            "--lambda-anchor", str(args.lambda_anchor),
            "--lambda-value-anchor", str(args.lambda_value_anchor),
            "--us-policy-weight", str(args.distill_us_policy_weight),
            "--ussr-policy-weight", str(args.distill_ussr_policy_weight),
            "--confidence-weight-floor", "0.1",
            "--min-q-advantage", str(args.distill_min_q_advantage),
            "--min-teacher-visit-share", str(args.distill_min_teacher_visit_share),
            "--min-value-examples-per-batch", str(args.min_value_examples_per_batch),
            "--separate-value-batch-size", str(args.separate_value_batch_size),
            "--parameter-scope", str(args.distill_parameter_scope),
            "--require-complete-game",
            "--require-flip" if args.distill_require_flip else "--no-require-flip",
            "--balance-sides",
            "--num-gpus", "1.0",
        ],
        args.output_dir,
        args.gpu,
    )
    child = checkpoint_from_report(distilled / "distillation_report.json")

    if ppo_games:
        ppo_dir = args.output_dir / "ppo_stabilization"
        run_stage(
            "ppo",
            [
                sys.executable, "-m", "struggle_ai.train_rllib",
                "--multi-agent", "--policy-sharing", "unified",
                "--unified-train-focus-side", "both",
                "--model-arch", "transformer_history",
                "--hidden", "256", "--history-layers", "2", "--card-history-layers", "2",
                "--history-attention-heads", "4", "--history-dropout", "0.05",
                "--graph-layers", "2", "--graph-neighbor-hops", "2",
                "--restore-from", str(child),
                "--stop-episodes", str(ppo_games),
                "--num-env-runners", str(args.ppo_env_runners),
                "--num-gpus", "1.0", "--num-gpus-per-env-runner", "0.0",
                "--train-batch-size", str(args.ppo_train_batch_size),
                "--minibatch-size", str(args.ppo_minibatch_size),
                "--batch-mode", "complete_episodes",
                "--checkpoint-dir", str(ppo_dir),
                "--checkpoint-eval-every-steps", "0",
                "--checkpoint-eval-every-episodes", "0",
                "--no-final-eval", "--no-force-setup-heuristic",
                "--defcon-suicide-mode", "hard_filter",
                "--terminal-reward-scale", "1.0", "--reward-shaping-scale", "0.0",
            ],
            args.output_dir,
            args.gpu,
        )
        child = checkpoint_from_log(args.output_dir / "ppo.log")

    evaluations = {
        "raw": (False, 2, 32),
        "cheap_d2_32": (True, 2, 32),
    }
    if args.run_practical_eval:
        evaluations["practical_d4_128"] = (True, 4, 128)
    for label, (search, depth, simulations) in evaluations.items():
        run_stage(
            f"eval_{label}",
            eval_command(args, child, args.output_dir / "evaluation" / label, search=search, depth=depth, simulations=simulations),
            args.output_dir,
            args.parallel_inference_gpus,
        )

    summaries = {label: load_summary(args.output_dir / "evaluation" / label) for label in evaluations}
    raw = summaries["raw"]
    cheap = summaries["cheap_d2_32"]
    side_scores = raw["by_candidate_side"]
    diagnostics_passed, diagnostic_evidence = diagnostic_gate(args.diagnostic_pass_file, child)
    promotion_checks = {
        "raw_score": float(raw["score"]) >= args.minimum_raw_score,
        "us_side": float(side_scores["us"]["score"]) >= args.minimum_side_score,
        "ussr_side": float(side_scores["ussr"]["score"]) >= args.minimum_side_score,
        "cheap_search_score": float(cheap["score"]) >= args.minimum_cheap_search_score,
        "frozen_diagnostics": diagnostics_passed,
    }
    report = {
        "parent_checkpoint": str(args.parent_checkpoint.expanduser().resolve()),
        "child_checkpoint": str(child),
        "search_games": search_games,
        "ppo_games": ppo_games,
        "evaluations": summaries,
        "diagnostic_evidence": diagnostic_evidence,
        "promotion_checks": promotion_checks,
        "promoted": all(promotion_checks.values()),
    }
    (args.output_dir / "promotion_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
