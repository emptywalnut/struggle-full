#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable multi-day mixed PUCT/PPO campaign.")
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-hours", type=float, default=72.0)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--games-per-cycle", type=int, default=128)
    parser.add_argument("--search-fraction", type=float, default=0.75)
    parser.add_argument("--search-workers", type=int, default=32)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--learner-gpu", default="4")
    parser.add_argument("--inference-replicas-per-gpu", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=89_000_000)
    parser.add_argument("--distill-updates", type=int, default=200)
    parser.add_argument("--ppo-env-runners", type=int, default=16)
    parser.add_argument("--ppo-train-batch-size", type=int, default=4096)
    parser.add_argument("--ppo-minibatch-size", type=int, default=256)
    parser.add_argument("--eval-seed-pairs", type=int, default=10)
    parser.add_argument("--large-eval-every-cycles", type=int, default=4)
    parser.add_argument("--large-eval-seed-pairs", type=int, default=30)
    parser.add_argument("--d4-states", type=int, default=8)
    parser.add_argument("--d8-states", type=int, default=0)
    parser.add_argument("--checkpoint-sweep-seed-pairs", type=int, default=4)
    parser.add_argument("--compatibility-gpu", default="5")
    parser.add_argument(
        "--compatibility-positions",
        default=(
            "olympics_t1,olympics_t3,olympics_t5,destalinization_t1,"
            "europe_scoring_t2,mexico_t5_04,mexico_t5_05,iran_coup_t7,"
            "final_scoring_t10"
        ),
    )
    parser.add_argument("--adaptive-distill-after-wins", type=int, default=2)
    parser.add_argument("--working-raw-floor", type=float, default=0.45)
    parser.add_argument("--working-d2-floor", type=float, default=0.45)
    parser.add_argument("--working-side-floor", type=float, default=0.30)
    parser.add_argument("--promotion-score", type=float, default=0.55)
    parser.add_argument("--promotion-side-floor", type=float, default=0.45)
    parser.add_argument("--promotion-ci-low", type=float, default=0.50)
    parser.add_argument("--wandb-project", default="twilight-struggle-puct-campaign")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_stage(name: str, command: list[str], cycle_dir: Path, *, cuda: str | None) -> None:
    marker = cycle_dir / f".{name}.complete"
    if marker.exists():
        return
    log_path = cycle_dir / f"{name}.log"
    env = dict(os.environ)
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": "stage_start", "stage": name, "command": command}) + "\n")
        log.flush()
        subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    marker.write_text("complete\n", encoding="utf-8")


def checkpoint_from_distill(path: Path) -> Path:
    report = json.loads((path / "distillation_report.json").read_text(encoding="utf-8"))
    return Path(str(report["checkpoint"])).expanduser().resolve()


def checkpoint_from_log(path: Path) -> Path:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if line.startswith("checkpoint="):
            return Path(line.split("=", 1)[1]).expanduser().resolve()
    raise RuntimeError(f"missing checkpoint marker in {path}")


def evaluation_summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "results.json").read_text(encoding="utf-8"))["summary"]


def eval_command(
    args: argparse.Namespace,
    candidate: Path,
    opponent: Path,
    output: Path,
    *,
    search: bool,
    seed_pairs: int,
    seed_base: int,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/eval_puct_agent_pair_parallel.py",
        "--candidate-checkpoint", str(candidate),
        "--opponent-checkpoint", str(opponent),
        "--candidate-label", "candidate",
        "--opponent-label", "accepted-parent",
        "--output-dir", str(output),
        "--seed-pairs", str(seed_pairs),
        "--seed-base", str(seed_base),
        "--workers", str(min(args.search_workers, seed_pairs * 2)),
        "--gpus", str(args.gpus),
        "--replicas-per-checkpoint", "1",
        "--max-inference-batch-size", "32",
        "--max-inference-delay-ms", "3",
        "--meaningful-depth", "2",
        "--belief-particles", "32",
        "--initial-simulations", "32",
        "--medium-simulations", "32",
        "--max-simulations", "32",
    ]
    command.append("--search" if search else "--no-search")
    return command


def log_wandb(args: argparse.Namespace, state: dict[str, Any], metrics: dict[str, Any]) -> None:
    if args.wandb_mode == "disabled":
        return
    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            id=state.setdefault("wandb_run_id", f"ts-puct-{args.output_dir.name}"),
            resume="allow",
            mode=args.wandb_mode,
            dir=str(args.output_dir / "wandb"),
        )
        run.log(metrics, step=int(state["cycle"]))
        run.finish()
    except Exception as exc:
        with (args.output_dir / "wandb_errors.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{time.time()} {exc!r}\n")


def campaign_identity(args: argparse.Namespace) -> dict[str, Any]:
    runtime_upgrade_keys = {
        "checkpoint_sweep_seed_pairs",
        "compatibility_gpu",
        "compatibility_positions",
        "adaptive_distill_after_wins",
    }
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"dry_run", *runtime_upgrade_keys}
    }


def distillation_checkpoints(report: dict[str, Any]) -> list[tuple[str, Path]]:
    checkpoints = [
        (f"u{int(item['update']):03d}", Path(str(item["checkpoint"])).resolve())
        for item in report.get("intermediate_checkpoints", [])
    ]
    final_update = int(report["updates"])
    checkpoints.append((f"u{final_update:03d}", Path(str(report["checkpoint"])).resolve()))
    return checkpoints


def apply_compatibility_adaptation(state: dict[str, Any], compatibility: dict[str, Any] | None) -> None:
    if not compatibility:
        return
    diagnosis = str(compatibility.get("diagnosis") or "")
    state["last_compatibility_diagnosis"] = diagnosis
    if diagnosis == "value_degradation":
        state["distill_lambda_value_override"] = 0.125
        state["distill_lambda_value_anchor_override"] = 0.20
    elif diagnosis == "policy_prior_degradation":
        state["distill_lambda_anchor_override"] = 0.35
        state["distill_search_policy_mix_override"] = 0.75
    elif diagnosis == "policy_value_interaction":
        state["distill_parameter_scope_override"] = "actor_value_heads"
        state["distill_lambda_value_override"] = 0.125
        state["distill_lambda_anchor_override"] = 0.30
    elif diagnosis == "mixed_or_shared_trunk":
        state["distill_lambda_value_override"] = 0.125
        state["distill_lambda_anchor_override"] = 0.35
        state["distill_search_policy_mix_override"] = 0.75


def campaign_should_continue(
    state: dict[str, Any],
    output_dir: Path,
    max_cycles: int,
    *,
    now: float | None = None,
) -> bool:
    current_cycle = int(state["cycle"])
    if current_cycle >= int(max_cycles):
        return False
    next_cycle_dir = output_dir / "cycles" / f"cycle-{current_cycle + 1:04d}"
    incomplete_started_cycle = (
        (next_cycle_dir / "cycle_config.json").is_file()
        and not (next_cycle_dir / "cycle_report.json").is_file()
    )
    return incomplete_started_cycle or float(now if now is not None else time.time()) < float(
        state["deadline"]
    )
def main() -> None:
    args = parse_args()
    if not 0.7 <= args.search_fraction <= 0.8:
        raise SystemExit("--search-fraction must be in [0.7, 0.8]")
    if args.games_per_cycle < 8:
        raise SystemExit("--games-per-cycle must be at least 8")
    args.initial_checkpoint = args.initial_checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "wandb").mkdir(exist_ok=True)
    identity = campaign_identity(args)
    identity_path = args.output_dir / "campaign_config.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise SystemExit(f"campaign configuration mismatch: {identity_path}")
    else:
        atomic_json(identity_path, identity)
    state_path = args.output_dir / "campaign_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "status": "running",
            "cycle": 0,
            "started_at": time.time(),
            "deadline": time.time() + float(args.duration_hours) * 3600.0,
            "initial_checkpoint": str(args.initial_checkpoint),
            "accepted_parent": str(args.initial_checkpoint),
            "working_checkpoint": str(args.initial_checkpoint),
            "promotion_candidates": [],
            "cycles": [],
        }
        atomic_json(state_path, state)
    if args.dry_run:
        print(json.dumps({"config": identity, "state": state}, indent=2, sort_keys=True))
        return

    while campaign_should_continue(state, args.output_dir, args.max_cycles):
        cycle = int(state["cycle"]) + 1
        cycle_dir = args.output_dir / "cycles" / f"cycle-{cycle:04d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        source = Path(str(state["working_checkpoint"])).resolve()
        accepted = Path(str(state["accepted_parent"])).resolve()
        search_games = max(1, round(args.games_per_cycle * args.search_fraction))
        ppo_games = max(1, args.games_per_cycle - search_games)
        cycle_seed = int(args.seed_base) + cycle * 1_000_000
        effective_distill_updates = int(state.get("distill_updates_override", args.distill_updates))
        distill_parameter_scope = str(state.get("distill_parameter_scope_override", "actor_value"))
        distill_lambda_anchor = float(state.get("distill_lambda_anchor_override", 0.20))
        distill_lambda_value = float(state.get("distill_lambda_value_override", 0.25))
        distill_lambda_value_anchor = float(
            state.get("distill_lambda_value_anchor_override", 0.10)
        )
        distill_search_policy_mix = float(
            state.get("distill_search_policy_mix_override", 1.0)
        )
        atomic_json(
            cycle_dir / "cycle_config.json",
            {
                "cycle": cycle,
                "source_checkpoint": str(source),
                "accepted_parent": str(accepted),
                "search_games": search_games,
                "ppo_games": ppo_games,
                "seed_base": cycle_seed,
                "distill_updates": effective_distill_updates,
                "distill_parameter_scope": distill_parameter_scope,
                "distill_lambda_anchor": distill_lambda_anchor,
                "distill_lambda_value": distill_lambda_value,
                "distill_lambda_value_anchor": distill_lambda_value_anchor,
                "distill_search_policy_mix": distill_search_policy_mix,
            },
        )

        collection = cycle_dir / "puct_selfplay"
        run_stage(
            "collect",
            [
                sys.executable, "scripts/collect_puct_selfplay_parallel.py",
                "--checkpoint", str(source),
                "--output-dir", str(collection),
                "--games", str(search_games),
                "--seed-base", str(cycle_seed),
                "--gpus", str(args.gpus),
                "--workers", str(args.search_workers),
                "--inference-replicas-per-gpu", str(args.inference_replicas_per_gpu),
                "--max-inference-batch-size", "32",
                "--max-inference-delay-ms", "3",
                "--meaningful-depth", "2",
                "--belief-particles", "32",
                "--initial-simulations", "32",
                "--medium-simulations", "32",
                "--max-simulations", "32",
                "--search-temperature", "1.0",
                "--behavior-temperature", "0.85",
            ],
            cycle_dir,
            cuda=args.gpus,
        )
        replay_args = ["--search-replay", str(collection / "search_replay")]

        if args.d4_states > 0:
            d4_all = cycle_dir / "d4_candidates_all.json"
            d4_selected = cycle_dir / "d4_candidates.json"
            d4_output = cycle_dir / "d4_adjudication"
            run_stage(
                "build_d4_candidates",
                [sys.executable, "scripts/build_d4_candidate_manifest.py", "--collection-dir", str(collection), "--output", str(d4_all)],
                cycle_dir,
                cuda=args.learner_gpu,
            )
            run_stage(
                "select_d4_candidates",
                [sys.executable, "scripts/select_d4_adjudication.py", "--d2-position-manifest", str(d4_all), "--output", str(d4_selected), "--max-positions", str(args.d4_states), "--max-per-game", "1"],
                cycle_dir,
                cuda=args.learner_gpu,
            )
            run_stage(
                "d4_adjudication",
                [
                    sys.executable, "scripts/generate_search_replay_parallel.py",
                    "--checkpoint", str(source), "--position-manifest", str(d4_selected),
                    "--output-dir", str(d4_output), "--max-searches", str(args.d4_states),
                    "--gpus", str(args.gpus), "--workers", str(min(args.search_workers, args.d4_states)),
                    "--inference-replicas-per-gpu", str(args.inference_replicas_per_gpu),
                    "--meaningful-depth", "4", "--belief-particles", "32",
                    "--initial-simulations", "128", "--medium-simulations", "128", "--max-simulations", "128",
                ],
                cycle_dir,
                cuda=args.gpus,
            )
            replay_args.extend(["--search-replay", str(d4_output / "search_replay")])
            if args.d8_states > 0:
                d8_selected = cycle_dir / "d8_candidates.json"
                run_stage(
                    "select_d8_candidates",
                    [
                        sys.executable, "scripts/select_d8_adjudication.py",
                        "--d2-position-manifest", str(d4_selected),
                        "--d4-search-replay", str(d4_output / "search_replay"),
                        "--output", str(d8_selected),
                        "--max-positions", str(args.d8_states),
                        "--completion-aware",
                    ],
                    cycle_dir,
                    cuda=args.learner_gpu,
                )
                selected_d8 = json.loads(d8_selected.read_text(encoding="utf-8")).get("positions", [])
                if selected_d8:
                    d8_output = cycle_dir / "d8_adjudication"
                    run_stage(
                        "d8_adjudication",
                        [
                            sys.executable, "scripts/generate_search_replay_parallel.py",
                            "--checkpoint", str(source), "--position-manifest", str(d8_selected),
                            "--output-dir", str(d8_output), "--max-searches", str(len(selected_d8)),
                            "--gpus", str(args.gpus), "--workers", str(len(selected_d8)),
                            "--inference-replicas-per-gpu", "1", "--meaningful-depth", "8",
                            "--belief-particles", "32", "--initial-simulations", "2048",
                            "--medium-simulations", "2048", "--max-simulations", "2048",
                            "--max-atomic-steps", "2048", "--max-expanded-nodes", "8192",
                        ],
                        cycle_dir,
                        cuda=args.gpus,
                    )
                    replay_args.extend(["--search-replay", str(d8_output / "search_replay")])

        student_dir = cycle_dir / "search_student"
        optimizer_restore_args = (
            [] if distill_parameter_scope == "actor_value" else ["--no-restore-optimizer-state"]
        )
        run_stage(
            "distill",
            [
                sys.executable, "scripts/distill_search_replay.py",
                "--restore-from", str(source), *replay_args,
                "--output-dir", str(student_dir),
                "--updates", str(effective_distill_updates),
                *optimizer_restore_args,
                "--checkpoint-every-updates", "50",
                "--batch-size", "128", "--separate-value-batch-size", "64",
                "--lr", "0.000001", "--parameter-scope", distill_parameter_scope,
                "--lambda-search", "1.0", "--search-policy-mix", str(distill_search_policy_mix),
                "--lambda-anchor", str(distill_lambda_anchor), "--anchor-replay-batch-size", "128",
                "--lambda-value", str(distill_lambda_value), "--lambda-search-q", "0.01",
                "--lambda-value-anchor", str(distill_lambda_value_anchor), "--confidence-weight-floor", "0.25",
                "--require-complete-game", "--no-require-flip", "--balance-sides",
                "--diversity-max-per-game", "16", "--diversity-max-per-turn", "24",
                "--diversity-max-per-card", "12", "--diversity-max-per-region", "16",
                "--diversity-max-per-trigger", "48", "--num-gpus", "1.0",
                "--seed", str(cycle_seed + 701),
            ],
            cycle_dir,
            cuda=args.learner_gpu,
        )
        distillation_report = json.loads(
            (student_dir / "distillation_report.json").read_text(encoding="utf-8")
        )
        distilled = checkpoint_from_distill(student_dir)

        ppo_dir = cycle_dir / "ppo_stabilization"
        recent_checkpoints = [
            str(json.loads(Path(str(item["report"])).read_text(encoding="utf-8"))["candidate_checkpoint"])
            for item in state.get("cycles", [])[-3:]
            if item.get("working_pass") and Path(str(item.get("report") or "")).is_file()
        ]
        anchors = json.dumps(
            list(dict.fromkeys([str(accepted), str(source), *recent_checkpoints]))
        )
        ppo_mix = json.dumps(
            {"current_self": 0.50, "anchor": 0.30, "random_legal": 0.10, "heuristic": 0.10}
        )
        run_stage(
            "ppo",
            [
                sys.executable, "-m", "struggle_ai.train_rllib",
                "--multi-agent", "--policy-sharing", "unified", "--side-specific-heads",
                "--unified-train-focus-side", "both", "--model-arch", "transformer_history",
                "--hidden", "256", "--history-layers", "2", "--card-history-layers", "2",
                "--history-attention-heads", "4", "--history-dropout", "0.05",
                "--graph-layers", "2", "--graph-neighbor-hops", "2",
                "--restore-from", str(distilled), "--stop-episodes", str(ppo_games),
                "--num-env-runners", str(args.ppo_env_runners), "--num-gpus", "1.0",
                "--ray-num-cpus", str(max(4, args.ppo_env_runners + 4)),
                "--torch-num-threads", "1",
                "--num-gpus-per-env-runner", "0.0", "--train-batch-size", str(args.ppo_train_batch_size),
                "--minibatch-size", str(args.ppo_minibatch_size), "--num-epochs", "4",
                "--lr", "0.000005", "--entropy-coeff", "0.008", "--policy-temperature", "1.15",
                "--batch-mode", "complete_episodes", "--persistent-random-prob", "0.02",
                "--unified-opponent-mix-json", ppo_mix, "--unified-anchor-policy-from", anchors,
                "--checkpoint-dir", str(ppo_dir), "--checkpoint-eval-every-steps", "0",
                "--checkpoint-eval-every-episodes", "0", "--no-final-eval", "--no-force-setup-heuristic",
                "--defcon-suicide-mode", "hard_filter", "--terminal-reward-scale", "1.0",
                "--reward-shaping-scale", "0.0", "--log-games-dir", str(ppo_dir / "game_logs"),
                "--log-games-every", "8", "--metrics-dir", str(ppo_dir / "metrics"),
            ],
            cycle_dir,
            cuda=args.learner_gpu,
        )
        ppo_candidate = checkpoint_from_log(cycle_dir / "ppo.log")
        search_optimizer = distilled / "search_distill_optimizers.pt"
        if search_optimizer.exists():
            shutil.copy2(search_optimizer, ppo_candidate / search_optimizer.name)

        sweep_dir = cycle_dir / "checkpoint_sweep"
        sweep_checkpoints = [
            item
            for label, checkpoint in distillation_checkpoints(distillation_report)
            for item in ("--checkpoint", f"{label}={checkpoint}")
        ]
        sweep_checkpoints.extend(["--checkpoint", f"ppo_final={ppo_candidate}"])
        run_stage(
            "checkpoint_sweep",
            [
                sys.executable,
                "scripts/sweep_puct_checkpoints.py",
                "--parent-checkpoint", str(accepted),
                *sweep_checkpoints,
                "--output-dir", str(sweep_dir),
                "--distillation-report", str(student_dir / "distillation_report.json"),
                "--distill-log", str(cycle_dir / "distill.log"),
                "--final-label", "ppo_final",
                "--seed-pairs", str(args.checkpoint_sweep_seed_pairs),
                "--seed-base", str(cycle_seed + 400_000),
                "--workers", str(min(args.search_workers, args.checkpoint_sweep_seed_pairs * 2)),
                "--gpus", str(args.gpus),
            ],
            cycle_dir,
            cuda=args.gpus,
        )
        checkpoint_sweep = json.loads(
            (sweep_dir / "checkpoint_sweep.json").read_text(encoding="utf-8")
        )
        candidate = Path(
            str(checkpoint_sweep["selection"]["selected_checkpoint"])
        ).resolve()

        raw_dir = cycle_dir / "evaluation" / "raw"
        d2_dir = cycle_dir / "evaluation" / "d2"
        run_stage("eval_raw", eval_command(args, candidate, accepted, raw_dir, search=False, seed_pairs=args.eval_seed_pairs, seed_base=cycle_seed + 500_000), cycle_dir, cuda=args.gpus)
        run_stage("eval_d2", eval_command(args, candidate, accepted, d2_dir, search=True, seed_pairs=args.eval_seed_pairs, seed_base=cycle_seed + 600_000), cycle_dir, cuda=args.gpus)
        raw = evaluation_summary(raw_dir)
        d2 = evaluation_summary(d2_dir)
        compatibility: dict[str, Any] | None = None
        compatibility_needed = (
            cycle == 4
            or (
                float(raw["score"]) >= args.working_raw_floor
                and float(d2["score"]) + 0.05 < float(raw["score"])
            )
        )
        if compatibility_needed:
            compatibility_dir = cycle_dir / "policy_value_compatibility"
            run_stage(
                "policy_value_compatibility",
                [
                    sys.executable,
                    "scripts/run_policy_value_compatibility.py",
                    "--parent-checkpoint", str(accepted),
                    "--child-checkpoint", str(candidate),
                    "--output-dir", str(compatibility_dir),
                    "--positions", str(args.compatibility_positions),
                    "--belief-particles", "32",
                    "--budget", "32",
                    "--gpu", str(args.compatibility_gpu),
                    "--seed", str(cycle_seed + 450_000),
                ],
                cycle_dir,
                cuda=args.compatibility_gpu,
            )
            compatibility = json.loads(
                (compatibility_dir / "compatibility_report.json").read_text(encoding="utf-8")
            )
        side_floor = min(
            float(raw["by_candidate_side"]["us"]["score"]),
            float(raw["by_candidate_side"]["ussr"]["score"]),
            float(d2["by_candidate_side"]["us"]["score"]),
            float(d2["by_candidate_side"]["ussr"]["score"]),
        )
        working_pass = (
            float(raw["score"]) >= args.working_raw_floor
            and float(d2["score"]) >= args.working_d2_floor
            and side_floor >= args.working_side_floor
        )

        large: dict[str, Any] | None = None
        preliminary_promotion = (
            float(raw["score"]) >= args.promotion_score
            and float(d2["score"]) >= args.promotion_score
            and side_floor >= args.promotion_side_floor
        )
        if preliminary_promotion or cycle % max(1, args.large_eval_every_cycles) == 0:
            large_raw_dir = cycle_dir / "evaluation" / "large_raw"
            large_d2_dir = cycle_dir / "evaluation" / "large_d2"
            run_stage("eval_large_raw", eval_command(args, candidate, accepted, large_raw_dir, search=False, seed_pairs=args.large_eval_seed_pairs, seed_base=cycle_seed + 700_000), cycle_dir, cuda=args.gpus)
            run_stage("eval_large_d2", eval_command(args, candidate, accepted, large_d2_dir, search=True, seed_pairs=args.large_eval_seed_pairs, seed_base=cycle_seed + 800_000), cycle_dir, cuda=args.gpus)
            large = {"raw": evaluation_summary(large_raw_dir), "d2": evaluation_summary(large_d2_dir)}
            run_stage(
                "frozen_diagnostics",
                [
                    sys.executable, "scripts/run_frozen_diagnostics.py",
                    "--checkpoint", str(candidate),
                    "--output-dir", str(cycle_dir / "frozen_diagnostics"),
                    "--belief-particles", "32", "--budgets", "32",
                    "--depth1-budget", "32", "--depth3-budget", "32",
                    "--positions",
                    "olympics_t1,olympics_t3,olympics_t5,destalinization_t1,"
                    "europe_scoring_t2,mexico_t5_04,mexico_t5_05,iran_coup_t7,"
                    "final_scoring_t10",
                    "--root-search-only",
                    "--num-gpus", "1.0",
                ],
                cycle_dir,
                cuda=args.learner_gpu,
            )

        promotion_candidate = bool(
            large
            and float(large["raw"]["score"]) >= args.promotion_score
            and float(large["d2"]["score"]) >= args.promotion_score
            and float(large["raw"]["paired_bootstrap_ci95_low"]) >= args.promotion_ci_low
            and float(large["d2"]["paired_bootstrap_ci95_low"]) >= args.promotion_ci_low
            and min(
                float(large["raw"]["by_candidate_side"][side]["score"])
                for side in ("us", "ussr")
            ) >= args.promotion_side_floor
            and min(
                float(large["d2"]["by_candidate_side"][side]["score"])
                for side in ("us", "ussr")
            ) >= args.promotion_side_floor
        )
        cycle_report = {
            "cycle": cycle,
            "source_checkpoint": str(source),
            "candidate_checkpoint": str(candidate),
            "accepted_parent": str(accepted),
            "working_pass": working_pass,
            "promotion_candidate": promotion_candidate,
            "raw": raw,
            "d2": d2,
            "large": large,
            "checkpoint_sweep": checkpoint_sweep,
            "policy_value_compatibility": compatibility,
            "collection": json.loads((collection / "collection_summary.json").read_text(encoding="utf-8")),
            "distillation": distillation_report,
        }
        atomic_json(cycle_dir / "cycle_report.json", cycle_report)
        state["cycle"] = cycle
        if working_pass:
            state["working_checkpoint"] = str(candidate)
        else:
            state["working_checkpoint"] = str(accepted)
        if promotion_candidate:
            state["promotion_candidates"].append(str(cycle_dir / "cycle_report.json"))
        sweep_selection = checkpoint_sweep["selection"]
        if sweep_selection.get("intermediate_clearly_dominates_final") and sweep_selection.get("selected_update"):
            state["overtraining_intermediate_wins"] = int(
                state.get("overtraining_intermediate_wins", 0)
            ) + 1
            state.setdefault("overtraining_selected_updates", []).append(
                int(sweep_selection["selected_update"])
            )
        else:
            state["overtraining_intermediate_wins"] = 0
        if int(state.get("overtraining_intermediate_wins", 0)) >= int(args.adaptive_distill_after_wins):
            recent_updates = state.get("overtraining_selected_updates", [])[-int(args.adaptive_distill_after_wins):]
            state["distill_updates_override"] = max(50, min(int(value) for value in recent_updates))
        apply_compatibility_adaptation(state, compatibility)
        state["cycles"].append(
            {
                "cycle": cycle,
                "report": str(cycle_dir / "cycle_report.json"),
                "working_pass": working_pass,
                "promotion_candidate": promotion_candidate,
            }
        )
        atomic_json(state_path, state)
        log_wandb(
            args,
            state,
            {
                "campaign/cycle": cycle,
                "campaign/working_pass": int(working_pass),
                "campaign/promotion_candidate": int(promotion_candidate),
                "eval/raw_score": raw["score"],
                "eval/raw_us": raw["by_candidate_side"]["us"]["score"],
                "eval/raw_ussr": raw["by_candidate_side"]["ussr"]["score"],
                "eval/d2_score": d2["score"],
                "eval/d2_us": d2["by_candidate_side"]["us"]["score"],
                "eval/d2_ussr": d2["by_candidate_side"]["ussr"]["score"],
                "eval/d2_flip_rate": d2["search_flip_rate"],
                "collection/games_per_hour": cycle_report["collection"]["games_per_hour"],
                "collection/search_flip_rate": cycle_report["collection"]["search_flip_rate"],
                "sweep/selected_update": sweep_selection.get("selected_update") or -1,
                "sweep/intermediate_dominates": int(
                    bool(sweep_selection.get("intermediate_clearly_dominates_final"))
                ),
                **{
                    f"sweep/{label}/{mode}_score": float(record[mode]["score"])
                    for label, record in checkpoint_sweep["checkpoints"].items()
                    for mode in ("raw", "d2")
                },
                **(
                    {
                        f"compatibility/{name}": float(value)
                        for name, value in compatibility["agreement_with_parent_search"].items()
                    }
                    if compatibility
                    else {}
                ),
                **{
                    key.replace("search_distill/", "train/"): value
                    for key, value in cycle_report["distillation"]["last_metrics"].items()
                    if isinstance(value, (int, float))
                },
            },
        )

    state["status"] = "complete" if time.time() >= float(state["deadline"]) else "stopped"
    state["finished_at"] = time.time()
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
