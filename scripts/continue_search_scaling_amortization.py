#!/usr/bin/env python3
"""Resume the Phase-3-u75 search-scaling campaign from durable artifacts.

This driver deliberately waits for, validates, and reuses the expensive search
artifacts that may already be running.  It fails closed before distillation if
complete-game or converged-teacher gates are not satisfied.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075"
VARIANCE = ROOT / "runs/search_scaling/phase3u75-variance-20260821"
HORIZON = VARIANCE / "horizon-adjudication"
PRIMARY = ROOT / "runs/puct_selfplay/search-scaling-phase3u75-selective-adaptive-d4d6-32games-w12-gpu45-20260821"
SUPPLEMENTAL = ROOT / "runs/puct_selfplay/search-scaling-phase3u75-selective-adaptive-d4d6-32games-w32-gpu45-20260822"
CAMPAIGN = ROOT / "runs/search_distill/search-scaling-phase3u75-amortization-u25-gpu4-20260821"
BROAD_PARENT_ANCHOR = ROOT / "runs/puct_selfplay/rootfix-phase4-d4-128-16games-w16-gpu45-20260819/search_replay/manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--supplemental-games", type=int, default=32)
    parser.add_argument("--supplemental-workers", type=int, default=4)
    parser.add_argument("--skip-supplemental", action="store_true")
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    record = {"event": event, "time": time.time(), **payload}
    print(json.dumps(record, sort_keys=True), flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for(paths: list[Path], *, poll_seconds: float, stage: str) -> None:
    while True:
        missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
        if not missing:
            emit("stage_inputs_ready", stage=stage, paths=len(paths))
            return
        emit("stage_waiting", stage=stage, missing=missing)
        time.sleep(max(1.0, poll_seconds))


def run_once(command: list[str], *, done_path: Path, stage: str, log_path: Path) -> None:
    if done_path.is_file():
        emit("stage_reused", stage=stage, done_path=str(done_path.relative_to(ROOT)))
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    emit("stage_started", stage=stage, command=command)
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not done_path.is_file():
        raise RuntimeError(
            f"{stage} failed: returncode={completed.returncode}, expected={done_path}"
        )
    emit("stage_complete", stage=stage, done_path=str(done_path.relative_to(ROOT)))


def validate_generation(path: Path, expected_roots: int) -> None:
    payload = read_json(path)
    completed = int(payload.get("roots_completed", payload.get("positions_completed", 0)))
    if completed != expected_roots:
        raise RuntimeError(f"incomplete root generation in {path}: {completed}/{expected_roots}")
    if int(payload.get("errors", 0)):
        raise RuntimeError(f"root generation errors in {path}: {payload.get('errors')}")


def validate_collection(path: Path, expected_games: int) -> dict[str, Any]:
    payload = read_json(path)
    completed = int(payload.get("games_completed", 0))
    if completed != expected_games:
        raise RuntimeError(f"incomplete self-play collection in {path}: {completed}/{expected_games}")
    if int(payload.get("worker_errors", payload.get("errors", 0))):
        raise RuntimeError(f"self-play worker errors in {path}")
    return payload


def collector_command(output_dir: Path, *, games: int, workers: int) -> list[str]:
    return [
        sys.executable,
        "scripts/collect_puct_selfplay_parallel.py",
        "--checkpoint", str(PARENT),
        "--output-dir", str(output_dir),
        "--games", str(games),
        "--seed-base", "97310000",
        "--gpus", "4,5",
        "--workers", str(workers),
        "--inference-replicas-per-gpu", "1",
        "--max-inference-batch-size", "64",
        "--max-inference-delay-ms", "4",
        "--inference-timeout-seconds", "300",
        "--initial-simulations", "128",
        "--medium-simulations", "128",
        "--max-simulations", "128",
        "--meaningful-depth", "4",
        "--belief-particles", "32",
        "--max-atomic-steps", "1536",
        "--max-expanded-nodes", "32768",
        "--child-initial", "4",
        "--child-max", "24",
        "--search-temperature", "1.0",
        "--behavior-temperature", "0.5",
        "--force-search",
        "--low-confidence-sample-rate", "0.0",
        "--max-episode-steps", "1200",
        "--shard-size", "128",
        "--staging-shard-size", "32",
        "--heap-trim-every-decisions", "8",
        "--heartbeat-every-decisions", "1",
        "--adaptive-search",
        "--no-adaptive-on-flip",
        "--adaptive-budget-simulations", "512",
        "--adaptive-deep-depth", "6",
        "--adaptive-deep-simulations", "512",
        "--adaptive-min-rank-correlation", "0.70",
        "--adaptive-meaningful-q-gap", "0.01",
        "--adaptive-escalation-q-gap", "0.03",
    ]


def main() -> None:
    args = parse_args()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)

    additional_summaries = [
        HORIZON / "additional-r0/generation_summary.json",
        HORIZON / "additional-r2/generation_summary.json",
        HORIZON / "additional-r1-missing/generation_summary.json",
    ]
    wait_for(additional_summaries, poll_seconds=args.poll_seconds, stage="d6_2048_tail")
    validate_generation(additional_summaries[0], 4)
    validate_generation(additional_summaries[1], 4)
    validate_generation(additional_summaries[2], 2)

    final_horizon = HORIZON / "final-v3"
    deep_replays = [
        HORIZON / "r0/search_replay/manifest.json",
        HORIZON / "r1/search_replay/manifest.json",
        HORIZON / "r2/search_replay/manifest.json",
        HORIZON / "additional-r0/search_replay/manifest.json",
        HORIZON / "additional-r2/search_replay/manifest.json",
        HORIZON / "additional-r1/workers/worker-000/search_replay/manifest.json",
        HORIZON / "additional-r1/workers/worker-003/search_replay/manifest.json",
        HORIZON / "additional-r1-missing/search_replay/manifest.json",
    ]
    command = [
        sys.executable,
        "scripts/adjudicate_horizon_ensemble.py",
        "--position-manifest", str(VARIANCE / "ensemble-analysis-v3/horizon_disagreement_positions.json"),
        "--d6-ensemble-replay", str(VARIANCE / "ensemble-analysis-v3/d6_ensemble_replay/manifest.json"),
    ]
    for replay in deep_replays:
        command.extend(("--deep-replay", str(replay)))
    command.extend(("--output-dir", str(final_horizon)))
    run_once(
        command,
        done_path=final_horizon / "summary.json",
        stage="final_horizon_adjudication",
        log_path=CAMPAIGN / "final_horizon_adjudication.log",
    )
    horizon_summary = read_json(final_horizon / "summary.json")
    if int(horizon_summary.get("teacher_examples", 0)) < 5:
        raise RuntimeError(f"insufficient high-budget horizon convergence: {horizon_summary}")

    primary_summary = PRIMARY / "collection_summary.json"
    wait_for([primary_summary], poll_seconds=args.poll_seconds, stage="primary_complete_game_collection")
    validate_collection(primary_summary, 32)

    if not args.skip_supplemental:
        supplemental_summary = SUPPLEMENTAL / "collection_summary.json"
        supplemental_identity = SUPPLEMENTAL / "collection_identity.json"
        if supplemental_identity.is_file() and not supplemental_summary.is_file():
            emit(
                "stage_attached",
                stage="supplemental_complete_game_collection",
                identity=str(supplemental_identity.relative_to(ROOT)),
            )
            wait_for(
                [supplemental_summary],
                poll_seconds=args.poll_seconds,
                stage="supplemental_complete_game_collection",
            )
        else:
            run_once(
                collector_command(
                    SUPPLEMENTAL,
                    games=int(args.supplemental_games),
                    workers=int(args.supplemental_workers),
                ),
                done_path=supplemental_summary,
                stage="supplemental_complete_game_collection",
                log_path=SUPPLEMENTAL / "run.log",
            )

    collection_summaries = [primary_summary]
    expected_games = [32]
    if not args.skip_supplemental:
        collection_summaries.append(SUPPLEMENTAL / "collection_summary.json")
        expected_games.append(int(args.supplemental_games))
    wait_for(collection_summaries, poll_seconds=args.poll_seconds, stage="complete_game_collection")
    for summary_path, expected in zip(collection_summaries, expected_games, strict=True):
        validate_collection(summary_path, expected)

    collector_replays = [path.parent / "search_replay/manifest.json" for path in collection_summaries]
    collector_anchor_replays = [
        path.parent / "anchor_replay/manifest.json"
        for path in collection_summaries
        if (path.parent / "anchor_replay/manifest.json").is_file()
    ]
    candidate_manifests = [path.parent / "deep_adjudication_candidates.json" for path in collection_summaries]
    wait_for(candidate_manifests, poll_seconds=args.poll_seconds, stage="particle_audit_candidates")
    audit_dir = CAMPAIGN / "particle_audit"
    audit_positions = audit_dir / "positions.json"
    panel_command = [sys.executable, "scripts/build_collected_search_audit_panel.py"]
    for summary_path in collection_summaries:
        panel_command.extend(("--completed-games-dir", str(summary_path.parent / "completed_games")))
    panel_command.extend(
        (
            "--output", str(audit_positions),
            "--per-side-era", "3",
            "--min-q-gap", "0.02",
            "--max-base-latency", "300",
            "--max-per-game", "1",
        )
    )
    run_once(
        panel_command,
        done_path=audit_positions,
        stage="particle_audit_panel",
        log_path=audit_dir / "panel.log",
    )

    audit_replays: list[tuple[str, int, Path]] = []
    for depth, simulations, max_nodes in ((4, 512, 32768), (6, 512, 65536)):
        for replicate in range(3):
            label = f"d{depth}_r{replicate}"
            output_dir = audit_dir / label
            generation_command = [
                sys.executable,
                "scripts/generate_search_replay_parallel.py",
                "--checkpoint", str(PARENT),
                "--position-manifest", str(audit_positions),
                "--output-dir", str(output_dir),
                "--gpus", "4,5",
                "--workers", "18",
                "--roots-per-task", "1",
                "--inference-replicas-per-gpu", "2",
                "--initial-simulations", str(simulations),
                "--medium-simulations", str(simulations),
                "--max-simulations", str(simulations),
                "--meaningful-depth", str(depth),
                "--belief-particles", "32",
                "--max-atomic-steps", "1536",
                "--max-expanded-nodes", str(max_nodes),
                "--child-initial", "4",
                "--child-max", "24",
                "--shard-size", "128",
                "--seed", str(82200000 + depth * 100000 + replicate * 20000),
                "--root-seeded-search",
            ]
            run_once(
                generation_command,
                done_path=output_dir / "generation_summary.json",
                stage=f"particle_audit_{label}",
                log_path=output_dir / "run.log",
            )
            validate_generation(output_dir / "generation_summary.json", 18)
            audit_replays.append((f"d{depth}", replicate, output_dir / "search_replay/manifest.json"))

    audit_analysis = audit_dir / "ensemble_analysis"
    analysis_command = [
        sys.executable,
        "scripts/analyze_search_variance.py",
        "--position-manifest", str(audit_positions),
    ]
    for depth, replicate, replay in audit_replays:
        analysis_command.extend(("--replay", f"{depth}:r{replicate}={replay}"))
    analysis_command.extend(
        (
            "--output-dir", str(audit_analysis),
            "--min-replicates", "3",
            "--min-rank-correlation", "0.70",
            "--meaningful-q-gap", "0.01",
        )
    )
    run_once(
        analysis_command,
        done_path=audit_analysis / "summary.json",
        stage="particle_audit_analysis",
        log_path=audit_dir / "analysis.log",
    )
    audit_summary = read_json(audit_analysis / "summary.json")
    statuses = dict(audit_summary.get("status_counts") or {})
    if int(statuses.get("INSUFFICIENT", 0)) or int(statuses.get("PARTICLE_VARIANCE", 0)) > 6:
        raise RuntimeError(f"particle audit gate failed: {audit_summary}")

    teacher_dir = CAMPAIGN / "converged_teacher"
    teacher_command = [sys.executable, "scripts/build_converged_teacher_replay.py"]
    for replay in collector_replays:
        teacher_command.extend(("--search-replay", str(replay)))
    teacher_command.extend(
        (
            "--exclude-position-manifest",
            str(audit_analysis / "particle_variance_positions.json"),
            "--output-dir", str(teacher_dir),
            "--q-temperature", "0.05",
            "--min-q-gap", "0.02",
            "--min-total-visits", "8",
            "--min-best-action-visits", "2",
            "--load-workers", "32",
        )
    )
    run_once(
        teacher_command,
        done_path=teacher_dir / "summary.json",
        stage="converged_teacher_build",
        log_path=CAMPAIGN / "converged_teacher_build.log",
    )
    teacher_summary = read_json(teacher_dir / "summary.json")
    accepted = int(teacher_summary.get("accepted", 0))
    us = int(teacher_summary.get("side:us", 0))
    ussr = int(teacher_summary.get("side:ussr", 0))
    if accepted < 32 or min(us, ussr) < 12:
        raise RuntimeError(
            f"converged teacher gate failed: accepted={accepted}, us={us}, ussr={ussr}"
        )

    distill_command = [
        sys.executable,
        "scripts/distill_search_replay.py",
        "--restore-from", str(PARENT),
        "--search-replay", str(teacher_dir / "search_replay/manifest.json"),
    ]
    anchor_replays = [BROAD_PARENT_ANCHOR, *collector_anchor_replays]
    wait_for(anchor_replays, poll_seconds=args.poll_seconds, stage="broad_parent_anchor")
    for replay in anchor_replays:
        distill_command.extend(("--anchor-replay", str(replay)))
    distill_command.extend(
        (
            "--output-dir", str(CAMPAIGN),
            "--updates", "25",
            "--checkpoint-every-updates", "5",
            "--no-restore-optimizer-state",
            "--replay-load-workers", "32",
            "--batch-size", "128",
            "--lr", "1e-6",
            "--parameter-scope", "action_head",
            "--lambda-search", "0.5",
            "--search-policy-mix", "0.5",
            "--lambda-pairwise", "0.15",
            "--pairwise-min-q-gap", "0.005",
            "--pairwise-q-scale", "0.03",
            "--pairwise-max-pairs", "8",
            "--lambda-anchor", "2.0",
            "--anchor-replay-batch-size", "256",
            "--anchor-sample-fraction", "0.125",
            "--lambda-value", "0.0",
            "--lambda-search-q", "0.0",
            "--confidence-weight-floor", "0.25",
            "--min-q-advantage", "0.0",
            "--max-atomic-horizon-cutoffs", "0",
            "--max-node-budget-cutoffs", "0",
            "--max-examples-per-source-root", "4",
            "--no-require-flip",
            "--require-complete-game",
            "--q-weight-scale", "0.05",
            "--max-q-weight", "2.0",
            "--max-grad-norm", "1.0",
            "--diversity-max-per-game", "32",
            "--diversity-max-per-turn", "8",
            "--diversity-max-per-card", "8",
            "--diversity-max-per-region", "8",
            "--diversity-max-per-trigger", "64",
            "--num-gpus", "1",
            "--seed", "82125001",
        )
    )
    run_once(
        distill_command,
        done_path=CAMPAIGN / "checkpoints/updates-000025/algorithm_state.pkl",
        stage="frozen_actor_amortization",
        log_path=CAMPAIGN / "distill.log",
    )
    expected_checkpoints = [
        CAMPAIGN / f"checkpoints/updates-{update:06d}/algorithm_state.pkl"
        for update in (5, 10, 15, 20, 25)
    ]
    wait_for(expected_checkpoints, poll_seconds=args.poll_seconds, stage="checkpoint_sweep")
    emit(
        "campaign_ready_for_triple_gate",
        campaign=str(CAMPAIGN.relative_to(ROOT)),
        checkpoints=[str(path.parent.relative_to(ROOT)) for path in expected_checkpoints],
        teacher=teacher_summary,
        horizon=horizon_summary,
        particle_audit=audit_summary,
    )


if __name__ == "__main__":
    main()
