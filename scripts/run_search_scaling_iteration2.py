#!/usr/bin/env python3
"""Run the second Phase-3-u75 planning/amortization iteration.

The expensive stages are idempotent: completed collection, teacher, and
checkpoint artifacts are reused after interruption. Search-generated behavior
is never treated as PPO on-policy data.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075"
COLLECTION = ROOT / "runs/puct_selfplay/search-scaling-iteration2-phase3u75-adaptive-d4d6-32games-gpu45-20260824"
CAMPAIGN = ROOT / "runs/search_distill/search-scaling-iteration2-phase3u75-u25-gpu4-20260824"
BROAD_PARENT_ANCHOR = ROOT / "runs/puct_selfplay/rootfix-phase4-d4-128-16games-w16-gpu45-20260819/search_replay/manifest.json"
PRESSURE_D8_STATUS = ROOT / "runs/search_scaling/iteration2-scoring-pressure-probe-20260824/pressure-step66-d8_2048-post-collection.status"


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, "time": time.time(), **payload}, sort_keys=True), flush=True)


def run_once(command: list[str], *, done_path: Path, log_path: Path, stage: str) -> None:
    if done_path.is_file():
        emit("stage_reused", stage=stage, done_path=str(done_path.relative_to(ROOT)))
        return
    done_path.parent.mkdir(parents=True, exist_ok=True)
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
    if completed.returncode or not done_path.is_file():
        raise RuntimeError(
            f"{stage} failed: returncode={completed.returncode}, expected={done_path}"
        )
    emit("stage_complete", stage=stage, done_path=str(done_path.relative_to(ROOT)))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for_file(path: Path, *, poll_seconds: float, stage: str) -> None:
    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        display_path = str(path)
    announced = False
    while not path.is_file():
        if not announced:
            emit("stage_waiting", stage=stage, path=display_path)
            announced = True
        time.sleep(max(0.01, float(poll_seconds)))
    emit("stage_dependency_ready", stage=stage, path=display_path)


def validate_phase1() -> None:
    base = ROOT / "runs/search_scaling/phase3u75-convergence-rootselectfix-v2-20260824"
    for label in ("d2_32", "d4_128", "d4_512"):
        summary = read_json(base / label / "generation_summary.json")
        if int(summary.get("complete_teacher_roots", 0)) < (75 if label == "d4_512" else 96):
            raise RuntimeError(f"incomplete Phase-1 evidence for {label}: {summary}")
        if not bool(summary.get("partial_trees_included", False)) and label == "d4_512":
            continue
        if label == "d4_512":
            raise RuntimeError("D4/512 partial trees must be excluded fail-closed")


def collector_command() -> list[str]:
    return [
        sys.executable,
        "scripts/collect_puct_selfplay_parallel.py",
        "--checkpoint", str(PARENT),
        "--output-dir", str(COLLECTION),
        "--games", "32",
        "--seed-base", "97600000",
        "--gpus", "4,5",
        "--workers", "16",
        "--inference-replicas-per-gpu", "2",
        "--max-inference-batch-size", "64",
        "--max-inference-delay-ms", "4",
        "--inference-timeout-seconds", "300",
        "--initial-simulations", "128",
        "--medium-simulations", "128",
        "--max-simulations", "128",
        "--meaningful-depth", "4",
        "--belief-particles", "32",
        "--max-atomic-steps", "1536",
        "--max-expanded-nodes", "131072",
        "--max-search-worker-rss-mib", "20480",
        "--max-search-seconds", "300",
        "--child-initial", "4",
        "--child-max", "24",
        "--search-temperature", "1.0",
        "--behavior-temperature", "0.35",
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
        "--adaptive-escalation-q-gap", "0.01",
    ]


def main() -> None:
    validate_phase1()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)

    collection_summary = COLLECTION / "collection_summary.json"
    run_once(
        collector_command(),
        done_path=collection_summary,
        log_path=COLLECTION / "run.log",
        stage="adaptive_complete_game_collection",
    )
    collection = read_json(collection_summary)
    if int(collection.get("games_completed", 0)) != 32:
        raise RuntimeError(f"collection is not 32 complete games: {collection}")
    if int(collection.get("discarded_games", 0)):
        raise RuntimeError(f"collection discarded games: {collection}")

    teacher_dir = CAMPAIGN / "converged_teacher"
    run_once(
        [
            sys.executable,
            "scripts/build_converged_teacher_replay.py",
            "--search-replay", str(COLLECTION / "search_replay/manifest.json"),
            "--output-dir", str(teacher_dir),
            "--q-temperature", "0.05",
            "--min-q-gap", "0.01",
            "--min-total-visits", "8",
            "--min-best-action-visits", "2",
            "--load-workers", "32",
            "--shard-size", "128",
        ],
        done_path=teacher_dir / "summary.json",
        log_path=CAMPAIGN / "teacher.log",
        stage="converged_teacher",
    )
    teacher = read_json(teacher_dir / "summary.json")
    accepted = int(teacher.get("accepted", 0))
    if accepted < 512 or min(int(teacher.get("side:us", 0)), int(teacher.get("side:ussr", 0))) < 192:
        raise RuntimeError(f"teacher coverage/balance gate failed: {teacher}")

    checkpoints = CAMPAIGN / "checkpoints"
    run_once(
        [
            "env",
            "CUDA_VISIBLE_DEVICES=0",
            sys.executable,
            "scripts/distill_search_replay.py",
            "--restore-from", str(PARENT),
            "--search-replay", str(teacher_dir / "search_replay/manifest.json"),
            "--anchor-replay", str(COLLECTION / "anchor_replay/manifest.json"),
            "--anchor-replay", str(BROAD_PARENT_ANCHOR),
            "--output-dir", str(CAMPAIGN),
            "--updates", "25",
            "--checkpoint-every-updates", "5",
            "--no-restore-optimizer-state",
            "--replay-load-workers", "32",
            "--batch-size", "256",
            "--lr", "2.5e-7",
            "--parameter-scope", "action_head",
            "--lambda-search", "0.25",
            "--search-policy-mix", "0.35",
            "--lambda-pairwise", "0.05",
            "--pairwise-min-q-gap", "0.01",
            "--pairwise-q-scale", "0.05",
            "--pairwise-max-pairs", "6",
            "--lambda-anchor", "4.0",
            "--anchor-replay-batch-size", "512",
            "--anchor-sample-fraction", "0.25",
            "--lambda-value", "0.0",
            "--lambda-search-q", "0.0",
            "--confidence-weight-floor", "0.15",
            "--max-atomic-horizon-cutoffs", "0",
            "--max-node-budget-cutoffs", "0",
            "--max-examples-per-source-root", "4",
            "--no-require-flip",
            "--require-complete-game",
            "--q-weight-scale", "0.10",
            "--max-q-weight", "1.25",
            "--max-grad-norm", "0.5",
            "--diversity-max-per-game", "24",
            "--diversity-max-per-turn", "6",
            "--diversity-max-per-card", "6",
            "--diversity-max-per-region", "6",
            "--diversity-max-per-trigger", "48",
            "--num-gpus", "1",
            "--seed", "82425001",
        ],
        done_path=checkpoints / "updates-000025/algorithm_state.pkl",
        log_path=CAMPAIGN / "distill.log",
        stage="frozen_actor_amortization",
    )
    expected = [checkpoints / f"updates-{update:06d}/algorithm_state.pkl" for update in (5, 10, 15, 20, 25)]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing early actor checkpoints: {missing}")
    emit(
        "iteration_ready_for_promotion_gates",
        parent=str(PARENT),
        collection=str(collection_summary),
        teacher=teacher,
        checkpoints=[str(path.parent) for path in expected],
    )

    wait_for_file(
        PRESSURE_D8_STATUS,
        poll_seconds=30.0,
        stage="post_collection_pressure_d8",
    )
    pressure_status = PRESSURE_D8_STATUS.read_text(encoding="utf-8").strip()
    if pressure_status != "complete":
        raise RuntimeError(f"post-collection pressure D8 failed: {pressure_status}")

    gate_summary = ROOT / "runs/eval/search-scaling-iteration2-phase3u75-gates-20260824/gate_summary.json"
    run_once(
        [
            "env",
            "SEARCH_SCALING_GPUS=0,1",
            sys.executable,
            "scripts/run_search_scaling_iteration2_promotion_gates.py",
            "--poll-seconds", "30",
            "--seed-pairs", "10",
            "--deeper-seed-pairs", "4",
            "--confirm-seed-pairs", "25",
            "--workers", "20",
        ],
        done_path=gate_summary,
        log_path=CAMPAIGN / "promotion_gates.log",
        stage="raw_d2_d4_parent_gates",
    )

    anomaly_summary = ROOT / "runs/diagnostics/search-scaling-iteration2-phase3u75-anomaly-gate-20260824/summary.json"
    run_once(
        [
            "env",
            "SEARCH_SCALING_ANOMALY_DEVICES=cuda:0,cuda:1",
            sys.executable,
            "scripts/run_search_scaling_iteration2_anomaly_gate.py",
            "--poll-seconds", "30",
        ],
        done_path=anomaly_summary,
        log_path=CAMPAIGN / "anomaly_gate.log",
        stage="frozen_anomaly_gate",
    )

    cross_mode_summary = ROOT / "runs/eval/search-scaling-iteration2-phase3u75-cross-mode-20260824/summary.json"
    run_once(
        [
            "env",
            "SEARCH_SCALING_GPUS=0,1",
            sys.executable,
            "scripts/run_search_scaling_iteration2_cross_mode_gates.py",
            "--poll-seconds", "30",
            "--seed-pairs", "10",
            "--deeper-seed-pairs", "4",
            "--confirm-seed-pairs", "25",
            "--confirm-deeper-seed-pairs", "8",
            "--confirm-d6-seed-pairs", "4",
        ],
        done_path=cross_mode_summary,
        log_path=CAMPAIGN / "cross_mode_gates.log",
        stage="same_checkpoint_search_scaling_gates",
    )
    promotion_decision = CAMPAIGN / "promotion_decision.json"
    run_once(
        [
            sys.executable,
            "scripts/finalize_search_scaling_promotion.py",
            "--gate-summary", str(gate_summary),
            "--anomaly-summary", str(anomaly_summary),
            "--cross-mode-summary", str(cross_mode_summary),
            "--parent", str(PARENT),
            "--output", str(promotion_decision),
        ],
        done_path=promotion_decision,
        log_path=CAMPAIGN / "promotion_decision.log",
        stage="fail_closed_general_parent_decision",
    )
    final = {
        "parent": str(PARENT.resolve()),
        "collection": str(collection_summary.resolve()),
        "teacher": str((teacher_dir / "summary.json").resolve()),
        "promotion_gates": str(gate_summary.resolve()),
        "anomaly_gate": str(anomaly_summary.resolve()),
        "cross_mode_gates": str(cross_mode_summary.resolve()),
        "promotion_decision": str(promotion_decision.resolve()),
    }
    (CAMPAIGN / "iteration_summary.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    emit("iteration_complete", **final)


if __name__ == "__main__":
    main()
