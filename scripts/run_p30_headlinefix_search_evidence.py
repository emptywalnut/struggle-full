#!/usr/bin/env python3
"""Rebuild search-scaling evidence under the corrected headline semantics.

Every expensive stage is durable and reused only after its output validates.  The
driver is intended to run after the four corrected complete-game collections have
finished and before actor amortization starts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PARENT = (
    ROOT
    / "runs/search_distill/search-scaling-phase3u75-parentactor-p30u25-composed-20260828/"
    "checkpoints/composed"
)
COLLECTIONS = tuple(
    ROOT / path
    for path in (
        "runs/puct_selfplay/search-scaling-p30u25-adaptive-d4d6-headlinefix-32games-gpu45-20260828",
        "runs/puct_selfplay/search-scaling-p30u25-adaptive-d4d6-headlinefix-32games-gpu45b-20260828",
        "runs/puct_selfplay/search-scaling-p30u25-adaptive-d4d6-headlinefix-32games-gpu45c-20260828",
        "runs/puct_selfplay/search-scaling-p30u25-adaptive-d4d6-headlinefix-32games-gpu45d-20260828",
    )
)
OUTPUT = ROOT / "runs/search_scaling/phase3u75-p30u25-headlinefix-convergence-20260829"
OPERATOR = ROOT / "runs/eval/search-scaling-phase3u75-p30u25-headlinefix-d4-vs-d2-20260829"
BRIDGE_AUDIT = (
    ROOT
    / "runs/search_distill/search-scaling-p30u25-amortization-u25-gpu4-20260828/"
    "bridge_replay_audit.json"
)
SEMANTICS_VERSION = "headline_priority_fixed_v1"
ROOT_SEARCH_SEED = 98_810_001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--roots", type=int, default=24)
    parser.add_argument(
        "--root-workers",
        type=int,
        default=20,
        help="Concurrent fixed-panel root probes per search configuration/GPU.",
    )
    parser.add_argument("--operator-seed-pairs", type=int, default=50)
    parser.add_argument("--operator-workers", type=int, default=32)
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, "time": time.time(), **payload}, sort_keys=True), flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_once(command: list[str], *, done_path: Path, log_path: Path, stage: str) -> None:
    if done_path.is_file():
        emit("stage_reused", stage=stage, done_path=str(done_path.relative_to(ROOT)))
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    emit("stage_started", stage=stage, command=command)
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode or not done_path.is_file():
        raise RuntimeError(
            f"{stage} failed: returncode={completed.returncode}, expected={done_path}"
        )
    emit("stage_complete", stage=stage, done_path=str(done_path.relative_to(ROOT)))


def validate_inputs() -> dict[str, Any]:
    if not BRIDGE_AUDIT.is_file():
        raise RuntimeError(f"missing current-bridge replay audit: {BRIDGE_AUDIT}")
    audit = read_json(BRIDGE_AUDIT)
    if (
        not bool(audit.get("passed"))
        or int(audit.get("games", -1)) != 128
        or int(audit.get("failed_games", -1)) != 0
    ):
        raise RuntimeError(f"current-bridge replay audit is not a 128-game pass: {audit}")
    for collection in COLLECTIONS:
        summary_path = collection / "collection_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"missing corrected collection summary: {summary_path}")
        summary = read_json(summary_path)
        if int(summary.get("games_completed", -1)) != 32:
            raise RuntimeError(f"incomplete corrected collection: {summary_path}")
        if int(summary.get("worker_errors", summary.get("errors", 0))) != 0:
            raise RuntimeError(f"collector errors in {summary_path}")
    return audit


def generation_command(
    label: str,
    *,
    manifest: Path,
    gpu: str,
    workers: int,
    depth: int,
    simulations: int,
    max_atomic_steps: int,
    max_nodes: int,
    max_search_seconds: float,
    max_worker_rss_mib: float,
) -> list[str]:
    output = OUTPUT / label
    return [
        sys.executable,
        "scripts/generate_search_replay_parallel.py",
        "--checkpoint", str(PARENT),
        "--position-manifest", str(manifest),
        "--output-dir", str(output),
        "--gpus", gpu,
        "--workers", str(workers),
        "--roots-per-task", "1",
        "--inference-replicas-per-gpu", "1",
        "--max-inference-batch-size", "64",
        "--max-inference-delay-ms", "4",
        "--inference-timeout-seconds", "300",
        "--initial-simulations", str(simulations),
        "--medium-simulations", str(simulations),
        "--max-simulations", str(simulations),
        "--meaningful-depth", str(depth),
        "--belief-particles", "32",
        "--max-atomic-steps", str(max_atomic_steps),
        "--max-expanded-nodes", str(max_nodes),
        "--max-search-seconds", str(max_search_seconds),
        "--max-search-worker-rss-mib", str(max_worker_rss_mib),
        "--child-initial", "4",
        "--child-max", "24",
        "--search-temperature", "1.0",
        "--shard-size", "64",
        "--seed", str(ROOT_SEARCH_SEED),
        "--root-seeded-search",
    ]


def run_generation(
    label: str,
    *,
    manifest: Path,
    gpu: str,
    workers: int,
    depth: int,
    simulations: int,
    max_atomic_steps: int,
    max_nodes: int,
    max_search_seconds: float,
    max_worker_rss_mib: float,
) -> None:
    output = OUTPUT / label
    run_once(
        generation_command(
            label,
            manifest=manifest,
            gpu=gpu,
            workers=workers,
            depth=depth,
            simulations=simulations,
            max_atomic_steps=max_atomic_steps,
            max_nodes=max_nodes,
            max_search_seconds=max_search_seconds,
            max_worker_rss_mib=max_worker_rss_mib,
        ),
        done_path=output / "generation_summary.json",
        log_path=output / "run.log",
        stage=label,
    )
    summary = read_json(output / "generation_summary.json")
    expected = manifest_count(manifest)
    config = dict(summary.get("config") or {})
    mismatches = {
        "positions_requested": (expected, int(summary.get("positions_requested", -1))),
        "roots_completed": (expected, int(summary.get("roots_completed", -1))),
        "meaningful_depth": (depth, int(config.get("meaningful_depth", -1))),
        "initial_simulations": (simulations, int(config.get("initial_simulations", -1))),
        "belief_particles": (32, int(config.get("belief_particles", -1))),
        "max_atomic_steps": (max_atomic_steps, int(config.get("max_atomic_steps", -1))),
        "max_expanded_nodes": (max_nodes, int(config.get("max_expanded_nodes", -1))),
        "max_search_seconds": (
            float(max_search_seconds),
            float(config.get("max_search_seconds", -1.0)),
        ),
        "max_search_worker_rss_mib": (
            float(max_worker_rss_mib),
            float(config.get("max_search_worker_rss_mib", -1.0)),
        ),
    }
    mismatches = {key: value for key, value in mismatches.items() if value[0] != value[1]}
    if mismatches:
        raise RuntimeError(f"invalid/reused targeted-search generation {label}: {mismatches}")


def replay(label: str) -> str:
    return f"{label}={OUTPUT / label / 'search_replay/manifest.json'}"


def classify(name: str, labels: list[str], *, max_d6: int, max_d8: int) -> Path:
    output = OUTPUT / name
    command = [
        sys.executable,
        "scripts/classify_search_convergence.py",
        "--position-manifest", str(OUTPUT / "positions.json"),
        "--output-dir", str(output),
        "--min-rank-correlation", "0.70",
        "--meaningful-q-gap", "0.01",
        "--q-temperature", "0.05",
        "--max-d6-adjudications", str(max_d6),
        "--max-d8-adjudications", str(max_d8),
    ]
    for label in labels:
        command.extend(("--replay", replay(label)))
    run_once(
        command,
        done_path=output / "summary.json",
        log_path=output / "run.log",
        stage=name,
    )
    return output


def manifest_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return len(read_json(path).get("positions") or [])


def validate_root_panel(panel: dict[str, Any], expected_roots: int) -> dict[str, Any]:
    """Require the fixed panel to exercise both perspectives and broad game phases."""

    summary = dict(panel.get("summary") or {})
    positions = list(panel.get("positions") or [])
    by_side = dict(summary.get("by_side") or {})
    by_era = dict(summary.get("by_era") or {})
    by_side_era = dict(summary.get("by_side_era") or {})
    by_branch = dict(summary.get("by_branch_bucket") or {})
    by_category = dict(summary.get("by_category") or {})
    reasons: list[str] = []
    if len(positions) != int(expected_roots):
        reasons.append("incomplete_root_count")
    if any(
        int(by_side.get(side, 0)) != int(expected_roots) // 2
        for side in ("us", "ussr")
    ):
        reasons.append("side_imbalance")
    minimum_era_roots = int(expected_roots) // 3
    if any(
        int(by_era.get(era, 0)) < minimum_era_roots
        for era in ("early", "mid", "late")
    ):
        reasons.append("era_imbalance")
    minimum_side_era_roots = int(expected_roots) // 6
    if any(
        int(by_side_era.get(f"{side}|{era}", 0)) < minimum_side_era_roots
        for side in ("us", "ussr")
        for era in ("early", "mid", "late")
    ):
        reasons.append("side_era_imbalance")
    if int(by_branch.get("low", 0)) < 1 or int(by_branch.get("high", 0)) < 1:
        reasons.append("branching_coverage")
    tactical_categories = {
        "influence_allocation",
        "coup_target",
        "realignment_target",
    }
    if any(int(by_category.get(category, 0)) < 1 for category in tactical_categories):
        reasons.append("tactical_category_coverage")
    card_categories = {
        "scoring_card",
        "scoring_pressure",
        "opponent_event",
        "forced_response",
        "chance_heavy",
    }
    if sum(
        int(by_category.get(category, 0))
        for category in ("scoring_card", "scoring_pressure")
    ) < 1:
        reasons.append("scoring_pressure_coverage")
    if sum(int(by_category.get(category, 0)) for category in card_categories) < 2:
        reasons.append("card_timing_coverage")
    if len([category for category, count in by_category.items() if int(count) > 0]) < 5:
        reasons.append("category_diversity")
    if reasons:
        raise RuntimeError(
            "fixed convergence root panel lacks required diversity: "
            + json.dumps({"reasons": reasons, "summary": summary}, sort_keys=True)
        )
    return {
        "passed": True,
        "roots": len(positions),
        "by_side": by_side,
        "by_era": by_era,
        "by_side_era": by_side_era,
        "by_branch_bucket": by_branch,
        "by_category": by_category,
    }


def write_final_summary(classification: Path, bridge_audit: dict[str, Any]) -> dict[str, Any]:
    source = read_json(classification / "summary.json")
    payload = {
        **source,
        "bridge_semantics": SEMANTICS_VERSION,
        "bridge_replay_audit": str(BRIDGE_AUDIT.resolve()),
        "bridge_replay_games": int(bridge_audit["games"]),
        "source_collections": [str(path.resolve()) for path in COLLECTIONS],
        "position_manifest": str((OUTPUT / "positions.json").resolve()),
        "classification": str(classification.resolve()),
    }
    target = OUTPUT / "final_summary.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_operator(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/eval_puct_agent_pair_parallel.py",
        "--candidate-checkpoint", str(PARENT),
        "--opponent-checkpoint", str(PARENT),
        "--candidate-label", "phase3u75_composed_d4_128_headlinefix",
        "--opponent-label", "phase3u75_composed_d2_32_headlinefix",
        "--output-dir", str(OPERATOR),
        "--seed-pairs", str(args.operator_seed_pairs),
        "--seed-base", "98830000",
        "--workers", str(args.operator_workers),
        "--gpus", str(args.gpus),
        "--replicas-per-checkpoint", "2",
        "--max-inference-batch-size", "64",
        "--max-inference-delay-ms", "4",
        "--inference-timeout-seconds", "300",
        "--candidate-search",
        "--opponent-search",
        "--candidate-simulations", "128",
        "--candidate-meaningful-depth", "4",
        "--candidate-max-atomic-steps", "4096",
        "--candidate-max-expanded-nodes", "262144",
        "--opponent-simulations", "32",
        "--opponent-meaningful-depth", "2",
        "--opponent-max-atomic-steps", "1536",
        "--opponent-max-expanded-nodes", "32768",
        "--belief-particles", "32",
        "--max-episode-steps", "1200",
        "--bootstrap-samples", "20000",
        "--resume",
    ]
    run_once(
        command,
        done_path=OPERATOR / "results.json",
        log_path=OPERATOR / "run.log",
        stage="headlinefix_d4_vs_d2_operator_gate",
    )
    result = read_json(OPERATOR / "results.json")
    if int(result.get("summary", {}).get("games", -1)) != int(args.operator_seed_pairs) * 2:
        raise RuntimeError(f"operator gate did not finish every paired game: {result.get('summary')}")
    payload = {
        **result,
        "bridge_semantics": SEMANTICS_VERSION,
        "bridge_replay_audit": str(BRIDGE_AUDIT.resolve()),
    }
    target = OPERATOR / "evidence.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bridge_audit = validate_inputs()
    candidate = OUTPUT / "candidate_positions.json"
    command = [sys.executable, "scripts/build_multidepth_position_manifest.py"]
    for collection in COLLECTIONS:
        command.extend(("--d4-replay", str(collection / "search_replay/manifest.json")))
        command.extend(("--completed-games", str(collection / "completed_games")))
    command.extend(
        (
            "--output", str(candidate),
            "--max-roots", "512",
            "--max-per-game", "8",
            "--load-workers", "32",
        )
    )
    run_once(
        command,
        done_path=candidate,
        log_path=OUTPUT / "candidate_positions.log",
        stage="headlinefix_candidate_root_panel",
    )
    run_once(
        [
            sys.executable,
            "scripts/build_search_scaling_manifest.py",
            "--source-manifest", str(candidate),
            "--output", str(OUTPUT / "positions.json"),
            "--roots", str(args.roots),
            "--max-per-game", "1",
        ],
        done_path=OUTPUT / "positions.json",
        log_path=OUTPUT / "positions.log",
        stage="headlinefix_stratified_root_panel",
    )
    panel = read_json(OUTPUT / "positions.json")
    panel_audit = validate_root_panel(panel, int(args.roots))

    gpus = [part.strip() for part in str(args.gpus).split(",") if part.strip()]
    if len(gpus) < 2:
        raise RuntimeError("this evidence campaign requires two GPU IDs")
    configurations = (
        ("d2_32", gpus[0], 2, 32, 1536, 32768, 300.0, 12288.0),
        ("d2_128", gpus[1], 2, 128, 1536, 131072, 300.0, 12288.0),
        ("d4_128", gpus[2] if len(gpus) >= 4 else gpus[0], 4, 128, 4096, 262144, 300.0, 16384.0),
        ("d4_512", gpus[3] if len(gpus) >= 4 else gpus[1], 4, 512, 8192, 1048576, 600.0, 20480.0),
    )
    # Four A100s run the complete panel concurrently; two-A100 fallback uses two waves.
    groups = (configurations,) if len(gpus) >= 4 else (configurations[:2], configurations[2:])
    for group in groups:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as executor:
            futures = [
                executor.submit(
                    run_generation,
                    label,
                    manifest=OUTPUT / "positions.json",
                    gpu=gpu,
                    workers=max(1, min(int(args.root_workers), int(args.roots))),
                    depth=depth,
                    simulations=simulations,
                    max_atomic_steps=atomic,
                    max_nodes=nodes,
                    max_search_seconds=seconds,
                    max_worker_rss_mib=rss,
                )
                for label, gpu, depth, simulations, atomic, nodes, seconds, rss in group
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    labels = ["d2_32", "d2_128", "d4_128", "d4_512"]
    stage = classify("classification-through-d4-512", labels, max_d6=24, max_d8=0)
    d6_512_manifest = stage / "d6_2048_candidates.json"
    if manifest_count(d6_512_manifest):
        run_generation(
            "d6_512",
            manifest=d6_512_manifest,
            gpu=str(args.gpus),
            workers=16,
            depth=6,
            simulations=512,
            max_atomic_steps=8192,
            max_nodes=1048576,
            max_search_seconds=600.0,
            max_worker_rss_mib=20480.0,
        )
        labels.append("d6_512")
    stage = classify("classification-through-d6-512", labels, max_d6=8, max_d8=0)
    d6_2048_manifest = stage / "d6_2048_candidates.json"
    if manifest_count(d6_2048_manifest):
        run_generation(
            "d6_2048",
            manifest=d6_2048_manifest,
            gpu=str(args.gpus),
            workers=8,
            depth=6,
            simulations=2048,
            max_atomic_steps=16384,
            max_nodes=4194304,
            max_search_seconds=1200.0,
            max_worker_rss_mib=24576.0,
        )
        labels.append("d6_2048")
    stage = classify("classification-through-d6-2048", labels, max_d6=0, max_d8=2)
    d8_manifest = stage / "d8_2048_candidates.json"
    if manifest_count(d8_manifest):
        run_generation(
            "d8_2048",
            manifest=d8_manifest,
            gpu=str(args.gpus),
            workers=2,
            depth=8,
            simulations=2048,
            max_atomic_steps=32768,
            max_nodes=8388608,
            max_search_seconds=1200.0,
            max_worker_rss_mib=24576.0,
        )
        labels.append("d8_2048")
    final_stage = classify("classification-final", labels, max_d6=0, max_d8=0)
    convergence = write_final_summary(final_stage, bridge_audit)
    if int(convergence.get("positions", -1)) != int(args.roots):
        raise RuntimeError(f"final convergence evidence lost roots: {convergence}")
    operator = run_operator(args)
    summary = {
        "bridge_semantics": SEMANTICS_VERSION,
        "convergence": convergence,
        "operator": operator["summary"],
        "parent": str(PARENT.resolve()),
        "root_panel_audit": panel_audit,
    }
    (OUTPUT / "evidence_campaign_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    emit("headlinefix_search_evidence_complete", summary=summary)


if __name__ == "__main__":
    main()
