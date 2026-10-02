#!/usr/bin/env python3
"""Measure direct test-time search gains for one promoted candidate checkpoint."""

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
CAMPAIGN = ROOT / "runs/search_distill/search-scaling-phase3u75-amortization-u25-gpu4-20260821"
ANOMALY_OUTPUT = ROOT / "runs/diagnostics/search-scaling-phase3u75-amortization-anomaly-gate-20260821"
OUTPUT = ROOT / "runs/eval/search-scaling-phase3u75-amortization-cross-mode-20260821"
GPUS = os.environ.get("SEARCH_SCALING_GPUS", "4,5")
REQUIRED_SCALING_LABELS = (
    "d2_vs_raw",
    "d4_128_vs_d2_32",
    "d4_512_vs_d4_128",
)
CONFIRMATORY_SEED_BASE = 97_510_000
CONFIRMATORY_SIDE_FLOORS = {
    "d2_vs_raw": 0.40,
    "d4_128_vs_d2_32": 0.40,
    "d4_512_vs_d4_128": 0.25,
}


def cumulative_mode_elo(records: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Express pairwise search-scaling results on one Raw-relative Elo axis."""

    ladder = {"raw": 0.0}
    edges = (
        ("d2_vs_raw", "d2_32", "raw"),
        ("d4_128_vs_d2_32", "d4_128", "d2_32"),
        ("d4_512_vs_d4_128", "d4_512", "d4_128"),
        ("d6_512_vs_d4_512", "d6_512", "d4_512"),
    )
    for label, stronger_mode, weaker_mode in edges:
        record = records.get(label)
        if record is None or weaker_mode not in ladder:
            continue
        ladder[stronger_mode] = ladder[weaker_mode] + float(record["elo_delta"])
    return ladder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--seed-pairs", type=int, default=10)
    parser.add_argument("--deeper-seed-pairs", type=int, default=4)
    parser.add_argument("--confirm-seed-pairs", type=int, default=25)
    parser.add_argument("--confirm-deeper-seed-pairs", type=int, default=8)
    parser.add_argument("--confirm-d6-seed-pairs", type=int, default=4)
    parser.add_argument(
        "--parallel-gates",
        type=int,
        default=2,
        help="Maximum independent search-scaling comparisons to run concurrently.",
    )
    parser.add_argument(
        "--run-d6",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the memory-heavy D6/512 full-game gate as optional evidence.",
    )
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, "time": time.time(), **payload}, sort_keys=True), flush=True)


def external_eval_active(output_dir: Path) -> bool:
    expected = str(output_dir.resolve())
    own_pid = os.getpid()
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_dir.name)
            if pid == own_pid:
                continue
            argv = (proc_dir / "cmdline").read_bytes().split(b"\0")
        except (OSError, ValueError):
            continue
        decoded = [arg.decode("utf-8", errors="replace") for arg in argv if arg]
        if "scripts/eval_puct_agent_pair_parallel.py" not in decoded:
            continue
        for index, argument in enumerate(decoded[:-1]):
            if argument == "--output-dir" and str(Path(decoded[index + 1]).resolve()) == expected:
                return True
    return False


def wait_for_external_eval(output_dir: Path, result_path: Path, poll_seconds: float = 30.0) -> None:
    announced = False
    while not result_path.is_file() and external_eval_active(output_dir):
        if not announced:
            emit("comparison_reusing_active_eval", output_dir=str(output_dir.relative_to(ROOT)))
            announced = True
        time.sleep(max(1.0, poll_seconds))


def wait_for_anomaly(poll_seconds: float) -> dict[str, Any]:
    path = ANOMALY_OUTPUT / "summary.json"
    while not path.is_file():
        emit("waiting_for_anomaly_gate", path=str(path.relative_to(ROOT)))
        time.sleep(max(1.0, poll_seconds))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("passed")):
        raise RuntimeError("cross-mode evaluation blocked by failed anomaly gate")
    return payload


def _resolved_checkpoint(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def validate_result_provenance(
    payload: dict[str, Any],
    *,
    checkpoint: Path,
    candidate_label: str,
    opponent_label: str,
    seed_pairs: int,
    seed_base: int,
) -> None:
    """Reject a completed comparison that belongs to another checkpoint/deal block."""

    config = payload.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("cross-mode result lacks config provenance")
    expected_checkpoint = checkpoint.resolve()
    for key in ("candidate_checkpoint", "opponent_checkpoint"):
        if key not in config or _resolved_checkpoint(config[key]) != expected_checkpoint:
            raise RuntimeError(
                f"cross-mode result {key} mismatch: "
                f"observed={config.get(key)!r} expected={str(expected_checkpoint)!r}"
            )
    scalar_expected = {
        "candidate_label": str(candidate_label),
        "opponent_label": str(opponent_label),
        "seed_pairs": int(seed_pairs),
        "seed_base": int(seed_base),
    }
    for key, value in scalar_expected.items():
        observed = config.get(key)
        if key in {"seed_pairs", "seed_base"}:
            try:
                observed = int(observed)
            except (TypeError, ValueError):
                pass
        if observed != value:
            raise RuntimeError(
                f"cross-mode result {key} mismatch: "
                f"observed={observed!r} expected={value!r}"
            )
    summary = payload.get("summary")
    if not isinstance(summary, dict) or int(summary.get("games", -1)) != 2 * int(seed_pairs):
        raise RuntimeError(
            "cross-mode result is not a complete side-swapped seed block: "
            f"games={None if not isinstance(summary, dict) else summary.get('games')} "
            f"expected={2 * int(seed_pairs)}"
        )


def model_search_args(model: str, depth: int, simulations: int) -> list[str]:
    if depth == 2:
        atomic_steps, expanded_nodes = 1536, 32768
    elif depth == 4 and simulations <= 128:
        atomic_steps, expanded_nodes = 1536, 262144
    elif depth == 4:
        atomic_steps, expanded_nodes = 1536, 1048576
    elif depth == 6:
        atomic_steps, expanded_nodes = 1536, 2097152
    else:
        raise ValueError((model, depth, simulations))
    return [
        f"--{model}-simulations", str(simulations),
        f"--{model}-meaningful-depth", str(depth),
        f"--{model}-max-atomic-steps", str(atomic_steps),
        f"--{model}-max-expanded-nodes", str(expanded_nodes),
    ]


def comparison_args(label: str) -> tuple[list[str], str, str]:
    if label == "d2_vs_raw":
        return [
            "--search", "--candidate-search", "--no-opponent-search",
            *model_search_args("candidate", 2, 32),
        ], "D2/32", "Raw"
    if label == "d4_128_vs_d2_32":
        return [
            "--search", "--candidate-search", "--opponent-search",
            *model_search_args("candidate", 4, 128),
            *model_search_args("opponent", 2, 32),
        ], "D4/128", "D2/32"
    if label == "d4_512_vs_d4_128":
        return [
            "--search", "--candidate-search", "--opponent-search",
            *model_search_args("candidate", 4, 512),
            *model_search_args("opponent", 4, 128),
        ], "D4/512", "D4/128"
    if label == "d6_512_vs_d4_512":
        return [
            "--search", "--candidate-search", "--opponent-search",
            *model_search_args("candidate", 6, 512),
            *model_search_args("opponent", 4, 512),
        ], "D6/512", "D4/512"
    raise ValueError(label)


def run_comparison(
    checkpoint: Path,
    label: str,
    *,
    seed_pairs: int,
    workers: int,
    output_label: str | None = None,
    seed_base: int = 97_500_000,
) -> dict[str, Any]:
    output_dir = OUTPUT / str(output_label or label)
    result_path = output_dir / "results.json"
    search_args, candidate_label, opponent_label = comparison_args(label)
    wait_for_external_eval(output_dir, result_path)
    if not result_path.is_file():
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "scripts/eval_puct_agent_pair_parallel.py",
            "--candidate-checkpoint", str(checkpoint),
            "--opponent-checkpoint", str(checkpoint),
            "--candidate-label", candidate_label,
            "--opponent-label", opponent_label,
            "--output-dir", str(output_dir),
            "--seed-pairs", str(seed_pairs),
            "--seed-base", str(seed_base),
            "--workers", str(workers),
            "--gpus", GPUS,
            "--replicas-per-checkpoint", "2",
            "--max-inference-batch-size", "32",
            "--max-inference-delay-ms", "3",
            "--inference-timeout-seconds", "300",
            "--belief-particles", "32",
            "--max-episode-steps", "1200",
            "--log-games",
            "--resume",
            *search_args,
        ]
        emit("comparison_started", label=label, command=command)
        with (output_dir / "run.log").open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode or not result_path.is_file():
            raise RuntimeError(f"cross-mode comparison failed: {label}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validate_result_provenance(
        payload,
        checkpoint=checkpoint,
        candidate_label=candidate_label,
        opponent_label=opponent_label,
        seed_pairs=seed_pairs,
        seed_base=seed_base,
    )
    summary = payload["summary"]
    if int(summary.get("atomic_horizon_cutoffs", 0)) or int(
        summary.get("node_budget_cutoffs", 0)
    ):
        raise RuntimeError(
            f"cutoff-contaminated cross-mode comparison {label}: "
            f"atomic={summary.get('atomic_horizon_cutoffs')} "
            f"nodes={summary.get('node_budget_cutoffs')}"
        )
    emit(
        "comparison_complete",
        label=label,
        games=summary["games"],
        score=summary["score"],
        us=summary["by_candidate_side"]["us"]["score"],
        ussr=summary["by_candidate_side"]["ussr"]["score"],
        elo_delta=summary["elo_delta"],
    )
    return summary


def run_comparison_jobs(
    jobs: list[tuple[str, dict[str, Any]]],
    *,
    max_parallel: int,
) -> dict[str, dict[str, Any]]:
    """Run independent, resumable cross-mode comparisons concurrently."""

    if not jobs:
        return {}
    parallelism = max(1, min(int(max_parallel), len(jobs)))
    if parallelism == 1:
        return {label: run_comparison(**kwargs) for label, kwargs in jobs}

    emit(
        "comparison_batch_started",
        jobs=[label for label, _ in jobs],
        parallelism=parallelism,
    )
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {
            executor.submit(run_comparison, **kwargs): label
            for label, kwargs in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            results[label] = future.result()
    emit(
        "comparison_batch_complete",
        jobs=sorted(results),
        parallelism=parallelism,
    )
    return results


def confirmatory_comparison_jobs(
    checkpoint: Path,
    plan: list[tuple[str, int]],
) -> list[tuple[str, dict[str, Any]]]:
    """Use one matched deal block across every search-scaling edge.

    Expensive modes may consume a shorter prefix, but their seeds and deals are
    still directly shared with the cheaper comparisons.
    """

    return [
        (
            label,
            {
                "checkpoint": checkpoint,
                "label": label,
                "seed_pairs": seed_pairs,
                "workers": min(20, 2 * seed_pairs),
                "output_label": f"confirmatory/{label}",
                "seed_base": CONFIRMATORY_SEED_BASE,
            },
        )
        for label, seed_pairs in plan
    ]


def main() -> None:
    args = parse_args()
    anomaly = wait_for_anomaly(args.poll_seconds)
    checkpoint = Path(anomaly["checkpoint"])
    if checkpoint.parent.parent.resolve() != CAMPAIGN.resolve():
        raise RuntimeError(f"unexpected candidate checkpoint: {checkpoint}")
    records = run_comparison_jobs(
        [
            (
                label,
                {
                    "checkpoint": checkpoint,
                    "label": label,
                    "seed_pairs": int(
                        args.deeper_seed_pairs
                        if label == "d4_512_vs_d4_128"
                        else args.seed_pairs
                    ),
                    "workers": 2
                    * int(
                        args.deeper_seed_pairs
                        if label == "d4_512_vs_d4_128"
                        else args.seed_pairs
                    ),
                },
            )
            for label in REQUIRED_SCALING_LABELS
        ],
        max_parallel=int(args.parallel_gates),
    )
    d4_512_record = records["d4_512_vs_d4_128"]
    d4_512_side_floor = min(
        float(d4_512_record["by_candidate_side"][side]["score"])
        for side in ("us", "ussr")
    )
    skipped_records: dict[str, dict[str, Any]] = {}
    if (
        bool(args.run_d6)
        and float(d4_512_record["score"]) >= 0.45
        and d4_512_side_floor >= 0.25
    ):
        records["d6_512_vs_d4_512"] = run_comparison(
            checkpoint,
            "d6_512_vs_d4_512",
            seed_pairs=int(args.deeper_seed_pairs),
            workers=2 * int(args.deeper_seed_pairs),
        )
    else:
        partial_path = OUTPUT / "d6_512_vs_d4_512/progress.json"
        skip_reason = (
            "routine_full_game_d6_disabled"
            if not bool(args.run_d6)
            else "d4_512_vs_d4_128_failed_preliminary_gate"
        )
        skipped_records["d6_512_vs_d4_512"] = {
            "reason": skip_reason,
            "required_score": 0.45,
            "observed_d4_512_score": float(d4_512_record["score"]),
            "observed_d4_512_side_floor": d4_512_side_floor,
            "partial_progress": (
                json.loads(partial_path.read_text(encoding="utf-8"))
                if partial_path.is_file()
                else None
            ),
        }
        emit(
            "comparison_skipped",
            label="d6_512_vs_d4_512",
            reason=skip_reason,
            d4_512_score=float(d4_512_record["score"]),
            d4_512_side_floor=d4_512_side_floor,
        )
    scaling_scores = {label: float(record["score"]) for label, record in records.items()}
    required_records = {label: records[label] for label in REQUIRED_SCALING_LABELS}
    side_floor = min(
        float(record["by_candidate_side"][side]["score"])
        for record in required_records.values()
        for side in ("us", "ussr")
    )
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "records": records,
        "marginal_search_scores": scaling_scores,
        "side_floor": side_floor,
        "skipped_records": skipped_records,
        "required_scaling_labels": list(REQUIRED_SCALING_LABELS),
        "non_decreasing_with_noise_tolerance": all(
            float(required_records[label]["score"]) >= 0.45
            for label in REQUIRED_SCALING_LABELS
        ),
        "strict_positive_point_estimates": all(
            float(required_records[label]["score"]) > 0.50
            for label in REQUIRED_SCALING_LABELS
        ),
        "mode_elo_relative_raw": cumulative_mode_elo(records),
    }
    confirmatory_records: dict[str, dict[str, Any]] = {}
    if summary["non_decreasing_with_noise_tolerance"] and side_floor >= 0.25:
        confirmatory_plan = [
            ("d2_vs_raw", int(args.confirm_seed_pairs)),
            ("d4_128_vs_d2_32", int(args.confirm_seed_pairs)),
            ("d4_512_vs_d4_128", int(args.confirm_deeper_seed_pairs)),
        ]
        if bool(args.run_d6) and "d6_512_vs_d4_512" in records:
            confirmatory_plan.append(
                ("d6_512_vs_d4_512", int(args.confirm_d6_seed_pairs))
            )
        confirmatory_records.update(
            run_comparison_jobs(
                confirmatory_comparison_jobs(checkpoint, confirmatory_plan),
                max_parallel=int(args.parallel_gates),
            )
        )
    summary["confirmatory_records"] = confirmatory_records
    summary["confirmatory_mode_elo_relative_raw"] = cumulative_mode_elo(
        confirmatory_records
    )
    summary["confirmatory_side_floors"] = {
        label: min(
            float(confirmatory_records[label]["by_candidate_side"][side]["score"])
            for side in ("us", "ussr")
        )
        for label in REQUIRED_SCALING_LABELS
        if label in confirmatory_records
    }
    summary["confirmatory_side_balance_supported"] = all(
        label in summary["confirmatory_side_floors"]
        and float(summary["confirmatory_side_floors"][label])
        >= float(CONFIRMATORY_SIDE_FLOORS[label])
        for label in REQUIRED_SCALING_LABELS
    )
    summary["confirmatory_scaling_supported"] = all(
        label in confirmatory_records
        and float(confirmatory_records[label]["paired_bootstrap_ci95_low"]) >= 0.45
        and float(confirmatory_records[label]["score"]) >= 0.50
        for label in REQUIRED_SCALING_LABELS
    ) and bool(summary["confirmatory_side_balance_supported"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    emit("cross_mode_gates_complete", **{k: summary[k] for k in ("side_floor", "non_decreasing_with_noise_tolerance", "strict_positive_point_estimates")})


if __name__ == "__main__":
    main()
