#!/usr/bin/env python3
"""Run resumable Raw/D2/D4/deeper gates for search-amortization checkpoints."""

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
PARENT = ROOT / "runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075"
CAMPAIGN = ROOT / "runs/search_distill/search-scaling-phase3u75-amortization-u25-gpu4-20260821"
OUTPUT = ROOT / "runs/eval/search-scaling-phase3u75-amortization-gates-20260821"
GPUS = os.environ.get("SEARCH_SCALING_GPUS", "4,5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--seed-pairs", type=int, default=10)
    parser.add_argument("--deeper-seed-pairs", type=int, default=4)
    parser.add_argument("--confirm-seed-pairs", type=int, default=25)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--parallel-gates",
        type=int,
        default=2,
        help="Maximum independent checkpoint/mode evaluations to run concurrently.",
    )
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, "time": time.time(), **payload}, sort_keys=True), flush=True)


def checkpoint(update: int) -> Path:
    return CAMPAIGN / f"checkpoints/updates-{update:06d}"


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
            emit("gate_reusing_active_eval", output_dir=str(output_dir.relative_to(ROOT)))
            announced = True
        time.sleep(max(1.0, poll_seconds))


def wait_for_checkpoints(poll_seconds: float) -> None:
    expected = [checkpoint(update) / "algorithm_state.pkl" for update in (5, 10, 15, 20, 25)]
    while True:
        missing = [str(path.relative_to(ROOT)) for path in expected if not path.is_file()]
        if not missing:
            emit("checkpoints_ready", count=len(expected))
            return
        emit("waiting_for_checkpoints", missing=missing)
        time.sleep(max(1.0, poll_seconds))


def _resolved_checkpoint(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def validate_result_provenance(
    payload: dict[str, Any],
    *,
    candidate_checkpoint: Path,
    opponent_checkpoint: Path,
    candidate_label: str,
    opponent_label: str,
    seed_pairs: int,
    seed_base: int,
) -> None:
    """Fail closed when a resumable result belongs to a different gate."""

    config = payload.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("evaluation result lacks config provenance")
    expected = {
        "candidate_checkpoint": candidate_checkpoint.resolve(),
        "opponent_checkpoint": opponent_checkpoint.resolve(),
    }
    for key, path in expected.items():
        if key not in config or _resolved_checkpoint(config[key]) != path:
            raise RuntimeError(
                f"evaluation result {key} mismatch: "
                f"observed={config.get(key)!r} expected={str(path)!r}"
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
                f"evaluation result {key} mismatch: "
                f"observed={observed!r} expected={value!r}"
            )
    summary = payload.get("summary")
    if not isinstance(summary, dict) or int(summary.get("games", -1)) != 2 * int(seed_pairs):
        raise RuntimeError(
            "evaluation result is not a complete side-swapped seed block: "
            f"games={None if not isinstance(summary, dict) else summary.get('games')} "
            f"expected={2 * int(seed_pairs)}"
        )


def config_args(mode: str) -> list[str]:
    if mode == "raw":
        return ["--no-search"]
    if mode == "d2":
        return [
            "--search", "--initial-simulations", "32", "--medium-simulations", "32",
            "--max-simulations", "32", "--meaningful-depth", "2",
            "--max-atomic-steps", "1536", "--max-expanded-nodes", "32768",
        ]
    if mode == "d4":
        return [
            "--search", "--initial-simulations", "128", "--medium-simulations", "128",
            "--max-simulations", "128", "--meaningful-depth", "4",
            "--max-atomic-steps", "1536", "--max-expanded-nodes", "262144",
        ]
    if mode == "d4_512":
        return [
            "--search", "--initial-simulations", "512", "--medium-simulations", "512",
            "--max-simulations", "512", "--meaningful-depth", "4",
            "--max-atomic-steps", "1536", "--max-expanded-nodes", "1048576",
        ]
    raise ValueError(mode)


def run_gate(
    update: int,
    mode: str,
    *,
    seed_pairs: int,
    workers: int,
    output_label: str | None = None,
    seed_base: int = 97_400_000,
) -> dict[str, Any]:
    output_dir = OUTPUT / f"u{update}/{output_label or mode}"
    result_path = output_dir / "results.json"
    wait_for_external_eval(output_dir, result_path)
    candidate_checkpoint = checkpoint(update)
    candidate_label = f"search-scaling-u{update}-{mode}"
    opponent_label = f"phase3-u75-parent-{mode}"
    if not result_path.is_file():
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "scripts/eval_puct_agent_pair_parallel.py",
            "--candidate-checkpoint", str(candidate_checkpoint),
            "--opponent-checkpoint", str(PARENT),
            "--candidate-label", candidate_label,
            "--opponent-label", opponent_label,
            "--output-dir", str(output_dir),
            "--seed-pairs", str(seed_pairs),
            "--seed-base", str(seed_base),
            "--workers", str(workers),
            "--gpus", GPUS,
            "--replicas-per-checkpoint", "1",
            "--max-inference-batch-size", "32",
            "--max-inference-delay-ms", "3",
            "--inference-timeout-seconds", "300",
            "--belief-particles", "32",
            "--max-episode-steps", "1200",
            "--log-games",
            "--resume",
            *config_args(mode),
        ]
        emit("gate_started", update=update, mode=mode, command=command)
        with (output_dir / "run.log").open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(f"gate failed for u{update}/{mode}: {completed.returncode}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validate_result_provenance(
        payload,
        candidate_checkpoint=candidate_checkpoint,
        opponent_checkpoint=PARENT,
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
            f"cutoff-contaminated gate for u{update}/{mode}: "
            f"atomic={summary.get('atomic_horizon_cutoffs')} "
            f"nodes={summary.get('node_budget_cutoffs')}"
        )
    emit(
        "gate_complete",
        update=update,
        mode=mode,
        games=summary["games"],
        score=summary["score"],
        us=summary["by_candidate_side"]["us"]["score"],
        ussr=summary["by_candidate_side"]["ussr"]["score"],
        elo_delta=summary["elo_delta"],
    )
    return summary


def run_gate_jobs(
    jobs: list[tuple[str, dict[str, Any]]],
    *,
    max_parallel: int,
) -> dict[str, dict[str, Any]]:
    """Run independent, resumable gates with bounded process-level parallelism."""

    if not jobs:
        return {}
    workers = max(1, min(int(max_parallel), len(jobs)))
    if workers == 1:
        return {key: run_gate(**kwargs) for key, kwargs in jobs}

    results: dict[str, dict[str, Any]] = {}
    emit("gate_batch_started", jobs=[key for key, _ in jobs], parallelism=workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_gate, **kwargs): key for key, kwargs in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            results[key] = future.result()
    emit("gate_batch_complete", jobs=sorted(results), parallelism=workers)
    return results


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    wait_for_checkpoints(args.poll_seconds)
    records: dict[str, dict[str, Any]] = {}
    preliminary_jobs = [
        (
            f"u{update}:{mode}",
            {
                "update": update,
                "mode": mode,
                "seed_pairs": args.seed_pairs,
                "workers": args.workers,
            },
        )
        for update in (5, 10, 15, 20, 25)
        for mode in ("raw", "d2")
    ]
    records.update(
        run_gate_jobs(preliminary_jobs, max_parallel=args.parallel_gates)
    )

    eligible = [
        update for update in (5, 10, 15, 20, 25)
        if records[f"u{update}:raw"]["score"] >= 0.40
    ]
    ranked = sorted(
        eligible or [5, 10, 15, 20, 25],
        key=lambda update: (
            min(records[f"u{update}:raw"]["score"], records[f"u{update}:d2"]["score"]),
            records[f"u{update}:raw"]["score"] + records[f"u{update}:d2"]["score"],
        ),
        reverse=True,
    )
    d4_candidates = ranked[:2]
    records.update(
        run_gate_jobs(
            [
                (
                    f"u{update}:d4",
                    {
                        "update": update,
                        "mode": "d4",
                        "seed_pairs": args.seed_pairs,
                        "workers": args.workers,
                    },
                )
                for update in d4_candidates
            ],
            max_parallel=args.parallel_gates,
        )
    )

    deeper_eligible = [
        update for update in d4_candidates
        if records[f"u{update}:raw"]["score"] >= 0.40
        and records[f"u{update}:d2"]["score"] >= 0.45
        and records[f"u{update}:d4"]["score"] >= 0.45
    ]
    if deeper_eligible:
        best = max(
            deeper_eligible,
            key=lambda update: sum(
                records[f"u{update}:{mode}"]["score"] for mode in ("raw", "d2", "d4")
            ),
        )
        records[f"u{best}:d4_512"] = run_gate(
            best,
            "d4_512",
            seed_pairs=args.deeper_seed_pairs,
            workers=min(args.workers, 2 * args.deeper_seed_pairs),
        )
    else:
        best = None

    confirmatory_candidate = None
    confirmatory_records: dict[str, dict[str, Any]] = {}
    preliminary_viable = [
        update
        for update in d4_candidates
        if records[f"u{update}:raw"]["score"] >= 0.45
        and records[f"u{update}:d2"]["score"] >= 0.50
        and records[f"u{update}:d4"]["score"] >= 0.45
        and min(
            float(records[f"u{update}:{mode}"]["by_candidate_side"][side]["score"])
            for mode in ("raw", "d2", "d4")
            for side in ("us", "ussr")
        ) >= 0.25
    ]
    if preliminary_viable:
        confirmatory_candidate = max(
            preliminary_viable,
            key=lambda update: (
                min(records[f"u{update}:{mode}"]["score"] for mode in ("raw", "d2", "d4")),
                sum(records[f"u{update}:{mode}"]["score"] for mode in ("raw", "d2", "d4")),
            ),
        )
        confirmatory_records.update(
            run_gate_jobs(
                [
                    (
                        mode,
                        {
                            "update": confirmatory_candidate,
                            "mode": mode,
                            "seed_pairs": int(args.confirm_seed_pairs),
                            "workers": min(
                                int(args.workers), 2 * int(args.confirm_seed_pairs)
                            ),
                            "output_label": f"confirmatory/{mode}",
                            "seed_base": 97_420_000,
                        },
                    )
                    for mode in ("raw", "d2", "d4")
                ],
                max_parallel=args.parallel_gates,
            )
        )

    summary = {
        "parent": str(PARENT.resolve()),
        "campaign": str(CAMPAIGN.resolve()),
        "d4_candidates": d4_candidates,
        "deeper_candidate": best,
        "confirmatory_candidate": confirmatory_candidate,
        "confirmatory_records": confirmatory_records,
        "records": records,
    }
    (OUTPUT / "gate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    emit("promotion_gates_complete", output=str((OUTPUT / "gate_summary.json").relative_to(ROOT)))


if __name__ == "__main__":
    main()
