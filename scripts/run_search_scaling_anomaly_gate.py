#!/usr/bin/env python3
"""Run the frozen four-root anomaly gate for the best search-scaling child."""

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
GATE_DIR = ROOT / "runs/eval/search-scaling-phase3u75-amortization-gates-20260821"
CAMPAIGN = ROOT / "runs/search_distill/search-scaling-phase3u75-amortization-u25-gpu4-20260821"
OUTPUT = ROOT / "runs/diagnostics/search-scaling-phase3u75-amortization-anomaly-gate-20260821"
ANOMALY_DEVICES = tuple(
    item.strip()
    for item in os.environ.get("SEARCH_SCALING_ANOMALY_DEVICES", "cuda:4,cuda:5").split(",")
    if item.strip()
)
if len(ANOMALY_DEVICES) != 2:
    raise RuntimeError("SEARCH_SCALING_ANOMALY_DEVICES must contain exactly two devices")
EXPECTED_MATRIX = {(2, 32), (4, 128), (6, 512)}
SUSPICIOUS = {
    "ussr_uk_overcontrol": "uk",
    "us_italy_overcontrol": "italy",
    "ussr_mexico_realignment": "mexico",
    "us_venezuela_realignment": "venezuela",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--candidate-update",
        type=int,
        default=None,
        help=(
            "Use an explicitly selected candidate update instead of waiting for the "
            "promotion summary. This is safe once its Raw/D2/D4 gates are complete."
        ),
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=None,
        help="Probe an explicit frozen checkpoint, including interpolated candidates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the durable anomaly output directory.",
    )
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, "time": time.time(), **payload}, sort_keys=True), flush=True)


def wait_for(path: Path, poll_seconds: float) -> None:
    while not path.is_file():
        emit("waiting", path=str(path.relative_to(ROOT)))
        time.sleep(max(1.0, poll_seconds))


def select_candidate(payload: dict[str, Any]) -> int:
    confirmatory = payload.get("confirmatory_candidate")
    if confirmatory is not None:
        modes = payload.get("confirmatory_records", {})
        if all(mode in modes for mode in ("raw", "d2", "d4")):
            return int(confirmatory)
    records = payload.get("records", {})
    candidates: list[tuple[tuple[float, float, float], int]] = []
    for update in payload.get("d4_candidates", []):
        scores = []
        for mode in ("raw", "d2", "d4"):
            record = records.get(f"u{int(update)}:{mode}")
            if not record:
                break
            scores.append(float(record["score"]))
        if len(scores) == 3:
            candidates.append(((min(scores), sum(scores), scores[2]), int(update)))
    if not candidates:
        raise RuntimeError("promotion summary has no candidate with complete Raw/D2/D4 gates")
    return max(candidates)[1]


def normalize_action(value: str) -> str:
    return value.split(" [", 1)[0].replace(" ", "").replace("-", "").casefold()


def suspicious_prior(result: dict[str, Any], suspicious: str) -> float | None:
    if result["spec"]["kind"] == "influence":
        actor = result["detail_actor"]
        return float(actor["logged_action_probability"])
    for row in result["search"][0]["root_table"]:
        if normalize_action(str(row["action"])) == suspicious:
            return float(row["P"])
    return None


def summarize_result(path: Path, checkpoint: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    name = str(result["spec"]["name"])
    suspicious = SUSPICIOUS[name]
    if Path(result["checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError(f"checkpoint mismatch in {path}")
    rows = list(result.get("search", []))
    found = {(int(row["depth"]), int(row["budget"])) for row in rows}
    if found != EXPECTED_MATRIX:
        raise RuntimeError(f"incomplete anomaly matrix for {name}: {sorted(found)}")
    selected = [
        row for row in rows if normalize_action(str(row["choice"])) == suspicious
    ]
    cutoffs = sum(
        int(row.get("atomic_horizon_cutoffs", 0)) + int(row.get("node_budget_cutoffs", 0))
        for row in rows
    )
    return {
        "root": name,
        "suspicious_action": suspicious,
        "suspicious_prior": suspicious_prior(result, suspicious),
        "search_runs": len(rows),
        "suspicious_selected": len(selected),
        "cutoffs": cutoffs,
        "choices": [
            {
                "depth": int(row["depth"]),
                "budget": int(row["budget"]),
                "repeat": int(row["repeat"]),
                "choice": row["choice"],
                "Q": float(row["choice_Q"]),
                "root_Q_gap": float(row["root_Q_gap"]),
                "visit_tie_count": int(row["root_selection"]["visit_tie_count"]),
                "q_override": bool(row["root_selection"]["q_override"]),
            }
            for row in rows
        ],
    }


def probe_command(
    checkpoint: Path, output_dir: Path, positions: str, search_root: str, device: str
) -> list[str]:
    return [
        sys.executable,
        "scripts/probe_d4_anomalies.py",
        "--checkpoint", str(checkpoint),
        "--output-dir", str(output_dir),
        "--positions", positions,
        "--search-root", search_root,
        "--device", device,
        "--belief-particles", "32",
        "--matrix", "2:32:2,4:128:2,6:512:1",
        "--max-atomic-steps", "1536",
        "--max-expanded-nodes", "32768",
        "--resume",
    ]


def main() -> None:
    args = parse_args()
    if args.candidate_checkpoint is not None and args.candidate_update is not None:
        raise RuntimeError("provide only one of --candidate-checkpoint and --candidate-update")
    if args.candidate_checkpoint is not None:
        checkpoint = args.candidate_checkpoint.expanduser().resolve()
        update = None
    elif args.candidate_update is None:
        gate_summary = GATE_DIR / "gate_summary.json"
        wait_for(gate_summary, args.poll_seconds)
        payload = json.loads(gate_summary.read_text(encoding="utf-8"))
        update = select_candidate(payload)
        checkpoint = CAMPAIGN / f"checkpoints/updates-{update:06d}"
    else:
        update = int(args.candidate_update)
        checkpoint = CAMPAIGN / f"checkpoints/updates-{update:06d}"
    if not (checkpoint / "algorithm_state.pkl").is_file():
        raise RuntimeError(f"selected checkpoint is incomplete: {checkpoint}")
    output = args.output_dir.expanduser().resolve() if args.output_dir else OUTPUT

    jobs = [
        (
            "influence",
            probe_command(
                checkpoint,
                output / "influence",
                "ussr_uk_overcontrol,us_italy_overcontrol",
                "detail",
                ANOMALY_DEVICES[0],
            ),
        ),
        (
            "realignment",
            probe_command(
                checkpoint,
                output / "realignment",
                "ussr_mexico_realignment,us_venezuela_realignment",
                "target",
                ANOMALY_DEVICES[1],
            ),
        ),
    ]
    processes: list[tuple[str, subprocess.Popen[Any], Any]] = []
    for label, command in jobs:
        output_dir = output / label
        output_dir.mkdir(parents=True, exist_ok=True)
        handle = (output_dir / "run.log").open("a", encoding="utf-8")
        emit("probe_started", label=label, command=command)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((label, process, handle))
    failures = []
    for label, process, handle in processes:
        returncode = process.wait()
        handle.close()
        emit("probe_finished", label=label, returncode=returncode)
        if returncode:
            failures.append((label, returncode))
    if failures:
        raise RuntimeError(f"anomaly probes failed: {failures}")

    paths = [
        output / "influence/ussr_uk_overcontrol.json",
        output / "influence/us_italy_overcontrol.json",
        output / "realignment/ussr_mexico_realignment.json",
        output / "realignment/us_venezuela_realignment.json",
    ]
    roots = [summarize_result(path, checkpoint) for path in paths]
    passed = all(row["suspicious_selected"] == 0 and row["cutoffs"] == 0 for row in roots)
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "update": update,
        "passed": passed,
        "information_masking": "validated_by_probe_harness",
        "root_search_mechanics": "repaired_configuration",
        "roots": roots,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    emit("anomaly_gate_complete", passed=passed, update=update)
    if not passed:
        raise RuntimeError("frozen anomaly gate failed")


if __name__ == "__main__":
    main()
