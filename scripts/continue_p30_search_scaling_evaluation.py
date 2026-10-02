#!/usr/bin/env python3
"""Continue p30/u25 promotion evaluation after actor amortization completes."""

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
PARENT = (
    ROOT
    / "runs/search_distill/search-scaling-phase3u75-parentactor-p30u25-composed-20260828/checkpoints/composed"
)
CAMPAIGN = ROOT / "runs/search_distill/search-scaling-p30u25-amortization-u25-gpu4-20260828"
GATE = ROOT / "runs/eval/search-scaling-p30u25-amortization-gates-20260828/gate_summary.json"
ANOMALY = ROOT / "runs/diagnostics/search-scaling-p30u25-amortization-anomaly-gate-20260828/summary.json"
CROSS_MODE = ROOT / "runs/eval/search-scaling-p30u25-amortization-cross-mode-20260828/summary.json"
DECISION = CAMPAIGN / "promotion_decision.json"
COMPLETION_AUDIT = CAMPAIGN / "completion_audit.json"
CONVERGENCE = (
    ROOT
    / "runs/search_scaling/phase3u75-p30u25-headlinefix-convergence-20260829/"
    "final_summary.json"
)
OPERATOR = (
    ROOT
    / "runs/eval/search-scaling-phase3u75-p30u25-headlinefix-d4-vs-d2-20260829/"
    "evidence.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, "time": time.time(), **payload}, sort_keys=True), flush=True)


def wait_for(path: Path, *, poll_seconds: float, stage: str) -> None:
    while not path.is_file():
        emit("stage_waiting", stage=stage, path=str(path.relative_to(ROOT)))
        time.sleep(max(1.0, float(poll_seconds)))
    emit("stage_dependency_ready", stage=stage, path=str(path.relative_to(ROOT)))


def run_once(
    command: list[str],
    *,
    done_path: Path,
    log_path: Path,
    stage: str,
    accept_nonzero_with_done: bool = False,
) -> None:
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
    if not done_path.is_file() or (
        completed.returncode and not bool(accept_nonzero_with_done)
    ):
        raise RuntimeError(
            f"{stage} failed: returncode={completed.returncode}, expected={done_path}"
        )
    emit(
        "stage_complete",
        stage=stage,
        done_path=str(done_path.relative_to(ROOT)),
        returncode=completed.returncode,
    )


def record_skipped_cross_mode(anomaly: dict[str, Any]) -> None:
    """Persist an auditable scaling failure without spending games after anomaly failure."""

    if CROSS_MODE.is_file():
        return
    CROSS_MODE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": anomaly.get("checkpoint"),
        "records": {},
        "confirmatory_records": {},
        "confirmatory_scaling_supported": False,
        "skipped_reason": "frozen_anomaly_gate_failed",
        "anomaly_update": anomaly.get("update"),
    }
    temporary = CROSS_MODE.with_suffix(CROSS_MODE.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(CROSS_MODE)
    emit(
        "stage_skipped",
        stage="same_checkpoint_search_scaling_gates",
        reason=payload["skipped_reason"],
        done_path=str(CROSS_MODE.relative_to(ROOT)),
    )


def main() -> None:
    args = parse_args()
    wait_for(
        CAMPAIGN / "amortization_summary.json",
        poll_seconds=args.poll_seconds,
        stage="actor_amortization",
    )
    run_once(
        [
            "env",
            "SEARCH_SCALING_GPUS=4,5,6,7",
            sys.executable,
            "scripts/run_p30_search_scaling_promotion_gates.py",
            "--poll-seconds",
            "30",
            "--seed-pairs",
            "10",
            "--deeper-seed-pairs",
            "4",
            "--confirm-seed-pairs",
            "25",
            "--workers",
            "24",
            "--parallel-gates",
            "2",
        ],
        done_path=GATE,
        log_path=CAMPAIGN / "promotion_gates.log",
        stage="raw_d2_d4_parent_gates",
    )
    run_once(
        [
            "env",
            "SEARCH_SCALING_ANOMALY_DEVICES=cuda:4,cuda:5",
            sys.executable,
            "scripts/run_p30_search_scaling_anomaly_gate.py",
            "--poll-seconds",
            "30",
        ],
        done_path=ANOMALY,
        log_path=CAMPAIGN / "anomaly_gate.log",
        stage="frozen_anomaly_gate",
        accept_nonzero_with_done=True,
    )
    anomaly = json.loads(ANOMALY.read_text(encoding="utf-8"))
    if bool(anomaly.get("passed")):
        run_once(
            [
                "env",
                "SEARCH_SCALING_GPUS=4,5,6,7",
                sys.executable,
                "scripts/run_p30_search_scaling_cross_mode_gates.py",
                "--poll-seconds",
                "30",
                "--seed-pairs",
                "10",
                "--deeper-seed-pairs",
                "4",
                "--confirm-seed-pairs",
                "25",
                "--confirm-deeper-seed-pairs",
                "8",
                "--confirm-d6-seed-pairs",
                "4",
                "--parallel-gates",
                "2",
            ],
            done_path=CROSS_MODE,
            log_path=CAMPAIGN / "cross_mode_gates.log",
            stage="same_checkpoint_search_scaling_gates",
        )
    else:
        record_skipped_cross_mode(anomaly)
    run_once(
        [
            sys.executable,
            "scripts/finalize_search_scaling_promotion.py",
            "--gate-summary",
            str(GATE),
            "--anomaly-summary",
            str(ANOMALY),
            "--cross-mode-summary",
            str(CROSS_MODE),
            "--parent",
            str(PARENT),
            "--output",
            str(DECISION),
        ],
        done_path=DECISION,
        log_path=CAMPAIGN / "promotion_decision.log",
        stage="fail_closed_general_parent_decision",
    )
    run_once(
        [
            sys.executable,
            "scripts/audit_p30_search_scaling_completion.py",
            "--convergence-summary",
            str(CONVERGENCE),
            "--operator-gate",
            str(OPERATOR),
            "--amortization-summary",
            str(CAMPAIGN / "amortization_summary.json"),
            "--promotion-gate",
            str(GATE),
            "--anomaly-summary",
            str(ANOMALY),
            "--cross-mode-summary",
            str(CROSS_MODE),
            "--decision",
            str(DECISION),
            "--output-json",
            str(COMPLETION_AUDIT),
            "--output-markdown",
            str(CAMPAIGN / "completion_audit.md"),
        ],
        done_path=COMPLETION_AUDIT,
        log_path=CAMPAIGN / "completion_audit.log",
        stage="search_scaling_completion_audit",
    )
    emit("search_scaling_iteration_complete", decision=json.loads(DECISION.read_text()))


if __name__ == "__main__":
    main()
