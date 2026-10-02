#!/usr/bin/env python3
"""Durable entry point for the p30 search-scaling campaign.

The original orchestration source was truncated when the shared filesystem ran
out of space. Its complete CPython bytecode is retained beside this file. Load
that verified implementation, then replace only collector process ownership so
long complete-game jobs are children of the tmux server instead of this
supervisor's remote job.
"""

from __future__ import annotations

import json
import hashlib
import marshal
import shlex
import subprocess
import time
import types
from pathlib import Path
from typing import Any


_SOURCE_PATH = Path(__file__).resolve()
_PYC_PATH = _SOURCE_PATH.parent / "__pycache__" / (
    "continue_p30_search_scaling_amortization.retained.cpython-311.pyc"
)
if not _PYC_PATH.is_file():
    raise RuntimeError(f"retained orchestration bytecode is missing: {_PYC_PATH}")
_EXPECTED_PYC_SHA256 = "d48e2c369930ce303b2164431462eb5d5a2f09dc22cd71dee136fce715c25a06"
if hashlib.sha256(_PYC_PATH.read_bytes()).hexdigest() != _EXPECTED_PYC_SHA256:
    raise RuntimeError(f"retained orchestration bytecode checksum mismatch: {_PYC_PATH}")

with _PYC_PATH.open("rb") as _handle:
    _handle.read(16)
    _code = marshal.load(_handle)

_legacy = types.ModuleType("scripts._continue_p30_search_scaling_amortization_retained")
_legacy.__dict__.update(
    {
        "__file__": str(_SOURCE_PATH),
        "__package__": "scripts",
        "__name__": _legacy.__name__,
    }
)
exec(_code, _legacy.__dict__)

for _name, _value in _legacy.__dict__.items():
    if not _name.startswith("__"):
        globals()[_name] = _value


_legacy_run_once = _legacy.run_once


def _available_evidence_gpus(
    candidates: tuple[int, ...] = (4, 5, 6, 7),
    *,
    minimum_free_mib: int = 16_384,
    maximum_utilization: int = 45,
) -> list[int]:
    """Return GPUs with enough headroom for a convergence-search worker."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.free",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return []
    available: list[tuple[int, int, int]] = []
    for line in completed.stdout.splitlines():
        try:
            index_text, utilization_text, free_text = (
                part.strip() for part in line.split(",")
            )
            index = int(index_text)
            utilization = int(utilization_text)
            free_mib = int(free_text)
        except (TypeError, ValueError):
            continue
        if (
            index in candidates
            and utilization <= int(maximum_utilization)
            and free_mib >= int(minimum_free_mib)
        ):
            available.append((index, utilization, free_mib))
    available.sort(key=lambda item: (item[1], -item[2], item[0]))
    return [index for index, _, _ in available]


def _capacity_aware_evidence_command(
    command: list[str], *, poll_seconds: float = 60.0
) -> list[str]:
    """Wait for useful GPU capacity and rewrite the evidence GPU assignment."""

    while True:
        available = _available_evidence_gpus()
        if len(available) >= 2:
            selected = available[:4] if len(available) >= 4 else available[:2]
            rewritten = list(command)
            gpu_flag = rewritten.index("--gpus")
            rewritten[gpu_flag + 1] = ",".join(str(index) for index in selected)
            emit(
                "evidence_gpu_capacity_acquired",
                available=available,
                selected=selected,
            )
            return rewritten
        emit(
            "stage_waiting",
            stage="evidence_gpu_capacity",
            available=available,
            required=2,
        )
        time.sleep(max(1.0, float(poll_seconds)))


def _capacity_aware_collector_command(
    command: list[str],
    collection: Path,
    *,
    poll_seconds: float = 60.0,
) -> list[str]:
    """Route resumed games only to GPUs that currently have useful headroom."""

    while True:
        available = _available_evidence_gpus(
            minimum_free_mib=12_288,
            maximum_utilization=60,
        )
        if available:
            completed = len(list((collection / "completed_games").glob("game-*.json")))
            remaining = max(1, 32 - completed)
            selected = available[: min(4, remaining)]
            rewritten = list(command)
            gpu_flag = rewritten.index("--gpus")
            rewritten[gpu_flag + 1] = ",".join(str(index) for index in selected)
            worker_flag = rewritten.index("--workers")
            rewritten[worker_flag + 1] = str(
                min(int(rewritten[worker_flag + 1]), remaining)
            )
            emit(
                "collector_gpu_capacity_acquired",
                collection=str(collection.relative_to(ROOT)),
                completed=completed,
                remaining=remaining,
                available=available,
                selected=selected,
            )
            return rewritten
        emit(
            "stage_waiting",
            stage="collector_gpu_capacity",
            collection=str(collection.relative_to(ROOT)),
            available=[],
            required=1,
        )
        time.sleep(max(1.0, float(poll_seconds)))


def run_once(
    command: list[str], *, done_path: Path, log_path: Path, stage: str
) -> None:
    """Apply a live GPU-capacity gate before the expensive evidence sweep."""

    if stage == "headlinefix_search_convergence_and_operator_gate" and not done_path.is_file():
        command = _capacity_aware_evidence_command(command)
    _legacy_run_once(
        command,
        done_path=done_path,
        log_path=log_path,
        stage=stage,
    )


_legacy.run_once = run_once


_legacy_validate_teacher_coverage = _legacy.validate_teacher_coverage


def validate_teacher_coverage(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve the legacy volume gate and require broad side/era coverage."""

    totals = dict(_legacy_validate_teacher_coverage(summaries))
    eras = ("early", "mid", "late")
    sides = ("us", "ussr")
    by_era = {
        era: sum(int(summary.get(f"era:{era}", 0)) for summary in summaries)
        for era in eras
    }
    by_side_era = {
        f"{side}:{era}": sum(
            int(summary.get(f"side_era:{side}:{era}", 0))
            for summary in summaries
        )
        for side in sides
        for era in eras
    }
    reasons: list[str] = []
    if any(count < 128 for count in by_era.values()):
        reasons.append("era_coverage")
    if any(count < 48 for count in by_side_era.values()):
        reasons.append("side_era_coverage")
    if reasons:
        raise RuntimeError(
            "converged teacher coverage/balance gate failed: "
            + json.dumps(
                {
                    "reasons": reasons,
                    "by_era": by_era,
                    "by_side_era": by_side_era,
                    "totals": totals,
                },
                sort_keys=True,
            )
        )
    totals["by_era"] = by_era
    totals["by_side_era"] = by_side_era
    return totals


_legacy.validate_teacher_coverage = validate_teacher_coverage


_legacy_distillation_command = _legacy.distillation_command


def distillation_command(teacher_dirs: list[Path]) -> list[str]:
    """Use every war era in each actor-only amortization minibatch."""

    command = list(_legacy_distillation_command(teacher_dirs))
    command.extend(
        [
            "--min-search-examples-per-era",
            "16",
            "--min-search-examples-per-side-era",
            "4",
        ]
    )
    return command


_legacy.distillation_command = distillation_command


def record_training_batch_coverage() -> dict[str, Any]:
    """Attach optimizer-consumed era coverage to the durable campaign summary."""

    report_path = CAMPAIGN / "distillation_report.json"
    summary_path = CAMPAIGN / "amortization_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    history = list(report.get("history") or [])
    era_floor = int(report.get("min_search_examples_per_era", 0))
    side_era_floor = int(report.get("min_search_examples_per_side_era", 0))
    by_era = {
        era: min(
            (int(row.get(f"batch_era_{era}", -1)) for row in history),
            default=-1,
        )
        for era in ("early", "mid", "late")
    }
    by_side_era = {
        f"{side}:{era}": min(
            (
                int(row.get(f"batch_side_era_{side}_{era}", -1))
                for row in history
            ),
            default=-1,
        )
        for side in ("us", "ussr")
        for era in ("early", "mid", "late")
    }
    coverage = {
        "passed": bool(history)
        and era_floor > 0
        and side_era_floor > 0
        and all(value >= era_floor for value in by_era.values())
        and all(value >= side_era_floor for value in by_side_era.values()),
        "updates": len(history),
        "configured_min_per_era": era_floor,
        "configured_min_per_side_era": side_era_floor,
        "minimum_observed_by_era": by_era,
        "minimum_observed_by_side_era": by_side_era,
        "distillation_report": str(report_path.resolve()),
    }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["training_batch_coverage"] = coverage
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    if not coverage["passed"]:
        raise RuntimeError(
            "actor amortization batch coverage gate failed: "
            + json.dumps(coverage, sort_keys=True)
        )
    return coverage


COLLECTOR_TMUX_SESSIONS = {
    collection: f"ts-p30-headlinefix-collector-{index}"
    for index, collection in enumerate(COLLECTIONS)
}


def tmux_session_exists(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def launch_collector_tmux(collection: Path, seed_base: int) -> tuple[str, list[str]]:
    """Launch a collection independently through the persistent tmux server."""

    session = COLLECTOR_TMUX_SESSIONS[collection]
    command = collector_command(collection, seed_base)
    if tmux_session_exists(session):
        return session, command
    command = _capacity_aware_collector_command(command, collection)
    log_path = collection / "run-supervised-resume.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_command = " ".join(
        (
            f"cd {shlex.quote(str(ROOT))}",
            "&& exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
            shlex.join(command),
            f">> {shlex.quote(str(log_path))} 2>&1",
        )
    )
    completed = subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "/bin/bash",
            "-lc",
            shell_command,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"failed to launch collector tmux session {session}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return session, command


def wait_for_collections(
    *,
    poll_seconds: float,
    absence_polls_before_resume: int = 3,
) -> None:
    """Wait for durable shards and independently restart missing collectors."""

    absence_counts = {collection: 0 for collection in COLLECTIONS}
    while True:
        missing = [
            collection
            for collection in COLLECTIONS
            if not (collection / "collection_summary.json").is_file()
        ]
        if not missing:
            emit("stage_inputs_ready", stage="complete_game_collections", paths=len(COLLECTIONS))
            return
        emit(
            "stage_waiting",
            stage="complete_game_collections",
            missing=[str(path.relative_to(ROOT)) for path in missing],
        )
        for collection in missing:
            active = collector_active(collection) or tmux_session_exists(
                COLLECTOR_TMUX_SESSIONS[collection]
            )
            absence_counts[collection] = 0 if active else absence_counts[collection] + 1
            if active or absence_counts[collection] < int(absence_polls_before_resume):
                continue
            session, command = launch_collector_tmux(
                collection, COLLECTION_SEEDS[collection]
            )
            absence_counts[collection] = 0
            emit(
                "collector_resumed",
                collection=str(collection.relative_to(ROOT)),
                session=session,
                command=command,
            )
        time.sleep(max(1.0, float(poll_seconds)))


_legacy.wait_for_collections = wait_for_collections


if __name__ == "__main__":
    _legacy.main()
    record_training_batch_coverage()
