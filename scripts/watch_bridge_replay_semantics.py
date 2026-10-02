#!/usr/bin/env python3
"""Incrementally replay committed self-play games against the current bridge."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from struggle_ai.bridge_replay_audit import audit_record_semantics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def retain_current_reports(
    reports_by_source: dict[str, dict], current_sources: set[str]
) -> tuple[dict[str, dict], set[str]]:
    """Drop cached audits for completed-game records that no longer exist."""

    removed_sources = set(reports_by_source).difference(current_sources)
    return (
        {
            source: report
            for source, report in reports_by_source.items()
            if source in current_sources
        },
        removed_sources,
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reports_by_source: dict[str, dict] = {}
    if args.output.is_file():
        try:
            previous = json.loads(args.output.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
        reports_by_source = {
            str(report["source_game"]): report
            for report in previous.get("game_reports", [])
        }
    while True:
        records = [
            path
            for collection in args.collection
            for path in sorted((collection / "completed_games").glob("game-*.json"))
        ]
        current_sources = {
            str(json.loads(path.read_text(encoding="utf-8"))["source_game"])
            for path in records
        }
        reports_by_source, removed_sources = retain_current_reports(
            reports_by_source, current_sources
        )
        pending = []
        for path in records:
            source = str(json.loads(path.read_text(encoding="utf-8"))["source_game"])
            if source not in reports_by_source:
                pending.append(path)
        if pending:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(int(args.workers), len(pending)))
            ) as executor:
                for report in executor.map(audit_record_semantics, pending):
                    reports_by_source[str(report["source_game"])] = report
        count = len(records)
        if pending or removed_sources or not args.output.is_file():
            reports = [reports_by_source[source] for source in sorted(reports_by_source)]
            failures = [report for report in reports if not report.get("passed")]
            summary = {
                "games": len(reports),
                "passed_games": len(reports) - len(failures),
                "failed_games": len(failures),
                "workers": int(args.workers),
                "passed": not failures,
                "failures": failures,
                "game_reports": reports,
            }
            summary["expected_games"] = int(args.expected_games)
            summary["complete"] = count == int(args.expected_games)
            summary["updated_at"] = time.time()
            args.output.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "event": "bridge_replay_audit_updated",
                        "games": summary["games"],
                        "failed_games": summary["failed_games"],
                        "passed": summary["passed"],
                        "complete": summary["complete"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if not summary["passed"]:
                raise RuntimeError(
                    f"bridge replay audit failed: {summary['failures'][:4]}"
                )
            if summary["complete"]:
                return
        elif (
            count == int(args.expected_games)
            and len(reports_by_source) == int(args.expected_games)
            and all(report.get("passed") for report in reports_by_source.values())
        ):
            return
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
