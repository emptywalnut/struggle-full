#!/usr/bin/env python3
"""Fail if the final Steam-trace audit artifact is internally inconsistent."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
result = json.loads((HERE / "audit_results.json").read_text())
checks = result["deterministic_checks"]
statuses = checks["by_status"]

assert checks["total"] == sum(statuses.values()) == 7569
assert checks["exact_matches"] == statuses["match"] == 7444
assert checks["parser_limitations"] == statuses["parser_limitation"] == 125
assert checks["hard_mismatches"] == statuses["hard_mismatch"] == 0
assert result["bridge_differentials"] == []
assert result["materialized_differentials"] == []
assert len(result["confirmed_defects"]) == 6
assert all(item["classification"] == "match" for item in result["confirmed_defects"])

oas = sorted(
    (check["trace"], check["lines"][0], check["status"])
    for file_result in result["files"]
    for check in file_result["checks"]
    if check["mechanic"] == "card_removal_marker" and check["status"] == "parser_limitation"
)
assert oas == [
    ("3341939.txt", 322, "parser_limitation"),
    ("3346510.txt", 475, "parser_limitation"),
    ("3348222.txt", 514, "parser_limitation"),
    ("3352375.txt", 725, "parser_limitation"),
]
assert result["parse"]["unparsed_lines"] == 0
assert result["parse"]["recognized_fraction_nonblank"] == 1.0

print(json.dumps({
    "status": "passed",
    "counts": checks,
    "oas_parser_limitations": oas,
    "materialized_differentials": 0,
    "bridge_differentials": 0,
}, indent=2))
