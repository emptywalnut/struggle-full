#!/usr/bin/env python3
"""Make a fail-closed general-parent decision from completed search-scaling gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_SCALING_LABELS = (
    "d2_vs_raw",
    "d4_128_vs_d2_32",
    "d4_512_vs_d4_128",
)


def _side_floor(records: dict[str, dict[str, Any]]) -> float:
    return min(
        float(record["by_candidate_side"][side]["score"])
        for record in records.values()
        for side in ("us", "ussr")
    )


def evaluate_promotion(
    gate: dict[str, Any],
    anomaly: dict[str, Any],
    cross_mode: dict[str, Any],
    *,
    parent: Path,
) -> dict[str, Any]:
    update = gate.get("confirmatory_candidate")
    confirm = gate.get("confirmatory_records", {})
    reasons: list[str] = []
    if update is None:
        reasons.append("no_preliminary_candidate_reached_confirmatory_evaluation")
    missing_modes = [mode for mode in ("raw", "d2", "d4") if mode not in confirm]
    if missing_modes:
        reasons.append(f"missing_confirmatory_parent_modes:{','.join(missing_modes)}")

    candidate = Path(str(anomaly.get("checkpoint", ""))) if anomaly.get("checkpoint") else None
    if update is not None and int(anomaly.get("update", -1)) != int(update):
        reasons.append("anomaly_checkpoint_update_mismatch")
    if not bool(anomaly.get("passed", False)):
        reasons.append("frozen_anomaly_gate_failed")

    if not missing_modes:
        raw, d2, d4 = (confirm[mode] for mode in ("raw", "d2", "d4"))
        if float(raw["score"]) < 0.45 or float(raw["paired_bootstrap_ci95_low"]) < 0.40:
            reasons.append("raw_parent_match_regression")
        if float(d2["score"]) <= 0.50 or float(d2["paired_bootstrap_ci95_low"]) < 0.50:
            reasons.append("d2_parent_improvement_not_statistically_supported")
        if float(d4["score"]) < 0.50 or float(d4["paired_bootstrap_ci95_low"]) < 0.45:
            reasons.append("d4_parent_compatibility_not_supported")
        if _side_floor(confirm) < 0.40:
            reasons.append("confirmatory_us_ussr_side_floor_failed")

    scaling = cross_mode.get("confirmatory_records", {})
    missing_scaling = [label for label in REQUIRED_SCALING_LABELS if label not in scaling]
    if missing_scaling:
        reasons.append(f"missing_confirmatory_scaling_modes:{','.join(missing_scaling)}")
    if not bool(cross_mode.get("confirmatory_scaling_supported", False)):
        reasons.append("positive_search_scaling_not_confirmed")
    if not bool(cross_mode.get("confirmatory_side_balance_supported", False)):
        reasons.append("search_scaling_us_ussr_balance_not_confirmed")
    if candidate is not None and cross_mode.get("checkpoint"):
        if candidate.resolve() != Path(str(cross_mode["checkpoint"])).resolve():
            reasons.append("cross_mode_checkpoint_mismatch")

    promoted = not reasons
    retained = candidate.resolve() if promoted and candidate is not None else parent.resolve()
    return {
        "promoted": promoted,
        "candidate_update": int(update) if update is not None else None,
        "candidate_checkpoint": str(candidate.resolve()) if candidate is not None else None,
        "retained_general_parent": str(retained),
        "previous_parent": str(parent.resolve()),
        "reasons": reasons,
        "criteria": {
            "raw_parent_match": "score>=0.45 and paired CI low>=0.40",
            "d2_parent_improvement": "score>0.50 and paired CI low>=0.50",
            "d4_parent_compatibility": "score>=0.50 and paired CI low>=0.45",
            "side_floor": ">=0.40 across confirmatory Raw/D2/D4",
            "search_scaling": list(REQUIRED_SCALING_LABELS),
            "search_scaling_side_floors": {
                "D2/32_vs_Raw": ">=0.40",
                "D4/128_vs_D2/32": ">=0.40",
                "D4/512_vs_D4/128": ">=0.25",
            },
            "anomaly_gate": "all four roots pass without cutoff/tie regression",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--anomaly-summary", type=Path, required=True)
    parser.add_argument("--cross-mode-summary", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate_promotion(
        json.loads(args.gate_summary.read_text(encoding="utf-8")),
        json.loads(args.anomaly_summary.read_text(encoding="utf-8")),
        json.loads(args.cross_mode_summary.read_text(encoding="utf-8")),
        parent=args.parent,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    pointer = args.output.parent / "GENERAL_PARENT"
    pointer.write_text(payload["retained_general_parent"] + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
