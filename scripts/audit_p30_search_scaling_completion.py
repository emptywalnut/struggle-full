#!/usr/bin/env python3
"""Build a fail-closed completion audit for the p30/u25 search-scaling iteration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def build_audit(
    *,
    convergence: dict[str, Any],
    operator_gate: dict[str, Any],
    amortization: dict[str, Any],
    promotion_gate: dict[str, Any],
    anomaly: dict[str, Any],
    cross_mode: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    status_counts = convergence.get("status_counts", {})
    root_count = sum(int(value) for value in status_counts.values())
    operator = operator_gate.get("summary", operator_gate)
    operator_sides = operator.get("by_candidate_side", {})
    teacher = amortization.get("teacher_totals", {})
    teacher_by_era = dict(teacher.get("by_era") or {})
    teacher_by_side_era = dict(teacher.get("by_side_era") or {})
    root_panel = dict(
        amortization.get("search_evidence", {}).get("root_panel_audit") or {}
    )
    anchor_provenance = amortization.get("anchor_provenance", {})
    reroot_audit = amortization.get("search_reroot_audit", {})
    bridge_replay_audit = amortization.get("bridge_replay_audit", {})
    checkpoints = amortization.get("checkpoints", [])
    freeze = amortization.get("freeze_verification", {})
    batch_coverage = amortization.get("training_batch_coverage", {})
    confirm = promotion_gate.get("confirmatory_records", {})
    required_modes = ("raw", "d2", "d4")
    scaling = cross_mode.get("confirmatory_records", {})
    required_scaling = ("d2_vs_raw", "d4_128_vs_d2_32", "d4_512_vs_d4_128")
    corrected_semantics = "headline_priority_fixed_v1"
    evidence_semantics_ok = (
        str(convergence.get("bridge_semantics")) == corrected_semantics
        and str(operator_gate.get("bridge_semantics")) == corrected_semantics
        and str(amortization.get("search_evidence", {}).get("bridge_semantics"))
        == corrected_semantics
    )

    checks = [
        _check(
            "search_convergence_panel",
            root_count == 24
            and int(status_counts.get("HORIZON_CONVERGENCE", 0)) > 0
            and evidence_semantics_ok,
            roots=root_count,
            status_counts=status_counts,
            bridge_semantics=convergence.get("bridge_semantics"),
            operator_bridge_semantics=operator_gate.get("bridge_semantics"),
            amortization_bridge_semantics=amortization.get("search_evidence", {}).get(
                "bridge_semantics"
            ),
        ),
        _check(
            "balanced_search_convergence_panel",
            bool(root_panel.get("passed"))
            and all(
                int(root_panel.get("by_era", {}).get(era, 0)) >= 8
                for era in ("early", "mid", "late")
            )
            and all(
                int(root_panel.get("by_side_era", {}).get(f"{side}|{era}", 0)) >= 4
                for side in ("us", "ussr")
                for era in ("early", "mid", "late")
            ),
            root_panel_audit=root_panel,
        ),
        _check(
            "d4_operator_strength",
            int(operator.get("games", 0)) >= 100
            and float(operator.get("score", 0.0)) > 0.5
            and float(operator.get("paired_bootstrap_ci95_low", 0.0)) > 0.5
            and min(
                (float(operator_sides.get(side, {}).get("score", 0.0)) for side in ("us", "ussr")),
                default=0.0,
            )
            > 0.5
            and int(operator.get("atomic_horizon_cutoffs", 1)) == 0
            and int(operator.get("node_budget_cutoffs", 1)) == 0,
            games=operator.get("games"),
            score=operator.get("score"),
            paired_ci_low=operator.get("paired_bootstrap_ci95_low"),
            by_candidate_side=operator_sides,
            atomic_horizon_cutoffs=operator.get("atomic_horizon_cutoffs"),
            node_budget_cutoffs=operator.get("node_budget_cutoffs"),
        ),
        _check(
            "complete_balanced_teacher_replay",
            sum(int(row.get("games_completed", 0)) for row in amortization.get("collections", []))
            == 128
            and int(teacher.get("accepted", 0)) >= 1024
            and min(int(teacher.get("us", 0)), int(teacher.get("ussr", 0))) >= 384
            and min(int(teacher.get("us", 0)), int(teacher.get("ussr", 0)))
            / max(1, int(teacher.get("accepted", 0)))
            >= 0.40
            and all(int(teacher_by_era.get(era, 0)) >= 128 for era in ("early", "mid", "late"))
            and all(
                int(teacher_by_side_era.get(f"{side}:{era}", 0)) >= 48
                for side in ("us", "ussr")
                for era in ("early", "mid", "late")
            ),
            games=sum(
                int(row.get("games_completed", 0)) for row in amortization.get("collections", [])
            ),
            teacher_totals=teacher,
        ),
        _check(
            "frozen_actor_checkpoint_sweep",
            bool(amortization.get("ready_for_promotion_gates"))
            and len(checkpoints) == 5
            and all(
                int(value) > 0
                for value in freeze.get("changed_actor_tensor_counts", {}).values()
            )
            and len(freeze.get("changed_actor_tensor_counts", {})) == 5,
            checkpoints=checkpoints,
            freeze_verification=freeze,
        ),
        _check(
            "balanced_optimizer_batches",
            bool(batch_coverage.get("passed"))
            and int(batch_coverage.get("updates", 0)) == 25
            and int(batch_coverage.get("configured_min_per_era", 0)) >= 16
            and int(batch_coverage.get("configured_min_per_side_era", 0)) >= 4
            and all(
                int(batch_coverage.get("minimum_observed_by_era", {}).get(era, 0))
                >= 16
                for era in ("early", "mid", "late")
            )
            and all(
                int(
                    batch_coverage.get("minimum_observed_by_side_era", {}).get(
                        f"{side}:{era}", 0
                    )
                )
                >= 4
                for side in ("us", "ussr")
                for era in ("early", "mid", "late")
            ),
            training_batch_coverage=batch_coverage,
        ),
        _check(
            "parent_anchor_provenance",
            bool(anchor_provenance.get("composition_verified"))
            and str(anchor_provenance.get("composed_parent", ""))
            == str(amortization.get("parent", ""))
            and bool(anchor_provenance.get("current_anchor_sources"))
            and all(
                str(source) == str(amortization.get("parent", ""))
                for source in anchor_provenance.get("current_anchor_sources", [])
            )
            and str(anchor_provenance.get("anchor_bridge_semantics", ""))
            == corrected_semantics
            and bool(anchor_provenance.get("legacy_anchor_replays_excluded", False)),
            provenance=anchor_provenance,
        ),
        _check(
            "independent_card_search_rerooting",
            bool(reroot_audit.get("passed"))
            and int(reroot_audit.get("corrected_games", -1)) == 128
            and not reroot_audit.get("corrected_missing_required_card_reroots"),
            corrected_games=reroot_audit.get("corrected_games"),
            corrected_missing=reroot_audit.get(
                "corrected_missing_required_card_reroots", []
            ),
            legacy_missing_count=len(
                reroot_audit.get("legacy_missing_required_card_reroots", [])
            ),
        ),
        _check(
            "current_bridge_full_game_replay",
            bool(bridge_replay_audit.get("passed"))
            and int(bridge_replay_audit.get("games", -1)) == 128
            and int(bridge_replay_audit.get("failed_games", -1)) == 0,
            games=bridge_replay_audit.get("games"),
            failed_games=bridge_replay_audit.get("failed_games"),
            failures=bridge_replay_audit.get("failures", []),
        ),
        _check(
            "matched_raw_d2_d4_parent_gates",
            all(mode in confirm for mode in required_modes),
            confirmatory_candidate=promotion_gate.get("confirmatory_candidate"),
            modes=sorted(confirm),
        ),
        _check(
            "frozen_anomaly_regression",
            bool(anomaly.get("passed", False)),
            update=anomaly.get("update"),
            failures=anomaly.get("failures", anomaly.get("reasons", [])),
        ),
        _check(
            "positive_search_scaling",
            bool(cross_mode.get("confirmatory_scaling_supported", False))
            and bool(cross_mode.get("confirmatory_side_balance_supported", False))
            and all(label in scaling for label in required_scaling),
            modes=sorted(scaling),
            supported=cross_mode.get("confirmatory_scaling_supported"),
            side_balance_supported=cross_mode.get("confirmatory_side_balance_supported"),
            side_floors=cross_mode.get("confirmatory_side_floors", {}),
        ),
        _check(
            "fail_closed_promotion",
            bool(decision.get("promoted", False)) and not decision.get("reasons"),
            promoted=decision.get("promoted"),
            candidate_update=decision.get("candidate_update"),
            retained_general_parent=decision.get("retained_general_parent"),
            reasons=decision.get("reasons", []),
        ),
    ]
    return {
        "goal_complete": all(check["passed"] for check in checks),
        "checks": checks,
        "retained_general_parent": decision.get("retained_general_parent"),
        "candidate_update": decision.get("candidate_update"),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Search-Scaling Completion Audit",
        "",
        f"Overall: **{'PASS' if audit['goal_complete'] else 'NOT COMPLETE'}**",
        "",
        "| Requirement | Status | Evidence |",
        "|---|---|---|",
    ]
    for check in audit["checks"]:
        evidence = json.dumps(check["evidence"], sort_keys=True, separators=(",", ":"))
        lines.append(
            f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | `{evidence}` |"
        )
    lines.extend(
        (
            "",
            f"Retained parent: `{audit.get('retained_general_parent')}`",
            f"Candidate update: `{audit.get('candidate_update')}`",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convergence-summary", type=Path, required=True)
    parser.add_argument("--operator-gate", type=Path, required=True)
    parser.add_argument("--amortization-summary", type=Path, required=True)
    parser.add_argument("--promotion-gate", type=Path, required=True)
    parser.add_argument("--anomaly-summary", type=Path, required=True)
    parser.add_argument("--cross-mode-summary", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    def load(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    audit = build_audit(
        convergence=load(args.convergence_summary),
        operator_gate=load(args.operator_gate),
        amortization=load(args.amortization_summary),
        promotion_gate=load(args.promotion_gate),
        anomaly=load(args.anomaly_summary),
        cross_mode=load(args.cross_mode_summary),
        decision=load(args.decision),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
