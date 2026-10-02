from __future__ import annotations

from scripts.audit_p30_search_scaling_completion import build_audit, render_markdown


def valid_inputs() -> dict:
    return {
        "convergence": {
            "bridge_semantics": "headline_priority_fixed_v1",
            "status_counts": {
                "HORIZON_CONVERGENCE": 5,
                "NEAR_TIE": 12,
                "UNRESOLVED": 7,
            }
        },
        "operator_gate": {
            "bridge_semantics": "headline_priority_fixed_v1",
            "summary": {
                "games": 144,
                "score": 0.60,
                "paired_bootstrap_ci95_low": 0.53,
                "by_candidate_side": {"us": {"score": 0.61}, "ussr": {"score": 0.59}},
                "atomic_horizon_cutoffs": 0,
                "node_budget_cutoffs": 0,
            }
        },
        "amortization": {
            "parent": "/parent",
            "collections": [{"games_completed": 32}] * 4,
            "teacher_totals": {
                "accepted": 2000,
                "us": 1000,
                "ussr": 1000,
                "by_era": {"early": 668, "mid": 666, "late": 666},
                "by_side_era": {
                    "us:early": 334,
                    "us:mid": 333,
                    "us:late": 333,
                    "ussr:early": 334,
                    "ussr:mid": 333,
                    "ussr:late": 333,
                },
            },
            "checkpoints": [f"u{update}" for update in (5, 10, 15, 20, 25)],
            "freeze_verification": {
                "changed_actor_tensor_counts": {
                    f"u{update}": 8 for update in (5, 10, 15, 20, 25)
                }
            },
            "ready_for_promotion_gates": True,
            "training_batch_coverage": {
                "passed": True,
                "updates": 25,
                "configured_min_per_era": 16,
                "configured_min_per_side_era": 4,
                "minimum_observed_by_era": {
                    "early": 16,
                    "mid": 16,
                    "late": 16,
                },
                "minimum_observed_by_side_era": {
                    f"{side}:{era}": 4
                    for side in ("us", "ussr")
                    for era in ("early", "mid", "late")
                },
            },
            "anchor_provenance": {
                "composition_verified": True,
                "composed_parent": "/parent",
                "actor_checkpoint": "/actor",
                "current_anchor_sources": ["/parent"] * 4,
                "anchor_bridge_semantics": "headline_priority_fixed_v1",
                "legacy_anchor_replays_excluded": True,
            },
            "search_reroot_audit": {
                "passed": True,
                "corrected_games": 128,
                "corrected_missing_required_card_reroots": [],
                "legacy_missing_required_card_reroots": [{"step": 20}],
            },
            "bridge_replay_audit": {
                "passed": True,
                "games": 128,
                "failed_games": 0,
                "failures": [],
            },
            "search_evidence": {
                "bridge_semantics": "headline_priority_fixed_v1",
                "root_panel_audit": {
                    "passed": True,
                    "by_era": {"early": 8, "mid": 8, "late": 8},
                    "by_side_era": {
                        f"{side}|{era}": 4
                        for side in ("us", "ussr")
                        for era in ("early", "mid", "late")
                    },
                },
            },
        },
        "promotion_gate": {
            "confirmatory_candidate": 10,
            "confirmatory_records": {"raw": {}, "d2": {}, "d4": {}},
        },
        "anomaly": {"passed": True, "update": 10},
        "cross_mode": {
            "confirmatory_scaling_supported": True,
            "confirmatory_side_balance_supported": True,
            "confirmatory_side_floors": {
                "d2_vs_raw": 0.52,
                "d4_128_vs_d2_32": 0.50,
                "d4_512_vs_d4_128": 0.38,
            },
            "confirmatory_records": {
                "d2_vs_raw": {},
                "d4_128_vs_d2_32": {},
                "d4_512_vs_d4_128": {},
            },
        },
        "decision": {
            "promoted": True,
            "candidate_update": 10,
            "retained_general_parent": "/child",
            "reasons": [],
        },
    }


def test_complete_audit_requires_every_gate() -> None:
    audit = build_audit(**valid_inputs())
    assert audit["goal_complete"] is True
    assert all(check["passed"] for check in audit["checks"])
    assert "Overall: **PASS**" in render_markdown(audit)


def test_rejected_promotion_keeps_goal_incomplete() -> None:
    inputs = valid_inputs()
    inputs["decision"] = {
        "promoted": False,
        "candidate_update": 10,
        "retained_general_parent": "/parent",
        "reasons": ["d4_parent_compatibility_not_supported"],
    }
    audit = build_audit(**inputs)
    assert audit["goal_complete"] is False
    promotion = next(check for check in audit["checks"] if check["name"] == "fail_closed_promotion")
    assert promotion["passed"] is False
    assert "NOT COMPLETE" in render_markdown(audit)


def test_completion_audit_rejects_search_scaling_side_collapse() -> None:
    inputs = valid_inputs()
    inputs["cross_mode"]["confirmatory_side_balance_supported"] = False
    audit = build_audit(**inputs)
    scaling = next(check for check in audit["checks"] if check["name"] == "positive_search_scaling")
    assert scaling["passed"] is False
    assert audit["goal_complete"] is False


def test_completion_audit_rejects_unbalanced_convergence_panel() -> None:
    inputs = valid_inputs()
    inputs["amortization"]["search_evidence"]["root_panel_audit"]["by_side_era"][
        "us|late"
    ] = 1
    audit = build_audit(**inputs)
    panel = next(
        check
        for check in audit["checks"]
        if check["name"] == "balanced_search_convergence_panel"
    )
    assert panel["passed"] is False
    assert audit["goal_complete"] is False


def test_completion_audit_rejects_teacher_era_collapse() -> None:
    inputs = valid_inputs()
    inputs["amortization"]["teacher_totals"]["by_era"]["late"] = 40
    inputs["amortization"]["teacher_totals"]["by_side_era"]["us:late"] = 20
    inputs["amortization"]["teacher_totals"]["by_side_era"]["ussr:late"] = 20
    audit = build_audit(**inputs)
    teacher = next(
        check
        for check in audit["checks"]
        if check["name"] == "complete_balanced_teacher_replay"
    )
    assert teacher["passed"] is False
    assert audit["goal_complete"] is False


def test_completion_audit_rejects_optimizer_batch_era_collapse() -> None:
    inputs = valid_inputs()
    inputs["amortization"]["training_batch_coverage"][
        "minimum_observed_by_side_era"
    ]["ussr:late"] = 0
    audit = build_audit(**inputs)
    coverage = next(
        check for check in audit["checks"] if check["name"] == "balanced_optimizer_batches"
    )
    assert coverage["passed"] is False
    assert audit["goal_complete"] is False


def test_completion_audit_rejects_anchor_provenance_mismatch() -> None:
    inputs = valid_inputs()
    inputs["amortization"]["anchor_provenance"]["anchor_bridge_semantics"] = "legacy"
    audit = build_audit(**inputs)
    provenance = next(
        check for check in audit["checks"] if check["name"] == "parent_anchor_provenance"
    )
    assert provenance["passed"] is False
    assert audit["goal_complete"] is False


def test_completion_audit_rejects_missing_corrected_card_reroot() -> None:
    inputs = valid_inputs()
    inputs["amortization"]["search_reroot_audit"] = {
        "passed": False,
        "corrected_games": 128,
        "corrected_missing_required_card_reroots": [
            {"source_game": "game-9", "step": 150}
        ],
    }
    audit = build_audit(**inputs)
    reroot = next(
        check
        for check in audit["checks"]
        if check["name"] == "independent_card_search_rerooting"
    )
    assert reroot["passed"] is False
    assert audit["goal_complete"] is False


def test_completion_audit_rejects_legacy_headline_semantics() -> None:
    inputs = valid_inputs()
    inputs["convergence"].pop("bridge_semantics")
    audit = build_audit(**inputs)
    convergence = next(
        check for check in audit["checks"] if check["name"] == "search_convergence_panel"
    )
    assert convergence["passed"] is False
    assert audit["goal_complete"] is False
