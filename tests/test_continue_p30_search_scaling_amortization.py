from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import scripts.continue_p30_search_scaling_amortization as continuation

from scripts.continue_p30_search_scaling_amortization import (
    COLLECTIONS,
    collector_command,
    distillation_command,
    validate_anchor_provenance,
    validate_committed_replay_index,
    validate_collection,
    validate_planner_semantics_migration,
    validate_search_evidence,
    validate_teacher_coverage,
)


def teacher_summary(*, accepted: int, us: int, ussr: int, source_games: int) -> dict:
    payload = {
        "accepted": accepted,
        "side:us": us,
        "side:ussr": ussr,
        "source_games": source_games,
    }
    for side, total in (("us", us), ("ussr", ussr)):
        quotient, remainder = divmod(total, 3)
        for index, era in enumerate(("early", "mid", "late")):
            count = quotient + int(index < remainder)
            payload[f"side_era:{side}:{era}"] = count
            payload[f"era:{era}"] = payload.get(f"era:{era}", 0) + count
    return payload


def test_validate_collection_requires_complete_cutoff_free_games(tmp_path) -> None:
    (tmp_path / "search_replay").mkdir()
    (tmp_path / "anchor_replay").mkdir()
    (tmp_path / "search_replay/manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "anchor_replay/manifest.json").write_text("{}", encoding="utf-8")
    summary = tmp_path / "collection_summary.json"
    summary.write_text(
        json.dumps({"games_completed": 31, "discarded_games": 1, "worker_errors": 0}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="31/32"):
        validate_collection(summary)


def test_validate_collection_accepts_durable_manifests(tmp_path) -> None:
    (tmp_path / "search_replay").mkdir()
    (tmp_path / "anchor_replay").mkdir()
    (tmp_path / "search_replay/manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "anchor_replay/manifest.json").write_text("{}", encoding="utf-8")
    summary = tmp_path / "collection_summary.json"
    payload = {"games_completed": 32, "discarded_games": 0, "worker_errors": 0}
    summary.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_collection(summary) == payload


def test_validate_teacher_coverage_aggregates_parallel_shards() -> None:
    shards = [
        teacher_summary(accepted=400, us=180, ussr=220, source_games=32),
        teacher_summary(accepted=420, us=220, ussr=200, source_games=32),
        teacher_summary(accepted=410, us=205, ussr=205, source_games=32),
    ]
    assert validate_teacher_coverage(shards) == {
        "accepted": 1230,
        "us": 605,
        "ussr": 625,
        "source_games": 96,
        "by_era": {"early": 413, "mid": 409, "late": 408},
        "by_side_era": {
            "us:early": 203,
            "us:mid": 201,
            "us:late": 201,
            "ussr:early": 210,
            "ussr:mid": 208,
            "ussr:late": 207,
        },
    }


def test_validate_teacher_coverage_rejects_side_collapse() -> None:
    shards = [teacher_summary(accepted=1200, us=900, ussr=300, source_games=96)]
    with pytest.raises(RuntimeError, match="coverage/balance"):
        validate_teacher_coverage(shards)


def test_validate_teacher_coverage_rejects_proportional_side_collapse() -> None:
    shards = [
        teacher_summary(accepted=10_000, us=9_500, ussr=500, source_games=96)
    ]
    with pytest.raises(RuntimeError, match="coverage/balance"):
        validate_teacher_coverage(shards)


def test_validate_teacher_coverage_rejects_narrow_game_coverage() -> None:
    shards = [
        teacher_summary(accepted=2000, us=1000, ussr=1000, source_games=32)
    ]
    with pytest.raises(RuntimeError, match="coverage/balance"):
        validate_teacher_coverage(shards)


def test_validate_teacher_coverage_rejects_late_war_collapse() -> None:
    summary = teacher_summary(accepted=1800, us=900, ussr=900, source_games=96)
    summary.update(
        {
            "era:early": 900,
            "era:mid": 860,
            "era:late": 40,
            "side_era:us:early": 450,
            "side_era:us:mid": 430,
            "side_era:us:late": 20,
            "side_era:ussr:early": 450,
            "side_era:ussr:mid": 430,
            "side_era:ussr:late": 20,
        }
    )
    with pytest.raises(RuntimeError, match="era_coverage"):
        validate_teacher_coverage([summary])


def test_validate_planner_semantics_migration_checks_filter_audit(tmp_path) -> None:
    audit = tmp_path / "filter-audit.json"
    audit.write_text(
        json.dumps(
            {
                "input": 100,
                "accepted": 20,
                "rejected:cross_macro_hierarchy": 70,
                "rejected:requires_horizon_validation": 10,
                "side:us": 9,
                "side:ussr": 11,
            }
        ),
        encoding="utf-8",
    )
    migration = tmp_path / "migration.json"
    payload = {
        "legacy_completed_games": [f"game-{index}" for index in range(8)],
        "legacy_game_count": 8,
        "remaining_corrected_game_count": 120,
        "legacy_filter_audit": {
            "path": audit.name,
            "input": 100,
            "accepted": 20,
            "cross_macro_hierarchy_rejected": 70,
            "scoring_horizon_rejected": 10,
            "side_us": 9,
            "side_ussr": 11,
        },
    }
    migration.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_planner_semantics_migration(migration, root=tmp_path) == payload


def test_validate_clean_planner_semantics_migration_needs_no_legacy_audit(tmp_path) -> None:
    migration = tmp_path / "migration.json"
    payload = {
        "legacy_completed_games": [],
        "legacy_game_count": 0,
        "remaining_corrected_game_count": 128,
        "legacy_filter_audit": {},
    }
    migration.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_planner_semantics_migration(migration, root=tmp_path) == payload


def test_supervised_resume_reuses_exact_collection_identity() -> None:
    command = collector_command(COLLECTIONS[1], 98_320_000)
    assert command[command.index("--output-dir") + 1] == str(COLLECTIONS[1])
    assert command[command.index("--seed-base") + 1] == "98320000"
    assert 8 <= int(command[command.index("--workers") + 1]) <= 32
    assert command[command.index("--meaningful-depth") + 1] == "4"
    assert command[command.index("--adaptive-deep-depth") + 1] == "6"
    assert command[command.index("--max-expanded-nodes") + 1] == "131072"


def test_existing_collector_tmux_session_is_reused(monkeypatch) -> None:
    collection = COLLECTIONS[0]
    monkeypatch.setattr(continuation, "tmux_session_exists", lambda _session: True)
    monkeypatch.setattr(
        continuation.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("must not launch a duplicate tmux session"),
    )

    session, command = continuation.launch_collector_tmux(
        collection, continuation.COLLECTION_SEEDS[collection]
    )

    assert session == continuation.COLLECTOR_TMUX_SESSIONS[collection]
    assert command == collector_command(collection, continuation.COLLECTION_SEEDS[collection])


def test_available_evidence_gpus_filters_busy_or_full_devices(monkeypatch) -> None:
    result = SimpleNamespace(
        returncode=0,
        stdout="\n".join(
            (
                "4, 12, 70000",
                "5, 80, 70000",
                "6, 10, 8000",
                "7, 20, 60000",
            )
        ),
    )
    monkeypatch.setattr(continuation.subprocess, "run", lambda *args, **kwargs: result)

    assert continuation._available_evidence_gpus() == [4, 7]


def test_capacity_aware_evidence_command_uses_two_gpu_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        continuation,
        "_available_evidence_gpus",
        lambda: [5, 4, 7],
    )
    command = ["python", "evidence.py", "--gpus", "4,5,6,7", "--roots", "24"]

    rewritten = continuation._capacity_aware_evidence_command(command)

    assert rewritten[rewritten.index("--gpus") + 1] == "5,4"
    assert command[command.index("--gpus") + 1] == "4,5,6,7"


def test_capacity_aware_collector_resume_scales_to_remaining_games(
    tmp_path, monkeypatch
) -> None:
    collection = tmp_path / "collection"
    completed = collection / "completed_games"
    completed.mkdir(parents=True)
    for index in range(31):
        (completed / f"game-{index:06d}.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        continuation,
        "_available_evidence_gpus",
        lambda **_kwargs: [6, 5, 7],
    )
    command = ["python", "collector.py", "--gpus", "4,5,6,7", "--workers", "14"]

    rewritten = continuation._capacity_aware_collector_command(command, collection)

    assert rewritten[rewritten.index("--gpus") + 1] == "6"
    assert rewritten[rewritten.index("--workers") + 1] == "1"


def test_run_once_skips_capacity_wait_for_completed_evidence(
    tmp_path, monkeypatch
) -> None:
    done = tmp_path / "done.json"
    done.write_text("{}\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        continuation,
        "_capacity_aware_evidence_command",
        lambda command: pytest.fail("completed evidence must not wait for GPU capacity"),
    )
    monkeypatch.setattr(
        continuation,
        "_legacy_run_once",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    command = ["python", "evidence.py", "--gpus", "4,5,6,7"]
    continuation.run_once(
        command,
        done_path=done,
        log_path=tmp_path / "run.log",
        stage="headlinefix_search_convergence_and_operator_gate",
    )

    assert calls == [
        (
            command,
            {
                "done_path": done,
                "log_path": tmp_path / "run.log",
                "stage": "headlinefix_search_convergence_and_operator_gate",
            },
        )
    ]


def test_worker_override_is_applied_without_changing_search_identity(tmp_path) -> None:
    collection = tmp_path / "collection"
    collection.mkdir()
    continuation.COLLECTION_WORKERS[collection] = 16
    try:
        (collection / ".collector_workers.json").write_text(
            '{"workers": 8}\n', encoding="utf-8"
        )
        command = continuation.collector_command(collection, 98_340_000)
        assert command[command.index("--workers") + 1] == "8"
        assert continuation.collection_worker_count(collection) == 8
        assert command[command.index("--meaningful-depth") + 1] == "4"
    finally:
        continuation.COLLECTION_WORKERS.pop(collection, None)


def test_scaled_collection_topology_keeps_four_disjoint_shards() -> None:
    assert len(COLLECTIONS) == 4
    assert len({str(path) for path in COLLECTIONS}) == 4
    commands = [
        collector_command(collection, seed)
        for collection, seed in zip(
            COLLECTIONS,
            (98_300_000, 98_320_000, 98_340_000, 98_360_000),
            strict=True,
        )
    ]
    assert [command[command.index("--seed-base") + 1] for command in commands] == [
        "98300000",
        "98320000",
        "98340000",
        "98360000",
    ]
    assert all(command[command.index("--games") + 1] == "32" for command in commands)
    worker_counts = [
        int(command[command.index("--workers") + 1]) for command in commands
    ]
    # Runtime memory throttling persists per-shard overrides in the live run
    # directories, so this topology test must accept the supported elastic
    # range rather than assume an idle-host worker count.
    assert all(8 <= workers <= 32 for workers in worker_counts)
    assert sum(worker_counts) >= 32


def test_distillation_uses_only_current_semantics_parent_anchors() -> None:
    teacher_dirs = [continuation.CAMPAIGN / f"teacher-{index}" for index in range(4)]
    command = distillation_command(teacher_dirs)
    anchor_paths = [
        command[index + 1]
        for index, argument in enumerate(command[:-1])
        if argument == "--anchor-replay"
    ]

    assert "--anchor-from-all-safe-replay" in command
    assert anchor_paths == [str(collection / "anchor_replay/manifest.json") for collection in COLLECTIONS]
    assert len(anchor_paths) == len(COLLECTIONS)
    assert command[command.index("--lambda-anchor") + 1] == "4.0"
    assert command[command.index("--anchor-replay-batch-size") + 1] == "512"
    assert command[command.index("--min-search-examples-per-era") + 1] == "16"
    assert command[command.index("--min-search-examples-per-side-era") + 1] == "4"


def test_record_training_batch_coverage_persists_observed_minima(
    tmp_path, monkeypatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    rows = []
    for late_count in (18, 16):
        row = {
            f"batch_era_{era}": late_count if era == "late" else 64
            for era in ("early", "mid", "late")
        }
        row.update(
            {
                f"batch_side_era_{side}_{era}": 4
                for side in ("us", "ussr")
                for era in ("early", "mid", "late")
            }
        )
        rows.append(row)
    (campaign / "distillation_report.json").write_text(
        json.dumps(
            {
                "min_search_examples_per_era": 16,
                "min_search_examples_per_side_era": 4,
                "history": rows,
            }
        ),
        encoding="utf-8",
    )
    (campaign / "amortization_summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(continuation, "CAMPAIGN", campaign)

    coverage = continuation.record_training_batch_coverage()

    assert coverage["passed"] is True
    assert coverage["minimum_observed_by_era"]["late"] == 16
    persisted = json.loads((campaign / "amortization_summary.json").read_text())
    assert persisted["training_batch_coverage"] == coverage


def test_anchor_provenance_matches_composed_actor_parent(tmp_path) -> None:
    campaign = tmp_path / "campaign"
    parent = campaign / "checkpoints/composed"
    parent.mkdir(parents=True)
    actor = tmp_path / "actor/checkpoint"
    actor.mkdir(parents=True)
    (campaign / "composition_manifest.json").write_text(
        json.dumps(
            {
                "verified": True,
                "checkpoint": str(parent),
                "actor_checkpoint": str(actor),
            }
        ),
        encoding="utf-8",
    )
    collections = tuple(tmp_path / f"collection-{index}" for index in range(2))
    for collection in collections:
        collection.mkdir()
        (collection / "collection_identity.json").write_text(
            json.dumps({"checkpoint": str(parent)}), encoding="utf-8"
        )
    result = validate_anchor_provenance(parent, collections)

    assert result["composition_verified"] is True
    assert result["anchor_bridge_semantics"] == "headline_priority_fixed_v1"
    assert result["legacy_anchor_replays_excluded"] is True


def test_anchor_provenance_rejects_stale_current_parent(tmp_path) -> None:
    campaign = tmp_path / "campaign"
    parent = campaign / "checkpoints/composed"
    parent.mkdir(parents=True)
    actor = tmp_path / "actor/checkpoint"
    actor.mkdir(parents=True)
    (campaign / "composition_manifest.json").write_text(
        json.dumps(
            {"verified": True, "checkpoint": str(parent), "actor_checkpoint": str(actor)}
        ),
        encoding="utf-8",
    )
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "collection_identity.json").write_text(
        json.dumps({"checkpoint": str(parent)}), encoding="utf-8"
    )
    (collection / "collection_identity.json").write_text(
        json.dumps({"checkpoint": str(tmp_path / "stale")}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="current anchor source mismatch"):
        validate_anchor_provenance(parent, (collection,))


def valid_search_evidence() -> dict:
    return {
        "bridge_semantics": "headline_priority_fixed_v1",
        "root_panel_audit": {"passed": True},
        "convergence": {
            "positions": 24,
            "teacher_examples": 16,
            "status_by_side": {
                "BUDGET_CONVERGENCE:us": 5,
                "HORIZON_CONVERGENCE:us": 3,
                "BUDGET_CONVERGENCE:ussr": 4,
                "HORIZON_CONVERGENCE:ussr": 4,
            },
        },
        "operator": {
            "games": 100,
            "score": 0.62,
            "paired_bootstrap_ci95_low": 0.53,
            "by_candidate_side": {"us": {"score": 0.60}, "ussr": {"score": 0.64}},
            "atomic_horizon_cutoffs": 0,
            "node_budget_cutoffs": 0,
        },
    }


def test_search_evidence_must_support_positive_d4_scaling() -> None:
    result = validate_search_evidence(valid_search_evidence())
    assert result["passed"] is True
    assert result["converged_by_side"] == {"us": 8, "ussr": 8}


def test_search_evidence_rejects_unsupported_operator() -> None:
    evidence = valid_search_evidence()
    evidence["operator"]["paired_bootstrap_ci95_low"] = 0.48
    with pytest.raises(RuntimeError, match="d4_strength_not_statistically_supported"):
        validate_search_evidence(evidence)


def write_metadata_replay(path, rows) -> None:
    path.mkdir(parents=True)
    shard = path / "search-replay-000000.npz"
    np.savez_compressed(
        shard,
        metadata_json=np.asarray([json.dumps(row) for row in rows]),
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "version": 5,
                "format": "compressed_npz_shards",
                "examples": len(rows),
                "shards": [{"path": shard.name, "examples": len(rows)}],
            }
        ),
        encoding="utf-8",
    )


def test_validate_committed_replay_index_checks_terminal_targets(tmp_path) -> None:
    completed = tmp_path / "completed_games"
    completed.mkdir()
    record = {
        "source_game": "game-1",
        "winner": "us",
        "terminal_reason": "final_scoring",
        "discard_reason": None,
        "partial_search_labels_rejected": 0,
        "committed_examples": 2,
        "anchor_committed_examples": 2,
    }
    (completed / "game-000000.json").write_text(json.dumps(record), encoding="utf-8")
    root = {
        "source_game": "game-1",
        "source_step": 10,
        "side": "us",
        "hierarchy_index": 0,
        "complete_game": True,
        "game_winner": "us",
        "terminal_reason": "final_scoring",
        "atomic_horizon_cutoffs": 0,
        "node_budget_cutoffs": 0,
        "root_value_target": 1.0,
        "convergence_status": "BUDGET_CONVERGENCE",
        "evidence_gate_rejected": False,
    }
    child = {
        **root,
        "source_step": 11,
        "side": "ussr",
        "hierarchy_index": 1,
        "root_value_target": None,
    }
    write_metadata_replay(tmp_path / "search_replay", [root, child])
    write_metadata_replay(tmp_path / "anchor_replay", [root, child])
    summary = {
        "games_completed": 1,
        "committed_examples": 2,
        "anchor_committed_examples": 2,
    }
    result = validate_committed_replay_index(tmp_path, summary)
    assert result["games"] == 1
    assert result["teacher"]["rows"] == 2
    assert result["winners"] == {"us": 1}


def test_validate_committed_replay_index_rejects_orphan_label(tmp_path) -> None:
    completed = tmp_path / "completed_games"
    completed.mkdir()
    record = {
        "source_game": "game-1",
        "winner": "us",
        "terminal_reason": "final_scoring",
        "discard_reason": None,
        "partial_search_labels_rejected": 0,
        "committed_examples": 1,
        "anchor_committed_examples": 1,
    }
    (completed / "game-000000.json").write_text(json.dumps(record), encoding="utf-8")
    orphan = {
        "source_game": "game-orphan",
        "source_step": 10,
        "side": "us",
        "hierarchy_index": 0,
        "complete_game": True,
        "game_winner": "us",
        "terminal_reason": "final_scoring",
        "atomic_horizon_cutoffs": 0,
        "node_budget_cutoffs": 0,
        "root_value_target": 1.0,
        "convergence_status": "BUDGET_CONVERGENCE",
        "evidence_gate_rejected": False,
    }
    write_metadata_replay(tmp_path / "search_replay", [orphan])
    write_metadata_replay(tmp_path / "anchor_replay", [orphan])
    with pytest.raises(RuntimeError, match="orphan replay source"):
        validate_committed_replay_index(
            tmp_path,
            {"games_completed": 1, "committed_examples": 1, "anchor_committed_examples": 1},
        )
