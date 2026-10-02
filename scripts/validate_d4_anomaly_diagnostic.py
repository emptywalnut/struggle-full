#!/usr/bin/env python3
"""Validate the frozen D4 anomaly diagnostic without rerunning expensive search."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.frozen_diagnostic import observation_digest
from struggle_ai.mcts import root_side, validate_information_safe_observation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "runs/diagnostics/d4-anomaly-probes-accepted-parent-20260818"
DEFAULT_REPORT = ROOT / "docs/d4_anomaly_diagnostic_report_20260818.md"
EXPECTED_CHECKPOINT = (
    ROOT
    / "runs/puct_selfplay/cycle0-student-balanced2games-100updates-gpu4-20260812/checkpoints/distilled"
)
BASE_MATRIX = {(2, 32, 0), (2, 32, 1), (4, 128, 0), (4, 128, 1), (6, 512, 0), (6, 512, 1)}
D8_MATRIX = {(8, 2048, 0)}


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    primary: str
    decision: str
    decision_fixture: str
    d8: str | None = None
    support: tuple[str, ...] = ()


SPECS = (
    ArtifactSpec(
        name="ussr_uk_overcontrol",
        primary="shards/ussr_uk_overcontrol/ussr_uk_overcontrol.json",
        decision="detail_shards/ussr_uk_overcontrol/ussr_uk_overcontrol.json",
        decision_fixture="detail_fixture",
        support=(
            "depth_budget_ablation/ussr_uk_overcontrol/ussr_uk_overcontrol.json",
            "belief_sensitivity/k64_ussr_uk_overcontrol/ussr_uk_overcontrol.json",
        ),
    ),
    ArtifactSpec(
        name="us_italy_overcontrol",
        primary="shards/us_italy_overcontrol/us_italy_overcontrol.json",
        decision="detail_shards/us_italy_overcontrol/us_italy_overcontrol.json",
        decision_fixture="detail_fixture",
        support=("depth_budget_ablation/us_italy_overcontrol/us_italy_overcontrol.json",),
    ),
    ArtifactSpec(
        name="ussr_mexico_realignment",
        primary="shards/ussr_mexico_realignment/ussr_mexico_realignment.json",
        decision="target_shards/ussr_mexico_realignment/ussr_mexico_realignment.json",
        decision_fixture="counterfactuals.target_fixture",
        d8="target_shards_d8/ussr_mexico_realignment/ussr_mexico_realignment.json",
        support=(
            "detail_shards_v2/ussr_mexico_realignment/ussr_mexico_realignment.json",
            "detail_shards_v3_d8/ussr_mexico_realignment/ussr_mexico_realignment.json",
        ),
    ),
    ArtifactSpec(
        name="us_venezuela_realignment",
        primary="shards/us_venezuela_realignment/us_venezuela_realignment.json",
        decision="target_shards/us_venezuela_realignment/us_venezuela_realignment.json",
        decision_fixture="counterfactuals.target_fixture",
        d8="target_shards_d8/us_venezuela_realignment/us_venezuela_realignment.json",
        support=("detail_shards/us_venezuela_realignment/us_venezuela_realignment.json",),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def env_config() -> dict[str, Any]:
    return {
        "force_setup_heuristic": False,
        "persistent_random_prob": 0.0,
        "heuristic_override_prob": 0.0,
        "scripted_side_prob": 0.0,
        "defcon_suicide_mode": "hard_filter",
        "terminal_reward_scale": 1.0,
        "reward_shaping_scale": 0.0,
        "nuke_death_penalty": 0.0,
        "scoring_card_held_penalty": 0.0,
        "max_episode_steps": 1200,
        "search_mode": True,
        "log_games_dir": None,
        "metrics_dir": None,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict[str, Any], dotted_key: str) -> dict[str, Any]:
    value: Any = data
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise AssertionError(f"missing {dotted_key}")
        value = value[key]
    if not isinstance(value, dict):
        raise AssertionError(f"{dotted_key} is not an object")
    return value


def digest_snapshot(snapshot: dict[str, Any]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.blake2b(payload.encode("ascii"), digest_size=16).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_key(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def validate_source(fixture: dict[str, Any]) -> dict[str, Any]:
    source = Path(str(fixture["source_log"]))
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 1 or records[0].get("kind") != "terminal":
        raise AssertionError(f"source must contain exactly one terminal game: {source}")
    game = records[0]
    index = int(fixture["root_action_index"])
    record = game["actions"][index]
    if int(game["seed"]) != int(fixture["source_seed"]):
        raise AssertionError(f"seed mismatch for {fixture['name']}")
    if record["before"] != fixture["before"]:
        raise AssertionError(f"source state mismatch for {fixture['name']}")
    if action_key(record["action"]) != action_key(fixture["logged_action"]):
        raise AssertionError(f"source action mismatch for {fixture['name']}")
    if int(record["action_index"]) != int(fixture["logged_action_index"]):
        raise AssertionError(f"source legal-action index mismatch for {fixture['name']}")
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "seed": int(game["seed"]),
        "action_index": index,
    }


def validate_source_decision(source_path: Path, fixture: dict[str, Any], *, expected_name: str) -> dict[str, Any]:
    records = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line]
    game = records[0]
    index = int(fixture["action_index"])
    record = game["actions"][index]
    if record["before"] != fixture["before"]:
        raise AssertionError(f"source decision state mismatch for {expected_name}")
    if action_key(record["action"]) != action_key(fixture["logged_action"]):
        raise AssertionError(f"source decision action mismatch for {expected_name}")
    logged_index = int(record["action_index"])
    legal_actions = fixture["legal_actions"]
    if logged_index >= len(legal_actions) or action_key(legal_actions[logged_index]) != action_key(record["action"]):
        raise AssertionError(f"source decision legal ordering mismatch for {expected_name}")
    return {"action_index": index, "legal_action_index": logged_index}


def validate_fixture(
    env: TwilightStruggleEnv,
    fixture: dict[str, Any],
    *,
    expected_name: str,
) -> dict[str, Any]:
    snapshot_path = Path(str(fixture["snapshot"]))
    if not snapshot_path.is_file():
        raise AssertionError(f"missing snapshot: {snapshot_path}")
    with gzip.open(snapshot_path, "rt", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    digest = digest_snapshot(snapshot)
    if digest != fixture["snapshot_digest"]:
        raise AssertionError(f"snapshot digest mismatch for {expected_name}: {snapshot_path}")

    obs = env.restore(snapshot)
    side = str(fixture["before"]["side"])
    validate_information_safe_observation(obs, side)
    if root_side(env.last_obs) != side:
        raise AssertionError(f"side mismatch after restore for {expected_name}")
    actual_digest = observation_digest(obs)
    if actual_digest != fixture["observation_digest"]:
        raise AssertionError(f"observation digest mismatch for {expected_name}: {snapshot_path}")
    if env.legal_actions != fixture["legal_actions"]:
        raise AssertionError(f"legal actions mismatch after restore for {expected_name}")
    logged = fixture["logged_action"]
    if action_key(logged) not in {action_key(action) for action in env.legal_actions}:
        raise AssertionError(f"logged action is not legal for {expected_name}")

    opponent_key = "ussr_hand" if side == "us" else "us_hand"
    opponent_visible = float(np.asarray(obs.get(opponent_key, [])).sum())
    if abs(opponent_visible) > 1e-6:
        raise AssertionError(f"opponent hand leaked for {expected_name}")
    return {
        "snapshot": str(snapshot_path.resolve()),
        "snapshot_sha256": sha256_file(snapshot_path),
        "snapshot_digest": digest,
        "observation_digest": actual_digest,
        "side": side,
        "legal_action_count": len(env.legal_actions),
        "opponent_hand_visible": opponent_visible,
    }


SEARCH_KEYS = {
    "depth",
    "budget",
    "belief_particles",
    "seed",
    "choice",
    "choice_Q",
    "root_Q_gap",
    "root_table",
    "root_expanded_actions",
    "root_unvisited_actions",
    "prior_order",
    "child_progressive_widening",
    "exact_chance_nodes",
    "sampled_large_chance_nodes",
    "atomic_horizon_cutoffs",
    "node_budget_cutoffs",
    "max_meaningful_depth_reached",
    "latency_seconds",
    "simulations_per_second",
    "repeat",
}


def search_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["depth"]), int(row["budget"]), int(row["repeat"])


def validate_search_rows(rows: Iterable[dict[str, Any]], expected: set[tuple[int, int, int]]) -> dict[str, Any]:
    rows = list(rows)
    found = {search_key(row) for row in rows}
    if found != expected:
        raise AssertionError(f"search matrix mismatch: expected={sorted(expected)} found={sorted(found)}")
    seeds: dict[tuple[int, int], set[int]] = {}
    for row in rows:
        missing = SEARCH_KEYS - set(row)
        if missing:
            raise AssertionError(f"search telemetry missing {sorted(missing)} at {search_key(row)}")
        if int(row["belief_particles"]) != 32:
            raise AssertionError(f"expected K=32 at {search_key(row)}")
        if not row["root_table"] or not row["prior_order"]:
            raise AssertionError(f"empty root telemetry at {search_key(row)}")
        if len(row["root_table"]) != int(row["root_legal_count"]):
            raise AssertionError(f"root table size mismatch at {search_key(row)}")
        if any(not {"action", "action_index", "N", "Q", "P"} <= set(item) for item in row["root_table"]):
            raise AssertionError(f"incomplete N/Q/P root row at {search_key(row)}")
        if any("first_visit_simulation" not in item for item in row["prior_order"]):
            raise AssertionError(f"missing progressive-widening order at {search_key(row)}")
        if float(row["latency_seconds"]) <= 0 or float(row["simulations_per_second"]) <= 0:
            raise AssertionError(f"invalid timing at {search_key(row)}")
        key = int(row["depth"]), int(row["budget"])
        seeds.setdefault(key, set()).add(int(row["seed"]))
    for depth, budget, repeat in expected:
        repeats = sum(1 for item in expected if item[:2] == (depth, budget))
        if repeats > 1 and len(seeds[(depth, budget)]) != repeats:
            raise AssertionError(f"search repeats reused a seed for D{depth}/{budget}")
    return {
        "runs": len(rows),
        "matrix": [list(item) for item in sorted(found)],
        "cutoffs": int(sum(int(row["atomic_horizon_cutoffs"]) + int(row["node_budget_cutoffs"]) for row in rows)),
        "exact_chance_nodes": int(sum(int(row["exact_chance_nodes"]) for row in rows)),
        "wall_seconds": float(sum(float(row["latency_seconds"]) for row in rows)),
    }


def validate_actor(result: dict[str, Any], *, realignment_target: bool) -> dict[str, Any]:
    raw = result["raw_actor"]
    required = {"entropy", "normalized_entropy", "logged_action_rank", "logged_action_probability", "top_actions"}
    if not required <= set(raw) or not raw["top_actions"]:
        raise AssertionError(f"incomplete raw actor report for {result['spec']['name']}")
    for row in raw["top_actions"]:
        if not {"rank", "action", "probability", "logit"} <= set(row):
            raise AssertionError(f"incomplete actor ranking for {result['spec']['name']}")
    for row in result["logged_hierarchical_path"]:
        if not {"action", "rank", "probability", "logit", "legal_count"} <= set(row):
            raise AssertionError(f"incomplete hierarchical actor path for {result['spec']['name']}")

    if realignment_target:
        targets = result["counterfactuals"]["realignment_targets"]
        if int(targets["sampled_large_chance_nodes"]) != 0:
            raise AssertionError(f"realignment expectation sampled chance for {result['spec']['name']}")
        if int(targets["exact_chance_nodes"]) != len(targets["targets"]):
            raise AssertionError(f"not every realignment target was exactly enumerated for {result['spec']['name']}")
        for target in targets["targets"]:
            if target["outcomes"] is not None:
                probability = sum(float(item["probability"]) for item in target["outcomes"])
                if not math.isclose(probability, 1.0, abs_tol=1e-9):
                    raise AssertionError(f"chance probabilities do not sum to one for {target['action']}")
        return {"exact_targets": len(targets["targets"]), "sampled_chance_nodes": 0}

    counterfactuals = result["counterfactuals"]
    if not counterfactuals["first_placement_values"] or not counterfactuals["suspicious_country_marginal_curve"]:
        raise AssertionError(f"missing influence counterfactuals for {result['spec']['name']}")
    complete_key = "critic_complete_allocation_probes"
    if not counterfactuals.get(complete_key):
        raise AssertionError(f"missing complete allocation values for {result['spec']['name']}")
    return {
        "first_placement_values": len(counterfactuals["first_placement_values"]),
        "marginal_curve_points": len(counterfactuals["suspicious_country_marginal_curve"]),
        "complete_allocation_probes": len(counterfactuals[complete_key]),
    }


def validate_report(report: Path) -> dict[str, Any]:
    text = report.read_text(encoding="utf-8")
    required = [
        "USSR UK Overcontrol",
        "US Italy Overcontrol",
        "USSR Mexico Realignment",
        "US Venezuela Realignment",
        "D2/32",
        "D4/128",
        "D6/512",
        "D8/2048",
        "ACTOR",
        "SEARCH_COVERAGE",
        "HORIZON",
        "BELIEF",
        "Recommended Next Changes",
    ]
    lowered = text.casefold()
    missing = [token for token in required if token.casefold() not in lowered]
    if missing:
        raise AssertionError(f"report is missing required sections/tokens: {missing}")
    return {"path": str(report.resolve()), "sha256": sha256_file(report), "required_tokens": len(required)}


def validate(run_dir: Path, report: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    report = report.resolve()
    env = TwilightStruggleEnv(env_config())
    roots: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}
    try:
        for spec in SPECS:
            primary_path = run_dir / spec.primary
            decision_path = run_dir / spec.decision
            primary = load_json(primary_path)
            decision = load_json(decision_path)
            if primary["spec"]["name"] != spec.name or decision["spec"]["name"] != spec.name:
                raise AssertionError(f"artifact identity mismatch for {spec.name}")
            if Path(primary["checkpoint"]).resolve() != EXPECTED_CHECKPOINT.resolve():
                raise AssertionError(f"checkpoint mismatch for {spec.name}")

            primary_fixture = primary["fixture"]
            source = validate_source(primary_fixture)
            primary_restore = validate_fixture(env, primary_fixture, expected_name=f"{spec.name}:primary")
            decision_fixture = nested(decision, spec.decision_fixture)
            decision_source = validate_source_decision(
                Path(source["path"]),
                decision_fixture,
                expected_name=f"{spec.name}:decision",
            )
            decision_restore = validate_fixture(env, decision_fixture, expected_name=f"{spec.name}:decision")
            primary_search = validate_search_rows(primary["search"], BASE_MATRIX | D8_MATRIX)

            decision_rows = list(decision["search"])
            decision_base_rows = [row for row in decision_rows if search_key(row) in BASE_MATRIX]
            decision_search = validate_search_rows(decision_base_rows, BASE_MATRIX)
            d8_data = decision
            if spec.d8 is not None:
                d8_data = load_json(run_dir / spec.d8)
            d8_rows = [row for row in d8_data["search"] if search_key(row) in D8_MATRIX]
            decision_d8 = validate_search_rows(d8_rows, D8_MATRIX)
            actor = validate_actor(decision, realignment_target=spec.decision_fixture.startswith("counterfactuals"))

            referenced = [primary_path, decision_path]
            if spec.d8 is not None:
                referenced.append(run_dir / spec.d8)
            referenced.extend(run_dir / item for item in spec.support)
            for path in referenced:
                load_json(path)
                artifacts[str(path.relative_to(run_dir))] = sha256_file(path)

            roots.append(
                {
                    "root": spec.name,
                    "source": source,
                    "primary_restore": primary_restore,
                    "decision_source": decision_source,
                    "decision_restore": decision_restore,
                    "actor_counterfactual_evidence": actor,
                    "primary_search": primary_search,
                    "decision_search": decision_search,
                    "decision_d8": decision_d8,
                    "support_artifacts": list(spec.support),
                }
            )
    finally:
        env.close()

    return {
        "schema_version": 2,
        "all_passed": True,
        "frozen_checkpoint": str(EXPECTED_CHECKPOINT.resolve()),
        "training_updates": 0,
        "production_search_modified": False,
        "report": validate_report(report),
        "roots": roots,
        "artifact_sha256": dict(sorted(artifacts.items())),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else run_dir / "validation.json"
    result = validate(run_dir, args.report.expanduser().resolve())
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": True, "roots": len(result["roots"]), "output": str(output)}))


if __name__ == "__main__":
    main()
