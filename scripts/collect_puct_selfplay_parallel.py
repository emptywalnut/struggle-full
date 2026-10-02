#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import ctypes
import fcntl
import gc
import json
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import TextIO

import numpy as np

from struggle_ai.batched_inference import (
    STAT_COUNT,
    BatchedInferenceClient,
    inference_server_main,
    inference_stats_payload,
)
from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.mcts import (
    SelectivePuctConfig,
    SelectivePuctPlanner,
    meaningful_decision_kind,
)
from struggle_ai.runtime_limits import (
    configure_search_recursion_limit,
    required_search_recursion_limit as _required_worker_recursion_limit,
)
from struggle_ai.search_replay import (
    SEARCH_REPLAY_VERSION,
    DiskBackedCompleteGameReplayTransaction,
    SearchReplayWriter,
    examples_from_decision,
    iter_search_replay,
)
from struggle_ai.search_convergence import compare_q_rankings

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect complete D2/32 PUCT games with shared dual-GPU batched inference."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=87_000_000)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--inference-replicas-per-gpu",
        type=int,
        default=2,
        help="Resident model-serving processes per physical GPU sharing one dynamic request queue.",
    )
    parser.add_argument("--max-inference-batch-size", type=int, default=32)
    parser.add_argument("--max-inference-delay-ms", type=float, default=3.0)
    parser.add_argument("--inference-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--inference-cache-capacity", type=int, default=65_536)
    parser.add_argument("--client-cache-capacity", type=int, default=4_096)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=10.0)
    parser.add_argument("--initial-simulations", type=int, default=32)
    parser.add_argument("--medium-simulations", type=int, default=32)
    parser.add_argument("--max-simulations", type=int, default=32)
    parser.add_argument("--meaningful-depth", type=int, default=2)
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--max-atomic-steps", type=int, default=512)
    parser.add_argument("--max-expanded-nodes", type=int, default=512)
    parser.add_argument(
        "--max-search-worker-rss-mib",
        type=float,
        default=0.0,
        help=(
            "Fail closed on the current search tree when this worker RSS limit is reached; "
            "0 disables the operational memory guard."
        ),
    )
    parser.add_argument(
        "--max-search-seconds",
        type=float,
        default=0.0,
        help=(
            "Fail closed on the current search tree after this wall-clock duration; "
            "0 disables the operational time guard."
        ),
    )
    parser.add_argument("--child-initial", type=int, default=4)
    parser.add_argument("--child-max", type=int, default=24)
    parser.add_argument("--search-temperature", type=float, default=1.0)
    parser.add_argument("--behavior-temperature", type=float, default=0.75)
    parser.add_argument("--max-scoring-searches-per-side-turn", type=int, default=1)
    parser.add_argument("--force-search", action="store_true")
    parser.add_argument("--low-confidence-sample-rate", type=float, default=0.05)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--staging-shard-size", type=int, default=32)
    parser.add_argument(
        "--heap-trim-every-decisions",
        type=int,
        default=8,
        help="Collect cyclic garbage and return free glibc pages every N real decisions; 0 disables.",
    )
    parser.add_argument("--heartbeat-every-decisions", type=int, default=1)
    parser.add_argument("--deep-candidate-q-gap", type=float, default=0.03)
    parser.add_argument("--deep-candidate-max", type=int, default=512)
    parser.add_argument("--adaptive-search", action="store_true")
    parser.add_argument("--adaptive-always-compare", action="store_true")
    parser.add_argument(
        "--adaptive-on-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Escalate a root solely because base search flips the raw action. "
            "Disable for broad complete-game generation where D4/128 is already "
            "the behavior policy and deeper search is reserved for uncertain/high-stakes roots."
        ),
    )
    parser.add_argument("--adaptive-budget-simulations", type=int, default=512)
    parser.add_argument("--adaptive-deep-depth", type=int, default=6)
    parser.add_argument("--adaptive-deep-simulations", type=int, default=512)
    parser.add_argument(
        "--adaptive-deep-always-trigger",
        action="append",
        default=[],
        help=(
            "Trigger category that must be compared with the deeper configuration even "
            "when the two same-depth budgets agree. Repeat for multiple categories."
        ),
    )
    parser.add_argument("--adaptive-min-rank-correlation", type=float, default=0.70)
    parser.add_argument("--adaptive-meaningful-q-gap", type=float, default=0.01)
    parser.add_argument("--adaptive-escalation-q-gap", type=float, default=0.03)
    parser.add_argument("--adaptive-fail-closed-trigger", action="append", default=[])
    parser.add_argument(
        "--adaptive-fail-closed-unresolved-to-budget",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the same-depth budget action when deeper Q rankings remain unresolved.",
    )
    parser.add_argument("--adaptive-fail-closed-min-depth", type=int, default=0)
    parser.add_argument(
        "--adaptive-fail-closed-require-terminal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _acquire_collection_lock(output_dir: Path) -> TextIO:
    """Hold an advisory single-writer lock for the lifetime of this collector."""
    lock_path = output_dir / ".collector.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(
            f"collection output is already owned by another collector: {output_dir}; {owner}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}, sort_keys=True) + "\n")
    handle.flush()
    return handle


def _failed_processes(processes: list[mp.Process]) -> list[str]:
    return [
        f"{process.name} pid={process.pid} exit={process.exitcode}"
        for process in processes
        if process.exitcode is not None
    ]


def _trim_worker_heap() -> None:
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _resident_memory_mib() -> float:
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, IndexError):
        return -1.0


def _candidate_era(candidate: dict[str, Any]) -> str:
    turn = int(candidate.get("turn") or 0)
    return "early" if turn <= 3 else "mid" if turn <= 7 else "late"


def stratified_deep_candidates(
    records: list[dict[str, Any]], maximum: int
) -> list[dict[str, Any]]:
    """Cap adjudication roots without biasing toward early game indices."""
    buckets: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        for original in record.get("deep_candidates", []):
            candidate = dict(original)
            key = str(candidate.get("side") or ""), _candidate_era(candidate)
            source_game = str(candidate.get("source_game") or record.get("source_game") or "")
            buckets.setdefault(key, {}).setdefault(source_game, []).append(candidate)

    bucket_queues: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, by_game in buckets.items():
        for candidates in by_game.values():
            candidates.sort(
                key=lambda row: (
                    -abs(float(row.get("convergence_q_gap") or row.get("root_q_gap") or 0.0)),
                    float(row.get("latency_seconds") or 0.0),
                    int(row.get("action_index") or 0),
                )
            )
        game_names = sorted(by_game)
        queue_rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            added = False
            for game_name in game_names:
                candidates = by_game[game_name]
                if offset < len(candidates):
                    queue_rows.append(candidates[offset])
                    added = True
            if not added:
                break
            offset += 1
        bucket_queues[key] = queue_rows

    limit = max(0, int(maximum))
    if limit == 0:
        return []
    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in bucket_queues}
    ordered_keys = sorted(bucket_queues)
    while len(selected) < limit:
        added = False
        for key in ordered_keys:
            offset = offsets[key]
            if offset < len(bucket_queues[key]):
                selected.append(bucket_queues[key][offset])
                offsets[key] += 1
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    return selected


def _collection_identity(args: Any) -> dict[str, Any]:
    identity = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "games": int(args.games),
        "seed_base": int(args.seed_base),
        "meaningful_depth": int(args.meaningful_depth),
        "belief_particles": int(args.belief_particles),
        "initial_simulations": int(args.initial_simulations),
        "medium_simulations": int(args.medium_simulations),
        "max_simulations": int(args.max_simulations),
        "search_temperature": float(args.search_temperature),
        "behavior_temperature": float(args.behavior_temperature),
    }
    if bool(args.adaptive_search):
        identity["adaptive_search"] = {
            "always_compare": bool(args.adaptive_always_compare),
            "on_flip": bool(args.adaptive_on_flip),
            "budget_simulations": int(args.adaptive_budget_simulations),
            "deep_depth": int(args.adaptive_deep_depth),
            "deep_simulations": int(args.adaptive_deep_simulations),
            "min_rank_correlation": float(args.adaptive_min_rank_correlation),
            "meaningful_q_gap": float(args.adaptive_meaningful_q_gap),
            "escalation_q_gap": float(args.adaptive_escalation_q_gap),
            "fail_closed_unresolved_to_budget": bool(
                args.adaptive_fail_closed_unresolved_to_budget
            ),
        }
        deep_always_triggers = sorted(set(args.adaptive_deep_always_trigger))
        if deep_always_triggers:
            identity["adaptive_search"]["deep_always_triggers"] = deep_always_triggers
        fail_closed_triggers = sorted(set(args.adaptive_fail_closed_trigger))
        if fail_closed_triggers:
            identity["adaptive_search"]["fail_closed_triggers"] = fail_closed_triggers
            identity["adaptive_search"]["fail_closed_min_depth"] = int(
                args.adaptive_fail_closed_min_depth
            )
            identity["adaptive_search"]["fail_closed_require_terminal"] = bool(
                args.adaptive_fail_closed_require_terminal
            )
    if int(args.max_scoring_searches_per_side_turn) != 1:
        identity["max_scoring_searches_per_side_turn"] = int(
            args.max_scoring_searches_per_side_turn
        )
    return identity


def _load_completed_records(output_dir: Path, identity: dict[str, Any]) -> list[dict[str, Any]]:
    identity_path = output_dir / "collection_identity.json"
    if identity_path.exists():
        prior = json.loads(identity_path.read_text(encoding="utf-8"))
        if prior != identity:
            raise SystemExit(
                f"refusing to resume collection with mismatched identity: {identity_path}"
            )
    else:
        _atomic_json(identity_path, identity)
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in sorted((output_dir / "completed_games").glob("game-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        game_index = int(record["game_index"])
        expected_seed = int(identity["seed_base"]) + game_index
        if not 0 <= game_index < int(identity["games"]):
            raise SystemExit(f"out-of-range completed game record: {path}")
        if int(record["seed"]) != expected_seed:
            raise SystemExit(f"seed mismatch in completed game record: {path}")
        if game_index in seen:
            raise SystemExit(f"duplicate completed game index {game_index}")
        seen.add(game_index)
        records.append(record)
    return records


def _recover_replay_swap(replay_dir: Path) -> None:
    temporary = replay_dir.with_name(replay_dir.name + ".resume-prune.tmp")
    backup = replay_dir.with_name(replay_dir.name + ".resume-prune.backup")
    if backup.exists():
        if replay_dir.exists():
            shutil.rmtree(backup, ignore_errors=True)
        else:
            os.replace(backup, replay_dir)
    shutil.rmtree(temporary, ignore_errors=True)


def _prune_interrupted_worker_replay(
    output_dir: Path,
    completed_records: list[dict[str, Any]],
    *,
    shard_size: int,
) -> dict[str, int]:
    """Retain only replay labels covered by an atomic completed-game marker.

    A process can die after a terminal transaction starts flushing into a worker
    replay but before its completed-game JSON is atomically installed. On resume,
    those labels must not survive or mix with the replayed game. The completed
    record's worker identity, winner, reason, and expected counts form the durable
    commit index.
    """

    by_worker: dict[int, dict[str, dict[str, Any]]] = {}
    for record in completed_records:
        worker_id = int(record["worker_id"])
        source_game = str(record["source_game"])
        by_worker.setdefault(worker_id, {})[source_game] = record

    stats: Counter[str] = Counter()
    for worker_dir in sorted((output_dir / "workers").glob("worker-*")):
        try:
            worker_id = int(worker_dir.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        shutil.rmtree(worker_dir / "staging", ignore_errors=True)
        valid = by_worker.get(worker_id, {})
        for replay_name, count_key in (
            ("search_replay", "committed_examples"),
            ("anchor_replay", "anchor_committed_examples"),
        ):
            replay_dir = worker_dir / replay_name
            _recover_replay_swap(replay_dir)
            manifest = replay_dir / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                json.loads(manifest.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                expected_for_worker = sum(
                    int(record.get(count_key, 0)) for record in valid.values()
                )
                if expected_for_worker:
                    raise RuntimeError(
                        "corrupt replay manifest contains committed-game labels: "
                        f"worker={worker_id}, replay={replay_name}, "
                        f"expected={expected_for_worker}, manifest={manifest}"
                    ) from exc
                shutil.rmtree(replay_dir, ignore_errors=True)
                stats[f"{replay_name}_corrupt_uncommitted_pruned"] += 1
                continue
            temporary = replay_dir.with_name(replay_dir.name + ".resume-prune.tmp")
            backup = replay_dir.with_name(replay_dir.name + ".resume-prune.backup")
            writer = SearchReplayWriter(temporary, shard_size=max(1, int(shard_size)))
            retained = 0
            observed = 0
            for example in iter_search_replay((manifest,)):
                observed += 1
                metadata = example.metadata
                record = valid.get(str(metadata.source_game))
                if (
                    record is None
                    or not metadata.complete_game
                    or str(metadata.game_winner) != str(record.get("winner"))
                    or str(metadata.terminal_reason) != str(record.get("terminal_reason"))
                ):
                    continue
                writer.add(example)
                retained += 1
            writer.close()
            os.replace(replay_dir, backup)
            try:
                os.replace(temporary, replay_dir)
            except BaseException:
                os.replace(backup, replay_dir)
                raise
            shutil.rmtree(backup, ignore_errors=True)
            stats[f"{replay_name}_observed"] += observed
            stats[f"{replay_name}_retained"] += retained
            stats[f"{replay_name}_dropped"] += observed - retained

    expected_teacher = sum(int(record.get("committed_examples", 0)) for record in completed_records)
    expected_anchor = sum(
        int(record.get("anchor_committed_examples", 0)) for record in completed_records
    )
    retained_teacher = int(stats["search_replay_retained"])
    retained_anchor = int(stats["anchor_replay_retained"])
    if retained_teacher != expected_teacher or retained_anchor != expected_anchor:
        raise RuntimeError(
            "completed-game replay index mismatch after resume pruning: "
            f"teacher={retained_teacher}/{expected_teacher}, "
            f"anchor={retained_anchor}/{expected_anchor}"
        )
    stats["completed_games_indexed"] = len(completed_records)
    return dict(stats)


def env_config(args: Any, log_dir: Path | None, *, search_mode: bool = False) -> dict[str, Any]:
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
        "max_episode_steps": int(args.max_episode_steps),
        "search_mode": bool(search_mode),
        "log_games_dir": str(log_dir) if log_dir else None,
        "log_games_every": 1,
        "log_action_details": log_dir is not None,
        "metrics_dir": None,
    }


def search_config(
    args: Any, *, meaningful_depth: int | None = None, simulations: int | None = None
) -> SelectivePuctConfig:
    depth = int(args.meaningful_depth if meaningful_depth is None else meaningful_depth)
    initial = int(args.initial_simulations if simulations is None else simulations)
    medium = int(args.medium_simulations if simulations is None else simulations)
    maximum = int(args.max_simulations if simulations is None else simulations)
    return SelectivePuctConfig(
        initial_simulations=initial,
        medium_simulations=medium,
        max_simulations=maximum,
        belief_particles=int(args.belief_particles),
        meaningful_depth=depth,
        forced_response_depth=depth,
        max_atomic_steps=int(args.max_atomic_steps),
        max_expanded_nodes=int(args.max_expanded_nodes),
        max_process_rss_mib=float(args.max_search_worker_rss_mib),
        max_search_seconds=float(args.max_search_seconds),
        max_scoring_searches_per_side_turn=int(args.max_scoring_searches_per_side_turn),
        child_initial=int(args.child_initial),
        child_max=int(args.child_max),
        temperature=float(args.search_temperature),
        behavior_temperature=float(args.behavior_temperature),
        low_confidence_sample_rate=float(args.low_confidence_sample_rate),
    )


def _decision_comparison(left: Any, right: Any, args: Any) -> dict[str, Any]:
    if len(left.visits) != len(right.visits):
        return {"comparable": False, "stable": False, "reason": "legal_action_count_changed"}
    if any(
        int(getattr(decision, field, 0)) > 0
        for decision in (left, right)
        for field in ("atomic_horizon_cutoffs", "node_budget_cutoffs")
    ) or bool(getattr(left, "evidence_gate_rejected", False)) or bool(
        getattr(right, "evidence_gate_rejected", False)
    ):
        return {
            "comparable": False,
            "stable": False,
            "near_tie": False,
            "reason": "incomplete_search_evidence",
        }
    return compare_q_rankings(
        np.asarray(left.visits),
        np.asarray(left.q_root),
        np.asarray(right.visits),
        np.asarray(right.q_root),
        min_rank_correlation=float(args.adaptive_min_rank_correlation),
        meaningful_q_gap=float(args.adaptive_meaningful_q_gap),
    )


def adaptive_root_search_seed(seed_base: int, game_seed: int, action_index: int) -> int:
    """Return a stable per-root seed shared by every adaptive search configuration."""
    return int(seed_base) + int(game_seed) * 1_000_003 + int(action_index) * 10_007


def _reseed_planner_for_root(planner: SelectivePuctPlanner, seed: int) -> None:
    """Pair belief/chance streams without disturbing a committed macro continuation."""
    planner.seed = int(seed)
    planner.rng = np.random.default_rng(int(seed))
    planner.search_index = 0


def _should_adapt(decision: Any, legal_count: int, args: Any) -> bool:
    if bool(args.adaptive_always_compare):
        return True
    if abs(float(decision.root_q_gap)) <= float(args.adaptive_escalation_q_gap):
        return True
    if bool(args.adaptive_on_flip) and bool(decision.action_flipped):
        return True
    return bool(
        set(decision.triggers)
        & {"scoring_card", "scoring_pressure", "forced_response", "opponent_event", "chance_heavy"}
    )


def _requires_deep_adjudication(decision: Any, args: Any) -> bool:
    required = {str(trigger) for trigger in args.adaptive_deep_always_trigger}
    return bool(required.intersection(decision.triggers))


def _fails_configured_search_evidence_gate(
    decision: Any,
    teacher_eligible: bool,
    args: Any,
) -> bool:
    configured = {str(trigger) for trigger in args.adaptive_fail_closed_trigger}
    if not configured.intersection(decision.triggers):
        return False
    if not teacher_eligible:
        return True
    if int(decision.meaningful_depth) < int(args.adaptive_fail_closed_min_depth):
        return True
    return bool(
        args.adaptive_fail_closed_require_terminal
        and int(decision.terminal_evaluations) <= 0
    )


def _sync_continuation(planners: dict[str, dict[str, Any]], selected: Any, side: str) -> None:
    path = [dict(item) for item in selected.pending_continuation]
    seed = selected.pending_continuation_seed
    for by_side in planners.values():
        by_side[side].pending_continuation = [dict(item) for item in path]
        by_side[side].pending_continuation_seed = seed


def _clear_independent_root_continuations(
    planners: dict[str, dict[str, Any]],
    *,
    side: str,
    last_obs: dict[str, Any] | None,
    legal_actions: list[dict[str, Any]],
) -> str | None:
    """Clear every adaptive planner before a new card/headline root."""

    boundary = meaningful_decision_kind(last_obs, legal_actions)
    if boundary in {"headline", "card_choice"}:
        for by_side in planners.values():
            by_side[side].pending_continuation.clear()
            by_side[side].pending_continuation_seed = None
    return boundary


def is_deep_candidate(event: dict[str, Any], q_gap_threshold: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    triggers = set(event["triggers"])
    if abs(float(event["root_q_gap"])) <= q_gap_threshold:
        reasons.append("small_q_gap")
    if event["action_flipped"]:
        reasons.append("raw_search_disagreement")
    for trigger in (
        "scoring_card",
        "scoring_pressure",
        "forced_response",
        "opponent_event",
        "chance_heavy",
    ):
        if trigger in triggers:
            reasons.append(trigger)
    return bool(reasons), reasons


def _worker_main(
    worker_id: int,
    args_payload: dict[str, Any],
    game_queue: Any,
    request_queue: Any,
    response_queue: Any,
    result_queue: Any,
) -> None:
    args = SimpleNamespace(**args_payload)
    # Meaningful-decision search may traverse long deterministic/event chains
    # recursively between strategic nodes. Leave enough interpreter stack for
    # bridge JSON decoding near the configured atomic horizon.
    configure_search_recursion_limit(args.max_atomic_steps)
    output_dir = Path(args.output_dir)
    worker_dir = output_dir / "workers" / f"worker-{worker_id:03d}"
    replay_dir = worker_dir / "search_replay"
    anchor_replay_dir = worker_dir / "anchor_replay"
    log_dir = worker_dir / "game_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    writer = SearchReplayWriter(replay_dir, shard_size=int(args.shard_size))
    anchor_writer = SearchReplayWriter(anchor_replay_dir, shard_size=int(args.shard_size))
    client = BatchedInferenceClient(
        worker_id=worker_id,
        request_queue=request_queue,
        response_queue=response_queue,
        timeout_seconds=float(args.inference_timeout_seconds),
        cache_capacity=int(args.client_cache_capacity),
    )
    configs = {"base": search_config(args)}
    if bool(args.adaptive_search):
        configs.update(
            {
                "budget": search_config(
                    args,
                    meaningful_depth=int(args.meaningful_depth),
                    simulations=int(args.adaptive_budget_simulations),
                ),
                "deep": search_config(
                    args,
                    meaningful_depth=int(args.adaptive_deep_depth),
                    simulations=int(args.adaptive_deep_simulations),
                ),
            }
        )
    search_envs = {
        name: {
            side: TwilightStruggleEnv(env_config(args, None, search_mode=True))
            for side in ("us", "ussr")
        }
        for name in configs
    }
    planners = {
        name: {
            side: SelectivePuctPlanner(
                client,
                search_envs[name][side],
                config,
                seed=(
                    int(args.seed_base)
                    + 1_000_003 * (worker_id + 1)
                    + 10_003 * side_index
                    + 1_000_000_007 * config_index
                ),
            )
            for side_index, side in enumerate(("us", "ussr"))
        }
        for config_index, (name, config) in enumerate(configs.items())
    }
    env = TwilightStruggleEnv(env_config(args, log_dir))
    worker_summary: Counter[str] = Counter()
    worker_summary_float: Counter[str] = Counter()
    side_searches: Counter[str] = Counter()
    trigger_counts: Counter[str] = Counter()
    terminal_reasons: Counter[str] = Counter()
    deep_candidates: list[dict[str, Any]] = []
    try:
        while True:
            task = game_queue.get()
            if task is None:
                break
            game_index, seed = map(int, task)
            for by_side in planners.values():
                for planner in by_side.values():
                    planner.pending_continuation.clear()
            source_game = f"puct-selfplay-{game_index:06d}-seed-{seed}"
            transaction = DiskBackedCompleteGameReplayTransaction(
                writer,
                worker_dir / "staging" / "teacher" / f"game-{game_index:06d}",
                shard_size=int(args.staging_shard_size),
            )
            anchor_transaction = DiskBackedCompleteGameReplayTransaction(
                anchor_writer,
                worker_dir / "staging" / "anchor" / f"game-{game_index:06d}",
                shard_size=int(args.staging_shard_size),
            )
            game_started = time.perf_counter()
            final_info: dict[str, Any] = {}
            terminated = truncated = False
            game_events: list[dict[str, Any]] = []
            independent_boundary_events: list[dict[str, Any]] = []
            try:
                obs, _ = env.reset(seed=seed)
                while not (terminated or truncated):
                    side = str((env.last_obs or {}).get("side") or "ussr")
                    legal_actions = [dict(action) for action in env.legal_actions]
                    boundary_kind = _clear_independent_root_continuations(
                        planners,
                        side=side,
                        last_obs=env.last_obs,
                        legal_actions=legal_actions,
                    )
                    snapshot = env.snapshot(compact=True)
                    paired_root_seed = adaptive_root_search_seed(
                        int(args.seed_base), seed, int(env.episode_step)
                    )
                    for by_side in planners.values():
                        _reseed_planner_for_root(by_side[side], paired_root_seed)
                    base_decision = planners["base"][side].plan(
                        snapshot,
                        obs,
                        force_search=bool(args.force_search),
                    )
                    decision = base_decision
                    selected_planner = planners["base"][side]
                    convergence_status = None
                    convergence_configs = (
                        f"d{int(args.meaningful_depth)}_{int(base_decision.simulations)}",
                    )
                    convergence_rank_correlation = None
                    convergence_q_gap = float(abs(base_decision.root_q_gap))
                    search_effort_simulations = int(base_decision.simulations)
                    search_effort_seconds = float(base_decision.latency_seconds)
                    convergence_comparisons: list[dict[str, Any]] = []
                    teacher_eligible = True
                    if (
                        bool(args.adaptive_search)
                        and base_decision.searched
                        and _should_adapt(base_decision, len(legal_actions), args)
                    ):
                        budget_decision = planners["budget"][side].plan(
                            snapshot, obs, force_search=True
                        )
                        search_effort_simulations += int(budget_decision.simulations)
                        search_effort_seconds += float(budget_decision.latency_seconds)
                        comparison = _decision_comparison(base_decision, budget_decision, args)
                        convergence_comparisons.append(
                            {"left": convergence_configs[-1], "right": f"d{int(args.meaningful_depth)}_{int(budget_decision.simulations)}", **comparison}
                        )
                        decision = budget_decision
                        selected_planner = planners["budget"][side]
                        convergence_configs = (
                            convergence_configs[0],
                            f"d{int(args.meaningful_depth)}_{int(budget_decision.simulations)}",
                        )
                        convergence_rank_correlation = comparison.get("rank_correlation")
                        convergence_q_gap = float(comparison.get("right_q_gap", abs(budget_decision.root_q_gap)))
                        require_deep = _requires_deep_adjudication(budget_decision, args)
                        if comparison.get("stable") and not require_deep:
                            convergence_status = "BUDGET_CONVERGENCE"
                        elif comparison.get("near_tie") and not require_deep:
                            convergence_status = "NEAR_TIE"
                            teacher_eligible = False
                        else:
                            deep_decision = planners["deep"][side].plan(
                                snapshot, obs, force_search=True
                            )
                            search_effort_simulations += int(deep_decision.simulations)
                            search_effort_seconds += float(deep_decision.latency_seconds)
                            deep_comparison = _decision_comparison(budget_decision, deep_decision, args)
                            convergence_comparisons.append(
                                {
                                    "left": convergence_configs[-1],
                                    "right": f"d{int(args.adaptive_deep_depth)}_{int(deep_decision.simulations)}",
                                    **deep_comparison,
                                }
                            )
                            decision = deep_decision
                            selected_planner = planners["deep"][side]
                            convergence_configs = (
                                *convergence_configs,
                                f"d{int(args.adaptive_deep_depth)}_{int(deep_decision.simulations)}",
                            )
                            convergence_rank_correlation = deep_comparison.get("rank_correlation")
                            convergence_q_gap = float(deep_comparison.get("right_q_gap", abs(deep_decision.root_q_gap)))
                            if deep_comparison.get("stable"):
                                convergence_status = "HORIZON_CONVERGENCE"
                            elif deep_comparison.get("near_tie"):
                                convergence_status = "NEAR_TIE"
                                teacher_eligible = False
                            else:
                                convergence_status = "UNRESOLVED"
                                teacher_eligible = False
                                if bool(args.adaptive_fail_closed_unresolved_to_budget):
                                    decision = budget_decision
                                    selected_planner = planners["budget"][side]
                                    worker_summary["adaptive_unresolved_budget_fallbacks"] += 1
                    elif bool(args.adaptive_search) and base_decision.searched:
                        convergence_status = "UNVALIDATED"
                        teacher_eligible = False
                    if _fails_configured_search_evidence_gate(decision, teacher_eligible, args):
                        teacher_eligible = False
                        convergence_status = "UNRESOLVED"
                        decision = dataclasses.replace(
                            decision,
                            action=int(decision.raw_action),
                            behavior_path=(),
                        )
                        selected_planner.pending_continuation = []
                        selected_planner.pending_continuation_seed = None
                        worker_summary["fail_closed_raw_fallbacks"] += 1
                    if decision.evidence_gate_rejected:
                        teacher_eligible = False
                        worker_summary["planner_evidence_gate_rejections"] += 1
                    _sync_continuation(planners, selected_planner, side)
                    if boundary_kind in {"headline", "card_choice"}:
                        independent_boundary_events.append(
                            {
                                "source_step": int(env.episode_step),
                                "side": side,
                                "boundary_kind": boundary_kind,
                                "legal_action_count": len(legal_actions),
                                "searched": bool(decision.searched),
                                "continued": "search_continuation" in decision.triggers,
                                "atomic_horizon_cutoffs": int(decision.atomic_horizon_cutoffs),
                                "node_budget_cutoffs": int(decision.node_budget_cutoffs),
                                "evidence_gate_rejected": bool(decision.evidence_gate_rejected),
                            }
                        )
                    worker_summary["decisions"] += 1
                    if decision.searched:
                        worker_summary["searched_decisions"] += 1
                        worker_summary["simulations"] += search_effort_simulations
                        worker_summary[f"convergence_{str(convergence_status or 'not_adaptive').lower()}"] += 1
                        worker_summary["action_flips"] += int(decision.action_flipped)
                        worker_summary["behavior_teacher_disagreements"] += int(
                            decision.action != decision.teacher_action
                        )
                        worker_summary_float["search_seconds"] += search_effort_seconds
                        worker_summary_float["q_advantage"] += float(decision.q_advantage_over_raw)
                        side_searches[side] += 1
                        for trigger in decision.triggers:
                            trigger_counts[trigger] += 1
                        examples = examples_from_decision(
                            obs,
                            legal_actions,
                            side,
                            decision,
                            search_depth=decision.meaningful_depth,
                            source_game=source_game,
                            source_step=env.episode_step,
                            behavior_policy="parallel_selective_puct_selfplay",
                            behavior_temperature=float(args.behavior_temperature),
                            convergence_status=convergence_status,
                            convergence_configs=convergence_configs,
                            convergence_rank_correlation=(
                                None
                                if convergence_rank_correlation is None
                                else float(convergence_rank_correlation)
                            ),
                            convergence_q_gap=convergence_q_gap,
                        )
                        anchor_transaction.add_decision(examples, decision)
                        if teacher_eligible:
                            transaction.add_decision(examples, decision)
                        event = {
                            "source_game": source_game,
                            "seed": seed,
                            "action_index": env.episode_step,
                            "side": side,
                            "turn": int((env.last_obs or {}).get("turn") or 0),
                            "action_round": int((env.last_obs or {}).get("action_round") or 0),
                            "triggers": list(decision.triggers),
                            "simulations": int(decision.simulations),
                            "search_effort_simulations": search_effort_simulations,
                            "meaningful_depth": int(decision.meaningful_depth),
                            "raw_action": int(decision.raw_action),
                            "teacher_action": decision.teacher_action,
                            "behavior_action": int(decision.action),
                            "action_flipped": bool(decision.action_flipped),
                            "root_q_gap": float(decision.root_q_gap),
                            "q_advantage_over_raw": float(decision.q_advantage_over_raw),
                            "latency_seconds": float(decision.latency_seconds),
                            "search_effort_seconds": search_effort_seconds,
                            "max_turn_delta": int(decision.max_turn_delta),
                            "terminal_evaluations": int(decision.terminal_evaluations),
                            "evidence_gate_rejected": bool(decision.evidence_gate_rejected),
                            "convergence_status": convergence_status,
                            "convergence_configs": list(convergence_configs),
                            "convergence_rank_correlation": convergence_rank_correlation,
                            "convergence_q_gap": convergence_q_gap,
                            "convergence_comparisons": convergence_comparisons,
                            "teacher_eligible": teacher_eligible,
                            "root_search_seed": paired_root_seed,
                        }
                        selected, reasons = is_deep_candidate(event, float(args.deep_candidate_q_gap))
                        if selected and len(deep_candidates) < int(args.deep_candidate_max):
                            deep_candidates.append({**event, "adjudication_reasons": reasons})
                        game_events.append(event)
                    obs, _reward, terminated, truncated, final_info = env.step(decision.action)
                    heartbeat_every = max(0, int(args.heartbeat_every_decisions))
                    if heartbeat_every and env.episode_step % heartbeat_every == 0:
                        _atomic_json(
                            worker_dir / "heartbeat.json",
                            {
                                "worker_id": worker_id,
                                "game_index": game_index,
                                "seed": seed,
                                "episode_step": int(env.episode_step),
                                "turn": int((env.last_obs or {}).get("turn") or 0),
                                "action_round": int((env.last_obs or {}).get("action_round") or 0),
                                "last_search_effort_seconds": float(search_effort_seconds),
                                "resident_memory_mib": _resident_memory_mib(),
                                "updated_at": time.time(),
                            },
                        )
                    trim_every = max(0, int(args.heap_trim_every_decisions))
                    if trim_every and env.episode_step % trim_every == 0:
                        _trim_worker_heap()

                winner = str(final_info.get("winner") or "")
                reason = str(final_info.get("terminal_reason") or "unknown")
                if truncated:
                    transaction.invalidate("truncated_game")
                    anchor_transaction.invalidate("truncated_game")
                elif winner not in {"us", "ussr"}:
                    transaction.invalidate("missing_terminal_winner")
                    anchor_transaction.invalidate("missing_terminal_winner")
                committed = transaction.commit(winner=winner, terminal_reason=reason) if not truncated else 0
                anchor_committed = (
                    anchor_transaction.commit(winner=winner, terminal_reason=reason)
                    if not truncated
                    else 0
                )
                worker_summary["games_completed"] += 1
                worker_summary["committed_examples"] += committed
                worker_summary["partial_search_labels_rejected"] += transaction.partial_decisions_rejected
                if committed == 0 and transaction.invalid_reason is not None:
                    worker_summary["discarded_games"] += 1
                terminal_reasons[reason] += 1
                # The environment appends every episode from one worker process
                # to the same JSONL. Reused workers therefore do not create a
                # new filename for their second game.
                process_logs = sorted(
                    log_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime_ns
                )
                game_log = str(process_logs[-1].resolve()) if process_logs else None
                for candidate in deep_candidates:
                    if candidate["source_game"] == source_game and "game_log" not in candidate:
                        candidate["game_log"] = game_log
                        candidate["game_occurrence"] = 0
                record = {
                    "worker_id": worker_id,
                    "game_index": game_index,
                    "source_game": source_game,
                    "seed": seed,
                    "winner": winner,
                    "terminal_reason": reason,
                    "steps": env.episode_step,
                    "searched_decisions": len(game_events),
                    "simulations": sum(int(event["search_effort_simulations"]) for event in game_events),
                    "action_flips": sum(bool(event["action_flipped"]) for event in game_events),
                    "behavior_teacher_disagreements": sum(
                        int(event["behavior_action"] != event["teacher_action"])
                        for event in game_events
                    ),
                    "search_seconds": sum(float(event["search_effort_seconds"]) for event in game_events),
                    "convergence_statuses": dict(
                        Counter(str(event["convergence_status"] or "not_adaptive") for event in game_events)
                    ),
                    "side_searches": dict(Counter(str(event["side"]) for event in game_events)),
                    "trigger_counts": dict(
                        Counter(trigger for event in game_events for trigger in event["triggers"])
                    ),
                    "committed_examples": committed,
                    "anchor_committed_examples": anchor_committed,
                    "discard_reason": transaction.invalid_reason,
                    "partial_search_labels_rejected": transaction.partial_decisions_rejected,
                    "wall_clock_seconds": time.perf_counter() - game_started,
                    "game_log": game_log,
                    "deep_candidates": [
                        candidate
                        for candidate in deep_candidates
                        if candidate["source_game"] == source_game
                    ],
                    "independent_boundary_events": independent_boundary_events,
                }
                writer.flush()
                anchor_writer.flush()
                _atomic_json(
                    output_dir / "completed_games" / f"game-{game_index:06d}.json",
                    record,
                )
                _atomic_json(
                    worker_dir / "heartbeat.json",
                    {
                        "worker_id": worker_id,
                        "game_index": game_index,
                        "seed": seed,
                        "episode_step": int(env.episode_step),
                        "status": "complete",
                        "resident_memory_mib": _resident_memory_mib(),
                        "updated_at": time.time(),
                    },
                )
                result_queue.put({"event": "game_complete", "record": record})
            except BaseException:
                transaction.invalidate("worker_exception")
                anchor_transaction.invalidate("worker_exception")
                raise
    except BaseException as exc:
        result_queue.put(
            {
                "event": "worker_error",
                "worker_id": worker_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "recursion_limit": sys.getrecursionlimit(),
            }
        )
        raise
    finally:
        manifest = writer.close()
        anchor_manifest = anchor_writer.close()
        env.close()
        for by_side in search_envs.values():
            for search_env in by_side.values():
                search_env.close()
        payload = {
            **dict(worker_summary),
            **dict(worker_summary_float),
            "worker_id": worker_id,
            "replay_manifest": str(manifest.resolve()),
            "anchor_replay_manifest": str(anchor_manifest.resolve()),
            "side_searches": dict(side_searches),
            "trigger_counts": dict(trigger_counts),
            "terminal_reasons": dict(terminal_reasons),
            "client_cache_hits": client.cache_hits,
            "client_cache_misses": client.cache_misses,
            "deep_candidates": deep_candidates,
        }
        result_queue.put({"event": "worker_done", "summary": payload})


def _gpu_sample(gpus: list[str]) -> list[dict[str, float | int]]:
    try:
        command = [
            "nvidia-smi",
            f"--id={','.join(gpus)}",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(command, text=True, timeout=5)
        result = []
        for line in output.splitlines():
            index, utilization, memory = [part.strip() for part in line.split(",")]
            result.append(
                {"gpu": int(index), "utilization": float(utilization), "memory_mib": float(memory)}
            )
        return result
    except Exception:
        return []


def _gpu_process_sample(gpus: list[str], owned_pids: set[int]) -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "pmon", "-i", ",".join(gpus), "-c", "1", "-s", "um"],
            text=True,
            timeout=5,
        )
        owned: list[dict[str, Any]] = []
        external: list[dict[str, Any]] = []
        for line in output.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 10 or fields[1] == "-":
                continue
            def number(value: str) -> float:
                return 0.0 if value == "-" else float(value)
            entry = {
                "gpu": int(fields[0]),
                "pid": int(fields[1]),
                "sm_percent": number(fields[3]),
                "memory_percent": number(fields[4]),
                "memory_mib": number(fields[9]),
                "command": fields[-1],
            }
            (owned if entry["pid"] in owned_pids else external).append(entry)
        return {"owned": owned, "external": external}
    except Exception:
        return {"owned": [], "external": []}


def _merge_replay_manifests(
    output_dir: Path,
    worker_summaries: list[dict[str, Any]],
    *,
    manifest_key: str = "replay_manifest",
    target_name: str = "search_replay",
) -> Path:
    # A resumed collection can intentionally use fewer workers than an earlier
    # process. Replay from completed games owned by now-inactive worker IDs was
    # already validated/pruned at startup and must still be included here.
    manifests_by_worker: dict[int, Path] = {}
    for summary in worker_summaries:
        worker_id = int(summary["worker_id"])
        manifest_path = Path(summary[manifest_key]).resolve()
        prior = manifests_by_worker.setdefault(worker_id, manifest_path)
        if prior != manifest_path:
            raise RuntimeError(f"conflicting replay manifests for worker {worker_id}")
    for worker_dir in sorted((output_dir / "workers").glob("worker-*")):
        try:
            worker_id = int(worker_dir.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        manifest_path = (worker_dir / target_name / "manifest.json").resolve()
        if not manifest_path.is_file():
            continue
        prior = manifests_by_worker.setdefault(worker_id, manifest_path)
        if prior != manifest_path:
            raise RuntimeError(f"conflicting replay manifests for worker {worker_id}")

    target_dir = output_dir / target_name
    temporary = target_dir.with_name(target_dir.name + ".merge.tmp")
    backup = target_dir.with_name(target_dir.name + ".merge.backup")
    if backup.exists():
        if target_dir.exists():
            shutil.rmtree(backup, ignore_errors=True)
        else:
            os.replace(backup, target_dir)
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    total = 0
    for worker_id, manifest_path in sorted(manifests_by_worker.items()):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for shard_index, shard in enumerate(payload.get("shards", [])):
            source = manifest_path.parent / str(shard["path"])
            name = f"worker-{worker_id:03d}-{shard_index:06d}.npz"
            target = temporary / name
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            count = int(shard["examples"])
            total += count
            shards.append({"path": name, "examples": count})
    manifest = temporary / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": SEARCH_REPLAY_VERSION,
                "format": "compressed_npz_shards",
                "examples": total,
                "shard_size": None,
                "shards": shards,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if target_dir.exists():
        os.replace(target_dir, backup)
    try:
        os.replace(temporary, target_dir)
    except BaseException:
        if backup.exists():
            os.replace(backup, target_dir)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    return target_dir / "manifest.json"


def _drain_results(result_queue: Any, records: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    while True:
        try:
            event = result_queue.get_nowait()
        except queue.Empty:
            break
        if event["event"] == "game_complete":
            records.append(event["record"])
            print({"event": "parallel_puct_game", **event["record"]}, flush=True)
        elif event["event"] == "worker_done":
            summaries.append(event["summary"])
        elif event["event"] == "worker_error":
            detail = f"worker {event['worker_id']}: {event['error']}"
            if event.get("traceback"):
                detail += f"\n{event['traceback']}"
            if event.get("recursion_limit"):
                detail += f"\nworker_recursion_limit={event['recursion_limit']}"
            errors.append(detail)
    return errors


def _resume_throughput_totals(
    records: list[dict[str, Any]],
    completed_records: list[dict[str, Any]],
) -> dict[str, int]:
    """Return work completed by this process, excluding durable resume inputs."""

    resumed_indices = {int(record["game_index"]) for record in completed_records}
    new_records = [
        record for record in records if int(record["game_index"]) not in resumed_indices
    ]
    return {
        "new_games_completed": len(new_records),
        "new_simulations": sum(int(record.get("simulations", 0)) for record in new_records),
        "new_searched_decisions": sum(
            int(record.get("searched_decisions", 0)) for record in new_records
        ),
    }


def _stop_inference_servers(request_queue: Any, servers: list[Any]) -> None:
    """Stop every server that reached ``Process.start`` without leaking children."""

    alive: list[Any] = []
    for server in servers:
        try:
            if server.is_alive():
                alive.append(server)
        except (AssertionError, ValueError):
            continue
    for _server in alive:
        try:
            request_queue.put(None, timeout=1.0)
        except queue.Full:
            break
    for server in alive:
        server.join(timeout=30)
        if server.is_alive():
            server.terminate()
            server.join(timeout=5)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if (args.output_dir / "collection_summary.json").exists():
        raise SystemExit(f"output already contains a completed collection: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    collection_lock = _acquire_collection_lock(args.output_dir)
    identity = _collection_identity(args)
    completed_records = _load_completed_records(args.output_dir, identity)
    resume_prune = _prune_interrupted_worker_replay(
        args.output_dir,
        completed_records,
        shard_size=int(args.shard_size),
    )
    if any(int(value) for key, value in resume_prune.items() if key.endswith("_dropped")):
        print({"event": "resume_replay_pruned", **resume_prune}, flush=True)
    completed_indices = {int(record["game_index"]) for record in completed_records}
    gpus = [item.strip() for item in str(args.gpus).split(",") if item.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one physical GPU index")
    if args.workers < len(gpus):
        raise SystemExit("--workers must be at least the number of inference GPUs")
    context = mp.get_context("spawn")
    request_queue = context.Queue(maxsize=max(256, args.workers * 32))
    response_queues = [context.Queue(maxsize=8) for _ in range(args.workers)]
    control_queue = context.Queue()
    result_queue = context.Queue()
    game_queue = context.Queue()
    server_devices = [
        gpu
        for gpu in gpus
        for _replica in range(max(1, int(args.inference_replicas_per_gpu)))
    ]
    server_stats = [context.Array("d", STAT_COUNT, lock=False) for _gpu in server_devices]
    servers = [
        context.Process(
            target=inference_server_main,
            kwargs={
                "server_id": server_id,
                "gpu": gpu,
                "checkpoint": str(args.checkpoint),
                "request_queue": request_queue,
                "response_queues": response_queues,
                "control_queue": control_queue,
                "stats": server_stats[server_id],
                "max_batch_size": args.max_inference_batch_size,
                "max_batch_delay_ms": args.max_inference_delay_ms,
                "cache_capacity": args.inference_cache_capacity,
            },
            name=f"puct-inference-gpu-{gpu}",
        )
        for server_id, gpu in enumerate(server_devices)
    ]
    try:
        for server in servers:
            server.start()
        owned_inference_pids = {
            int(server.pid) for server in servers if server.pid is not None
        }
        ready = []
        deadline = time.monotonic() + 300.0
        while len(ready) < len(servers):
            timeout = deadline - time.monotonic()
            if timeout <= 0.0:
                raise TimeoutError("timed out waiting for inference servers")
            failed_servers = _failed_processes(servers)
            if failed_servers:
                raise RuntimeError(
                    "inference server exited before readiness: " + "; ".join(failed_servers)
                )
            try:
                event = control_queue.get(timeout=min(1.0, timeout))
            except queue.Empty:
                continue
            print(event, flush=True)
            if event["event"] == "inference_server_error":
                raise RuntimeError(event["error"])
            if event["event"] == "inference_server_ready":
                ready.append(event)
    except BaseException:
        _stop_inference_servers(request_queue, servers)
        collection_lock.close()
        raise

    args_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    for game_index in range(args.games):
        if game_index not in completed_indices:
            game_queue.put((game_index, args.seed_base + game_index))
    for _worker in range(args.workers):
        game_queue.put(None)
    workers = [
        context.Process(
            target=_worker_main,
            args=(worker_id, args_payload, game_queue, request_queue, response_queues[worker_id], result_queue),
            name=f"puct-game-worker-{worker_id:03d}",
        )
        for worker_id in range(args.workers)
    ]
    started = time.perf_counter()
    for worker in workers:
        worker.start()
    records: list[dict[str, Any]] = list(completed_records)
    worker_summaries: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    errors: list[str] = []
    next_telemetry = 0.0
    try:
        while any(worker.is_alive() for worker in workers):
            failed_servers = _failed_processes(servers)
            if failed_servers:
                raise RuntimeError("inference server exited: " + "; ".join(failed_servers))
            errors.extend(_drain_results(result_queue, records, worker_summaries))
            if errors:
                raise RuntimeError("; ".join(errors))
            now = time.perf_counter()
            if now >= next_telemetry:
                try:
                    queue_depth = int(request_queue.qsize())
                except (NotImplementedError, OSError):
                    queue_depth = -1
                try:
                    import psutil

                    cpu_percent = float(psutil.cpu_percent(interval=None))
                except Exception:
                    cpu_percent = -1.0
                sample = {
                    "elapsed_seconds": now - started,
                    "games_completed": len(records),
                    "request_queue_depth": queue_depth,
                    "cpu_percent": cpu_percent,
                    "load_average": list(os.getloadavg()),
                    "gpus": _gpu_sample(gpus),
                    "gpu_processes": _gpu_process_sample(gpus, owned_inference_pids),
                    "inference": [
                        inference_stats_payload(index, gpu, server_stats[index])
                        for index, gpu in enumerate(server_devices)
                    ],
                }
                telemetry.append(sample)
                with (args.output_dir / "throughput.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample, sort_keys=True) + "\n")
                print({"event": "parallel_puct_throughput", **sample}, flush=True)
                next_telemetry = now + max(1.0, float(args.telemetry_interval_seconds))
            time.sleep(0.25)
        for worker in workers:
            worker.join(timeout=5)
        errors.extend(_drain_results(result_queue, records, worker_summaries))
        failed = [f"{worker.name} exit={worker.exitcode}" for worker in workers if worker.exitcode != 0]
        if errors or failed:
            raise RuntimeError("; ".join(errors + failed))
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        _stop_inference_servers(request_queue, servers)

    elapsed = time.perf_counter() - started
    manifest = _merge_replay_manifests(args.output_dir, worker_summaries)
    anchor_manifest = _merge_replay_manifests(
        args.output_dir,
        worker_summaries,
        manifest_key="anchor_replay_manifest",
        target_name="anchor_replay",
    )
    records.sort(key=lambda item: int(item["game_index"]))
    with (args.output_dir / "games.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    deep_candidates = stratified_deep_candidates(records, args.deep_candidate_max)
    (args.output_dir / "deep_adjudication_candidates.json").write_text(
        json.dumps({"positions": deep_candidates}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def sum_int(key: str) -> int:
        return sum(int(summary.get(key, 0)) for summary in worker_summaries)

    def sum_float(key: str) -> float:
        return sum(float(summary.get(key, 0.0)) for summary in worker_summaries)

    searched = sum(int(record.get("searched_decisions", 0)) for record in records)
    decisions = sum(int(record.get("steps", 0)) for record in records)
    simulations = sum(int(record.get("simulations", 0)) for record in records)
    trigger_counts: Counter[str] = Counter()
    side_searches: Counter[str] = Counter()
    terminal_reasons: Counter[str] = Counter()
    for record in records:
        trigger_counts.update(record.get("trigger_counts", {}))
        side_searches.update(record.get("side_searches", {}))
        terminal_reasons[str(record.get("terminal_reason") or "unknown")] += 1
    gpu_samples = [entry for sample in telemetry for entry in sample["gpus"]]
    owned_gpu_process_samples = [
        entry
        for sample in telemetry
        for entry in sample.get("gpu_processes", {}).get("owned", [])
    ]
    external_gpu_process_samples = [
        entry
        for sample in telemetry
        for entry in sample.get("gpu_processes", {}).get("external", [])
    ]
    queue_samples = [sample["request_queue_depth"] for sample in telemetry if sample["request_queue_depth"] >= 0]
    cpu_samples = [sample["cpu_percent"] for sample in telemetry if sample["cpu_percent"] >= 0.0]
    resume_throughput = _resume_throughput_totals(records, completed_records)
    summary = {
        "checkpoint": str(args.checkpoint),
        "resumed_games": len(completed_records),
        "games_requested": args.games,
        "games_completed": len(records),
        "discarded_games": sum(bool(record.get("discard_reason")) for record in records),
        "partial_search_labels_rejected": sum(
            int(record.get("partial_search_labels_rejected", 0)) for record in records
        ),
        "committed_examples": sum(int(record.get("committed_examples", 0)) for record in records),
        "anchor_committed_examples": sum(
            int(record.get("anchor_committed_examples", 0)) for record in records
        ),
        "decisions": decisions,
        "searched_decisions": searched,
        "search_fraction": searched / max(1, decisions),
        "search_flip_rate": sum(int(record.get("action_flips", 0)) for record in records) / max(1, searched),
        "behavior_teacher_disagreement_rate": sum(
            int(record.get("behavior_teacher_disagreements", 0)) for record in records
        ) / max(1, searched),
        "average_simulations_per_search": simulations / max(1, searched),
        "average_search_latency_seconds": sum(
            float(record.get("search_seconds", 0.0)) for record in records
        ) / max(1, searched),
        "new_games_completed": resume_throughput["new_games_completed"],
        "new_simulations": resume_throughput["new_simulations"],
        "new_searched_decisions": resume_throughput["new_searched_decisions"],
        "simulations_per_second": resume_throughput["new_simulations"] / max(1e-9, elapsed),
        "games_per_hour": resume_throughput["new_games_completed"] * 3600.0 / max(1e-9, elapsed),
        "average_wall_clock_seconds_per_game": sum(float(record["wall_clock_seconds"]) for record in records) / max(1, len(records)),
        "elapsed_seconds": elapsed,
        "side_searches": dict(side_searches),
        "trigger_counts": dict(trigger_counts),
        "terminal_reasons": dict(terminal_reasons),
        "replay_manifest": str(manifest.resolve()),
        "anchor_replay_manifest": str(anchor_manifest.resolve()),
        "inference_servers": [
            inference_stats_payload(index, gpu, server_stats[index])
            for index, gpu in enumerate(server_devices)
        ],
        "average_gpu_utilization": sum(float(item["utilization"]) for item in gpu_samples) / max(1, len(gpu_samples)),
        "average_gpu_memory_mib": sum(float(item["memory_mib"]) for item in gpu_samples) / max(1, len(gpu_samples)),
        "average_owned_inference_sm_percent": sum(
            float(item["sm_percent"]) for item in owned_gpu_process_samples
        ) / max(1, len(owned_gpu_process_samples)),
        "average_external_sm_percent": sum(
            float(item["sm_percent"]) for item in external_gpu_process_samples
        ) / max(1, len(external_gpu_process_samples)),
        "average_cpu_percent": sum(cpu_samples) / max(1, len(cpu_samples)),
        "average_inference_queue_depth": sum(queue_samples) / max(1, len(queue_samples)),
        "max_inference_queue_depth": max(queue_samples, default=-1),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "collection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    collection_lock.close()


if __name__ == "__main__":
    main()
