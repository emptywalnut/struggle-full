#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from struggle_ai.batched_inference import (
    STAT_COUNT,
    BatchedInferenceClient,
    inference_server_main,
    inference_stats_payload,
)
from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.frozen_diagnostic import find_action_index
from struggle_ai.mcts import SelectivePuctConfig, SelectivePuctPlanner
from struggle_ai.search_replay import (
    SearchReplayWriter,
    examples_from_decision,
    iter_search_replay,
    with_terminal_outcome,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate targeted D4/D8 search labels through shared batched GPU inference."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--position-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-searches", type=int, default=None)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--roots-per-task",
        type=int,
        default=0,
        help=(
            "Maximum roots from one source game handled serially by a worker. "
            "0 preserves one replay task per game; 1 maximizes deep-root parallelism."
        ),
    )
    parser.add_argument("--inference-replicas-per-gpu", type=int, default=2)
    parser.add_argument("--max-inference-batch-size", type=int, default=32)
    parser.add_argument("--max-inference-delay-ms", type=float, default=3.0)
    parser.add_argument("--inference-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--inference-cache-capacity", type=int, default=65_536)
    parser.add_argument("--client-cache-capacity", type=int, default=4_096)
    parser.add_argument("--initial-simulations", type=int, default=128)
    parser.add_argument("--medium-simulations", type=int, default=128)
    parser.add_argument("--max-simulations", type=int, default=128)
    parser.add_argument("--meaningful-depth", type=int, default=4)
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--max-atomic-steps", type=int, default=1_024)
    parser.add_argument("--max-expanded-nodes", type=int, default=2_048)
    parser.add_argument(
        "--max-search-worker-rss-mib",
        type=float,
        default=0.0,
        help="Fail closed when one search worker exceeds this resident-memory limit.",
    )
    parser.add_argument(
        "--max-search-seconds",
        type=float,
        default=0.0,
        help="Fail closed when one root search exceeds this wall-clock limit.",
    )
    parser.add_argument("--child-initial", type=int, default=4)
    parser.add_argument("--child-max", type=int, default=24)
    parser.add_argument("--max-exact-chance-outcomes", type=int, default=36)
    parser.add_argument("--search-temperature", type=float, default=1.0)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=91_000_000)
    parser.add_argument(
        "--root-seeded-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reset planner RNG from seed/game/action before every root. Use the same --seed "
            "across budget/depth sweeps to keep belief/chance streams paired."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse identity-verified durable per-root results after interruption.",
    )
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
        "max_episode_steps": 1_200,
        "search_mode": True,
        "log_action_details": False,
        "log_games_dir": None,
        "metrics_dir": None,
    }


def search_config(args: Any) -> SelectivePuctConfig:
    return SelectivePuctConfig(
        initial_simulations=int(args.initial_simulations),
        medium_simulations=int(args.medium_simulations),
        max_simulations=int(args.max_simulations),
        belief_particles=int(args.belief_particles),
        meaningful_depth=int(args.meaningful_depth),
        max_atomic_steps=int(args.max_atomic_steps),
        max_expanded_nodes=int(args.max_expanded_nodes),
        max_process_rss_mib=float(args.max_search_worker_rss_mib),
        max_search_seconds=float(args.max_search_seconds),
        child_initial=int(args.child_initial),
        child_max=int(args.child_max),
        max_exact_chance_outcomes=int(args.max_exact_chance_outcomes),
        temperature=float(args.search_temperature),
    )


def read_terminal_game(path: Path, seed: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle) if path.suffix == ".json" else [
            json.loads(line) for line in handle if line.strip()
        ]
    if isinstance(records, dict):
        records = [records]
    matches = [
        record
        for record in records
        if record.get("kind") == "terminal"
        and record.get("actions")
        and int(record.get("seed", -1)) == seed
    ]
    if not matches:
        raise ValueError(f"no complete terminal game for seed={seed} in {path}")
    return matches[0]


def root_search_seed(base_seed: int, game_seed: int, action_index: int) -> int:
    return int(base_seed) + int(game_seed) * 1_000_003 + int(action_index) * 10_007


def root_key(seed: int, occurrence: int, action_index: int) -> tuple[int, int, int]:
    return int(seed), int(occurrence), int(action_index)


def root_result_path(output_dir: Path, key: tuple[int, int, int]) -> Path:
    seed, occurrence, action_index = key
    return output_dir / "completed_roots" / (
        f"seed-{seed:012d}-occurrence-{occurrence:03d}-action-{action_index:06d}.json"
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def generation_identity(args: Any, position_manifest: Path) -> dict[str, Any]:
    manifest_bytes = position_manifest.read_bytes()
    ignored = {
        "gpus",
        "workers",
        "inference_replicas_per_gpu",
        "max_inference_batch_size",
        "max_inference_delay_ms",
        "inference_timeout_seconds",
        "inference_cache_capacity",
        "client_cache_capacity",
        "output_dir",
        "resume",
    }
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ignored
    }
    return {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "position_manifest": str(position_manifest),
        "position_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "config": config,
    }


def load_completed_roots(output_dir: Path, identity: dict[str, Any], *, resume: bool) -> list[dict[str, Any]]:
    identity_path = output_dir / "generation_identity.json"
    if identity_path.is_file():
        prior = json.loads(identity_path.read_text(encoding="utf-8"))
        if prior != identity:
            raise ValueError(f"cannot resume targeted search with mismatched identity: {identity_path}")
    else:
        atomic_json(identity_path, identity)
    if not resume:
        return []
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for path in sorted((output_dir / "completed_roots").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = root_key(record["seed"], record.get("game_occurrence", 0), record["action_index"])
        if key in seen:
            raise ValueError(f"duplicate durable targeted-search root: {key}")
        seen.add(key)
        records.append(record)
    return records


def filter_completed_tasks(
    tasks: list[dict[str, Any]], completed: set[tuple[int, int, int]]
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for task in tasks:
        seed = int(task["seed"])
        occurrence = int(task.get("game_occurrence", 0))
        positions = [
            position
            for position in task["positions"]
            if root_key(seed, occurrence, int(position["action_index"])) not in completed
        ]
        if positions:
            pending.append({**task, "positions": positions})
    return pending


def reset_planner_for_root(planner: SelectivePuctPlanner, seed: int) -> None:
    planner.seed = int(seed)
    planner.rng = np.random.default_rng(int(seed))
    planner.search_index = 0
    planner.pending_continuation.clear()
    planner.pending_continuation_seed = None
    planner.scoring_search_counts.clear()
    planner.chance_search_counts.clear()
    planner.allocation_search_counts.clear()


def build_root_tasks(
    positions: list[dict[str, Any]], roots_per_task: int = 0
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for position in positions:
        if not position.get("game_log"):
            raise ValueError(f"position has no complete game_log: {position}")
        key = (
            str(Path(position["game_log"]).expanduser().resolve()),
            int(position["seed"]),
            int(position.get("game_occurrence", 0)),
        )
        grouped[key].append(position)
    tasks: list[dict[str, Any]] = []
    cap = int(roots_per_task)
    for (game_log, seed, occurrence), selected in grouped.items():
        ordered = sorted(selected, key=lambda position: int(position["action_index"]))
        chunks = (
            [ordered]
            if cap <= 0
            else [ordered[index : index + cap] for index in range(0, len(ordered), cap)]
        )
        tasks.extend(
            {
                "game_log": game_log,
                "seed": seed,
                "game_occurrence": occurrence,
                "positions": chunk,
            }
            for chunk in chunks
        )
    return tasks


def _worker_main(
    worker_id: int,
    args_payload: dict[str, Any],
    task_queue: Any,
    request_queue: Any,
    response_queue: Any,
    result_queue: Any,
) -> None:
    args = SimpleNamespace(**args_payload)
    worker_dir = Path(args.output_dir) / "workers" / f"worker-{worker_id:03d}"
    writer = SearchReplayWriter(worker_dir / "search_replay", shard_size=int(args.shard_size))
    client = BatchedInferenceClient(
        worker_id=worker_id,
        request_queue=request_queue,
        response_queue=response_queue,
        timeout_seconds=float(args.inference_timeout_seconds),
        cache_capacity=int(args.client_cache_capacity),
    )
    replay_env = TwilightStruggleEnv(env_config())
    search_env = TwilightStruggleEnv(env_config())
    planner = SelectivePuctPlanner(
        client,
        search_env,
        search_config(args),
        seed=int(args.seed) + worker_id * 10_003,
    )
    summary: Counter[str] = Counter()
    summary_float: Counter[str] = Counter()
    side_searches: Counter[str] = Counter()
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            game_log = Path(task["game_log"])
            game_seed = int(task["seed"])
            occurrence = int(task.get("game_occurrence", 0))
            positions = {int(position["action_index"]): position for position in task["positions"]}
            game = read_terminal_game(game_log, game_seed)
            game_info = dict(game.get("info") or {})
            winner = str(game.get("winner") or game_info.get("winner") or "")
            if winner not in {"us", "ussr"}:
                raise ValueError(f"targeted teacher requires winner in complete game: {game_log}")
            obs, _ = replay_env.reset(seed=game_seed)
            for step_index, record in enumerate(game["actions"]):
                if step_index in positions:
                    started = time.perf_counter()
                    legal_actions = copy.deepcopy(replay_env.legal_actions)
                    paired_root_seed = None
                    if bool(args.root_seeded_search):
                        paired_root_seed = root_search_seed(int(args.seed), game_seed, step_index)
                        reset_planner_for_root(planner, paired_root_seed)
                    decision = planner.plan(
                        replay_env.snapshot(compact=True), obs, force_search=True
                    )
                    summary["searched"] += 1
                    summary["simulations"] += int(decision.simulations)
                    summary["flips"] += int(decision.action_flipped)
                    summary_float["latency_seconds"] += float(decision.latency_seconds)
                    side = str((replay_env.last_obs or {}).get("side") or "ussr")
                    side_searches[side] += 1
                    complete = not (
                        decision.atomic_horizon_cutoffs or decision.node_budget_cutoffs
                    )
                    written = 0
                    if decision.searched and complete:
                        generated = examples_from_decision(
                            obs,
                            legal_actions,
                            side,
                            decision,
                            search_depth=decision.meaningful_depth,
                            source_game=f"seed-{game_seed}-occurrence-{occurrence}",
                            source_step=step_index,
                            behavior_policy="parallel_targeted_puct_teacher",
                            behavior_temperature=0.0,
                        )
                        generated = with_terminal_outcome(
                            generated,
                            winner=winner,
                            terminal_reason=str(
                                game.get("terminal_reason")
                                or game_info.get("terminal_reason")
                                or "unknown"
                            ),
                        )
                        for example in generated:
                            writer.add(example)
                            written += 1
                        writer.flush()
                        summary["examples"] += written
                    else:
                        summary["discarded_partial_roots"] += 1
                    result_queue.put(
                        {
                            "event": "root_complete",
                            "worker_id": worker_id,
                            "seed": game_seed,
                            "game_occurrence": occurrence,
                            "action_index": step_index,
                            "side": side,
                            "simulations": int(decision.simulations),
                            "examples": written,
                            "search_complete": complete,
                            "action_flipped": bool(decision.action_flipped),
                            "root_q_gap": float(decision.root_q_gap),
                            "latency_seconds": time.perf_counter() - started,
                            "root_search_seed": paired_root_seed,
                        }
                    )
                logged_index = find_action_index(replay_env.legal_actions, record["action"])
                obs, _, terminated, truncated, _ = replay_env.step(logged_index)
                if terminated or truncated:
                    break
            summary["games_replayed"] += 1
    except BaseException as exc:
        result_queue.put({"event": "worker_error", "worker_id": worker_id, "error": repr(exc)})
        raise
    finally:
        manifest = writer.close()
        replay_env.close()
        search_env.close()
        result_queue.put(
            {
                "event": "worker_done",
                "summary": {
                    **dict(summary),
                    **dict(summary_float),
                    "worker_id": worker_id,
                    "side_searches": dict(side_searches),
                    "replay_manifest": str(manifest.resolve()),
                    "client_cache_hits": client.cache_hits,
                    "client_cache_misses": client.cache_misses,
                },
            }
        )


def merge_manifests(output_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    del summaries  # Durable worker manifests, including prior runs, are authoritative.
    manifests = sorted((output_dir / "workers").glob("worker-*/search_replay/manifest.json"))
    temporary_dir = output_dir / "search_replay.merge-tmp"
    shutil.rmtree(temporary_dir, ignore_errors=True)
    with SearchReplayWriter(temporary_dir, shard_size=64) as writer:
        for example in iter_search_replay(manifests):
            writer.add(example)
    target_dir = output_dir / "search_replay"
    backup_dir = output_dir / "search_replay.merge-old"
    shutil.rmtree(backup_dir, ignore_errors=True)
    if target_dir.exists():
        os.replace(target_dir, backup_dir)
    os.replace(temporary_dir, target_dir)
    shutil.rmtree(backup_dir, ignore_errors=True)
    return target_dir / "manifest.json"


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = json.loads(
        args.position_manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    position_manifest = args.position_manifest.expanduser().resolve()
    identity = generation_identity(args, position_manifest)
    resumed_roots = load_completed_roots(
        args.output_dir, identity, resume=bool(args.resume)
    )
    positions = list(manifest_payload.get("positions") or [])
    if args.max_searches is not None:
        positions = positions[: max(0, int(args.max_searches))]
    if not positions:
        raise SystemExit("position manifest contains no selected roots")

    try:
        tasks = build_root_tasks(positions, int(args.roots_per_task))
        completed_keys = {
            root_key(row["seed"], row.get("game_occurrence", 0), row["action_index"])
            for row in resumed_roots
        }
        tasks = filter_completed_tasks(tasks, completed_keys)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    gpus = [part.strip() for part in str(args.gpus).split(",") if part.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one GPU")
    if not tasks:
        replay_manifest = merge_manifests(args.output_dir, [])
        roots = resumed_roots
        elapsed = 0.0
        summary = build_generation_summary(
            args,
            positions=positions,
            tasks=[],
            worker_count=0,
            roots=roots,
            replay_manifest=replay_manifest,
            elapsed=elapsed,
            server_devices=[],
            server_stats=[],
            resumed_roots=len(resumed_roots),
        )
        write_generation_outputs(args.output_dir, summary, roots)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return
    worker_count = min(max(1, int(args.workers)), len(tasks))
    context = mp.get_context("spawn")
    request_queue = context.Queue(maxsize=max(256, worker_count * 32))
    response_queues = [context.Queue(maxsize=8) for _ in range(worker_count)]
    control_queue = context.Queue()
    result_queue = context.Queue()
    task_queue = context.Queue()
    server_devices = [
        gpu for gpu in gpus for _ in range(max(1, int(args.inference_replicas_per_gpu)))
    ]
    server_stats = [context.Array("d", STAT_COUNT, lock=False) for _ in server_devices]
    servers = [
        context.Process(
            target=inference_server_main,
            kwargs={
                "server_id": index,
                "gpu": gpu,
                "checkpoint": str(args.checkpoint),
                "request_queue": request_queue,
                "response_queues": response_queues,
                "control_queue": control_queue,
                "stats": server_stats[index],
                "max_batch_size": args.max_inference_batch_size,
                "max_batch_delay_ms": args.max_inference_delay_ms,
                "cache_capacity": args.inference_cache_capacity,
            },
            name=f"targeted-inference-gpu-{gpu}-{index}",
        )
        for index, gpu in enumerate(server_devices)
    ]
    for server in servers:
        server.start()
    ready = 0
    deadline = time.monotonic() + 300.0
    while ready < len(servers):
        event = control_queue.get(timeout=max(0.1, deadline - time.monotonic()))
        print(event, flush=True)
        if event["event"] == "inference_server_error":
            raise RuntimeError(event["error"])
        ready += int(event["event"] == "inference_server_ready")

    args_payload = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    for task in tasks:
        task_queue.put(task)
    for _ in range(worker_count):
        task_queue.put(None)
    workers = [
        context.Process(
            target=_worker_main,
            args=(index, args_payload, task_queue, request_queue, response_queues[index], result_queue),
            name=f"targeted-search-worker-{index:03d}",
        )
        for index in range(worker_count)
    ]
    started = time.perf_counter()
    for worker in workers:
        worker.start()
    roots: list[dict[str, Any]] = list(resumed_roots)
    summaries: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        while any(worker.is_alive() for worker in workers):
            try:
                event = result_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if event["event"] == "root_complete":
                roots.append(event)
                atomic_json(
                    root_result_path(
                        args.output_dir,
                        root_key(
                            event["seed"],
                            event.get("game_occurrence", 0),
                            event["action_index"],
                        ),
                    ),
                    event,
                )
                print(event, flush=True)
            elif event["event"] == "worker_done":
                summaries.append(event["summary"])
            elif event["event"] == "worker_error":
                errors.append(f"worker {event['worker_id']}: {event['error']}")
        for worker in workers:
            worker.join(timeout=5)
        while True:
            try:
                event = result_queue.get_nowait()
            except queue.Empty:
                break
            if event["event"] == "root_complete":
                roots.append(event)
                atomic_json(
                    root_result_path(
                        args.output_dir,
                        root_key(
                            event["seed"],
                            event.get("game_occurrence", 0),
                            event["action_index"],
                        ),
                    ),
                    event,
                )
            elif event["event"] == "worker_done":
                summaries.append(event["summary"])
            elif event["event"] == "worker_error":
                errors.append(f"worker {event['worker_id']}: {event['error']}")
        failed = [f"{worker.name} exit={worker.exitcode}" for worker in workers if worker.exitcode]
        if errors or failed:
            raise RuntimeError("; ".join(errors + failed))
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for _ in servers:
            request_queue.put(None)
        for server in servers:
            server.join(timeout=30)
            if server.is_alive():
                server.terminate()

    elapsed = time.perf_counter() - started
    replay_manifest = merge_manifests(args.output_dir, summaries)
    summary = build_generation_summary(
        args,
        positions=positions,
        tasks=tasks,
        worker_count=worker_count,
        roots=roots,
        replay_manifest=replay_manifest,
        elapsed=elapsed,
        server_devices=server_devices,
        server_stats=server_stats,
        resumed_roots=len(resumed_roots),
    )
    write_generation_outputs(args.output_dir, summary, roots)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def build_generation_summary(
    args: Any,
    *,
    positions: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    worker_count: int,
    roots: list[dict[str, Any]],
    replay_manifest: Path,
    elapsed: float,
    server_devices: list[str],
    server_stats: list[Any],
    resumed_roots: int,
) -> dict[str, Any]:
    simulations = sum(int(root["simulations"]) for root in roots)
    new_roots = roots[int(resumed_roots) :]
    new_simulations = sum(int(root["simulations"]) for root in new_roots)
    return {
        "checkpoint": str(args.checkpoint),
        "position_manifest": str(args.position_manifest.expanduser().resolve()),
        "positions_requested": len(positions),
        "tasks": len(tasks),
        "worker_count": worker_count,
        "resumed_roots": int(resumed_roots),
        "new_roots_completed": len(new_roots),
        "roots_completed": len(roots),
        "complete_teacher_roots": sum(bool(root["search_complete"]) for root in roots),
        "examples": sum(int(root["examples"]) for root in roots),
        "simulations": simulations,
        "simulations_per_second": new_simulations / max(elapsed, 1e-9),
        "roots_per_hour": len(new_roots) * 3600.0 / max(elapsed, 1e-9),
        "elapsed_seconds": elapsed,
        "side_searches": dict(Counter(root["side"] for root in roots)),
        "replay_manifest": str(replay_manifest.resolve()),
        "inference_servers": [
            inference_stats_payload(index, gpu, server_stats[index])
            for index, gpu in enumerate(server_devices)
        ],
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
    }


def write_generation_outputs(
    output_dir: Path, summary: dict[str, Any], roots: list[dict[str, Any]]
) -> None:
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "roots.jsonl").write_text(
        "".join(json.dumps(root, sort_keys=True) + "\n" for root in roots), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
