#!/usr/bin/env python3
"""Non-interactive paired evaluator for the explicitly qualified proxy.

Dry-run is the default. Games start only with ``--execute``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Any

from struggle_ai.baselines.steam_builtin import BridgeProxySearchRunner, SteamProxyConfig
from struggle_ai.bridge_client import TwilightBridgeClient
from struggle_ai.policies import choose_heuristic_action


def build_pair_schedule(seed_base: int, seed_pairs: int) -> list[dict[str, Any]]:
    return [
        {"pair": pair, "seed": seed_base + pair, "proxy_side": side}
        for pair in range(seed_pairs)
        for side in ("us", "ussr")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired evaluation of Steam-reconstructed proxy; no Steam parity claim")
    parser.add_argument("--execute", action="store_true", help="actually play games; default only prints the schedule")
    parser.add_argument("--opponent", choices=("heuristic", "random", "proxy"), default="heuristic")
    parser.add_argument("--seed-base", type=int, default=91_000_000)
    parser.add_argument("--seed-pairs", type=int, default=10)
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def play_game(entry: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    seed, proxy_side = int(entry["seed"]), str(entry["proxy_side"])
    rng = random.Random(seed ^ 0x5A17)
    config = SteamProxyConfig(seed=seed, search_state_depth=args.search_depth,
                              max_candidates=args.max_candidates)
    started = time.perf_counter()
    expanded = decisions = 0
    with TwilightBridgeClient() as bridge:
        proxy = BridgeProxySearchRunner(bridge, config)
        opponent_proxy = BridgeProxySearchRunner(bridge, config)
        obs = bridge.reset(seed)
        while not obs["terminal"] and decisions < args.max_steps:
            side = str(obs.get("current_player") or obs.get("side"))
            legal = obs["legal_actions"]
            if side == proxy_side:
                decision = proxy.choose_action(obs)
                action_index = decision.action_index
                expanded += decision.expanded_nodes
            elif args.opponent == "proxy":
                decision = opponent_proxy.choose_action(obs)
                action_index = decision.action_index
                expanded += decision.expanded_nodes
            elif args.opponent == "random":
                action_index = rng.randrange(len(legal))
            else:
                action_index = choose_heuristic_action(legal, rng, obs)
            obs = bridge.step(legal[action_index])["observation"]
            decisions += 1
    winner = str(obs.get("winner") or "")
    return {
        **entry,
        "label": "Steam-reconstructed proxy",
        "opponent": args.opponent,
        "winner": winner,
        "proxy_score": 1.0 if winner == proxy_side else 0.0 if winner in {"us", "ussr"} else 0.5,
        "terminal": bool(obs.get("terminal")),
        "decisions": decisions,
        "expanded_nodes": expanded,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    schedule = build_pair_schedule(args.seed_base, args.seed_pairs)
    if not args.execute:
        print(json.dumps({
            "status": "dry_run_no_games_started",
            "policy": "Steam-reconstructed proxy",
            "ruleset": "optional_us_plus_2",
            "games": len(schedule),
            "schedule": schedule,
        }, indent=2))
        return
    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, len(schedule)),
            mp_context=mp.get_context("spawn"),
        ) as pool:
            results = list(pool.map(play_game, schedule, [args] * len(schedule)))
    else:
        results = [play_game(entry, args) for entry in schedule]
    payload = {
        "policy": "Steam-reconstructed proxy",
        "ruleset": "optional_us_plus_2",
        "summary": {
            "games": len(results),
            "terminal_games": sum(bool(result["terminal"]) for result in results),
            "proxy_score": sum(float(result["proxy_score"]) for result in results) / len(results),
            "average_wall_seconds": sum(float(result["wall_seconds"]) for result in results) / len(results),
        },
        "games": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
