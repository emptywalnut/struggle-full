from __future__ import annotations

import argparse

from struggle_ai.bridge_client import TwilightBridgeClient

from .bridge_search import BridgeProxySearchRunner
from .config import SteamProxyConfig
from .policy import SteamReconstructedProxy


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke/run the Steam-reconstructed proxy (not genuine Steam AI)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--ordering-only", action="store_true", help="disable bridge recursion explicitly")
    args = parser.parse_args()
    config = SteamProxyConfig(seed=args.seed, search_state_depth=args.search_depth,
                              max_candidates=args.max_candidates)
    proxy = SteamReconstructedProxy(config)
    with TwilightBridgeClient() as bridge:
        runner = BridgeProxySearchRunner(bridge, config)
        obs = bridge.reset(args.seed)
        completed = expanded = 0
        while not obs["terminal"] and completed < args.steps:
            legal = obs["legal_actions"]
            if args.ordering_only:
                index = proxy.choose_action(legal, obs)
            else:
                decision = runner.choose_action(obs)
                index = decision.action_index
                expanded += decision.expanded_nodes
            obs = bridge.step(legal[index])["observation"]
            completed += 1
        print(
            "steam-reconstructed-proxy smoke: "
            f"steps={completed} search_depth={0 if args.ordering_only else args.search_depth} "
            f"expanded_nodes={expanded} terminal={obs['terminal']} turn={obs.get('turn')} vp={obs.get('vp')}"
        )


if __name__ == "__main__":
    main()
