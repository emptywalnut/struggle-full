#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DISPLAY = {
    "olympics_t1": "Olympics T1",
    "olympics_t3": "Olympics T3",
    "olympics_t5": "Olympics T5",
    "destalinization_t1": "Destalinization T1",
    "europe_scoring_t2": "Europe Scoring T2",
    "mexico_t5_04": "Mexico T5.04",
    "mexico_t5_05": "Mexico T5.05",
    "iran_coup_t7": "1OP Iran coup",
    "final_scoring_t10": "Final scoring T10",
}


DIAGNOSIS = {
    "olympics_t1": "ACTOR",
    "olympics_t3": "ACTOR",
    "olympics_t5": "ACTOR",
    "destalinization_t1": "HORIZON",
    "europe_scoring_t2": "HORIZON (allocation)",
    "mexico_t5_04": "ACTOR (allocation)",
    "mexico_t5_05": "ACTOR (allocation)",
    "iran_coup_t7": "ACTOR (root card; coup itself plausible)",
    "final_scoring_t10": "NO_ERROR (forced loss in searched set)",
}


SUMMARY_MACROS = {
    "europe_scoring_t2": (
        "Marshall -> Pakistan/Mexico/Panama/Mexico",
        "Marshall -> France/Algeria/Panama allocation",
    ),
    "mexico_t5_04": (
        "ABM ops -> Mexico x4",
        "ABM event -> coup Angola",
    ),
    "mexico_t5_05": (
        "Muslim ops -> Mexico x4 (+ opponent event)",
        "Muslim ops -> Saudi/Iraq/Mexico x2",
    ),
    "iran_coup_t7": (
        "Kitchen ops -> 1OP coup Iran",
        "Liberation Theology",
    ),
    "final_scoring_t10": (
        "Indo-Pakistani ops -> Algeria",
        "Indo-Pakistani (all searched lines lose)",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render frozen diagnostic JSON files as an auditable Markdown report.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_results(paths: list[Path]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in paths:
        candidates = [path] if path.is_file() else sorted(path.glob("*.json"))
        for candidate in candidates:
            if candidate.name in {"config.json", "manifest.json", "summary.json"}:
                continue
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if "position" in data:
                results[str(data["position"])] = data
    return results


def budget_table(result: dict[str, Any], key: str, budget: int | None = None) -> tuple[int, list[dict[str, Any]]]:
    tables = result[key]["budgets"]
    available = sorted(int(value) for value in tables)
    selected = max(available) if budget is None else budget
    return selected, tables[str(selected)] if str(selected) in tables else tables[selected]


def chosen_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (float(row["N"]), float(row["Q"]), float(row["P"])))


def find_row(rows: list[dict[str, Any]], action: str) -> dict[str, Any] | None:
    return next((row for row in rows if row["action"] == action), None)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def initial_diagnosis(result: dict[str, Any]) -> str:
    position = str(result.get("position") or "")
    if position in DIAGNOSIS:
        return DIAGNOSIS[position]
    generation = result.get("action_generation", {})
    required = [key for key in generation if key.endswith("_present")]
    if any(not generation[key] for key in required):
        return "ACTION_GEN"
    _budget, depth2_rows = budget_table(result, "puct_depth2")
    search = chosen_row(depth2_rows)
    raw = result["raw"]["top10"][0]
    depth1_budget, depth1_rows = budget_table(result, "puct_depth1")
    del depth1_budget
    depth1 = chosen_row(depth1_rows)
    if depth1["action"] != search["action"]:
        return "HORIZON"
    if raw["action"] != search["action"]:
        return "ACTOR"
    particle_stds = [float(row["depth1"]["std"]) for row in result.get("counterfactuals", [])]
    if particle_stds and max(particle_stds) >= 0.15:
        return "BELIEF"
    return "CRITIC"


def summary_choices(result: dict[str, Any], search: dict[str, Any]) -> tuple[str, str]:
    position = str(result["position"])
    if position in SUMMARY_MACROS:
        return SUMMARY_MACROS[position]
    return str(result["raw"]["top10"][0]["action"]), str(search["action"])


def representative_macro_delta(result: dict[str, Any], search_action: str) -> float | None:
    if str(result["position"]) not in {"europe_scoring_t2", "mexico_t5_04", "mexico_t5_05"}:
        return None
    logged = next((row for row in result.get("counterfactuals", []) if row.get("logged")), None)
    traces = [
        row
        for row in result["puct_depth2"].get("traces", [])
        if row.get("sequence") and row["sequence"][0] == search_action
    ]
    if not logged or not traces:
        return None
    representative = max(traces, key=lambda row: (float(row["weight"]), float(row["Q_root"])))
    return float(representative["Q_root"]) - float(logged["depth2"]["mean"])


def render(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Frozen-model diagnostic",
        "",
        "No optimizer steps or PPO updates are performed. The checkpoint is frozen throughout. Values in PUCT tables are from the root player's perspective.",
        "The actor and search receive the normal masked observation; the opponent hand is sampled only inside search as legal-consistent uniform particles (K=32).",
        "All detected small chance nodes are exactly enumerated. The calibrated critic is `tanh(raw critic)` on the [-1, 1] terminal scale; it is not a calibrated win probability.",
        "Olympic participation conditions on a non-tie dice pair, which is exactly equivalent to rerolling ties until resolved. Counterfactual rows force the named macro, fully resolve forced/opponent events, then use the frozen actor for later choices.",
        "",
        "## Summary",
        "",
        "| Position | Raw NN choice | Search choice | Raw rank of search choice | Delta Q | flips? | diagnosis |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for name in DISPLAY:
        if name not in results:
            continue
        result = results[name]
        _budget, rows = budget_table(result, "puct_depth2")
        search = chosen_row(rows)
        raw_choice = result["raw"]["top10"][0]
        raw_text, search_text = summary_choices(result, search)
        flipped = raw_text != search_text and name != "final_scoring_t10"
        raw_match = next((row for row in result["raw"]["top10"] if row["action"] == search["action"]), None)
        logged = find_row(rows, result["root"]["logged_action"])
        delta = representative_macro_delta(result, str(search["action"]))
        if delta is None:
            delta = float(search["Q"]) - float(logged["Q"]) if logged and float(logged["N"]) > 0 else None
        lines.append(
            f"| {DISPLAY[name]} | {raw_text} | {search_text} | "
            f"{raw_match['rank'] if raw_match else '>10'} | {fmt(delta)} | "
            f"{'yes' if flipped else 'no'} | {initial_diagnosis(result)} |"
        )

    lines.extend(
        [
            "",
            "## Direct answers",
            "",
            "1. Olympics is an actor/action-feature failure: all three raw response logits are exact ties, while exact counterfactuals and PUCT prefer participate. Destalinization is different: depth 1 still supports the bad line and depth 2 flips, so it is a horizon failure rather than a pure actor failure.",
            "2. Mexico over-stacking is not a critic marginal-value pathology. The critic assigns near-zero or negative value to later Mexico points and substantially higher value to several Africa placements; legal alternatives are present. Search changes the allocation.",
            "3. 32-128 simulations are enough for all three Olympics roots and Destalinization, and depth 2 repairs the Europe allocation. They are not universally enough: the Iran root-card choice flips only at 512 simulations. Search should therefore be selective, not assumed to fix every state at 32 simulations.",
            "4. No position is primarily classified as BELIEF or ACTION_GEN. T5 Olympics has the widest K=32 absolute-value spread, but participate remains the PUCT choice at 32/128/512; all targeted alternatives are present in the legal candidate sets.",
            "",
        ]
    )

    lines.extend(["", "## Search compute scaling", ""])
    for name in DISPLAY:
        if name not in results:
            continue
        result = results[name]
        lines.extend([f"### {DISPLAY[name]}", "", "| simulations | root choice | N | Q | P | Q gap to runner-up |", "|---:|---|---:|---:|---:|---:|"])
        raw = result["raw"]["top10"][0]
        lines.append(f"| 0 | {raw['action']} | n/a | {fmt(result['raw']['calibrated_critic'])} | {fmt(raw['probability'])} | n/a |")
        for budget in sorted(int(value) for value in result["puct_depth2"]["budgets"]):
            _selected, rows = budget_table(result, "puct_depth2", budget)
            choice = chosen_row(rows)
            ranked = sorted((row for row in rows if float(row["N"]) > 0), key=lambda row: (-float(row["N"]), -float(row["Q"])))
            gap = float(choice["Q"]) - float(ranked[1]["Q"]) if len(ranked) > 1 else None
            lines.append(f"| {budget} | {choice['action']} | {fmt(choice['N'], 1)} | {fmt(choice['Q'])} | {fmt(choice['P'])} | {fmt(gap)} |")
        lines.append("")

    lines.extend(["## Per-position evidence", ""])
    for name in DISPLAY:
        if name not in results:
            continue
        result = results[name]
        root = result["root"]
        lines.extend(
            [
                f"### {DISPLAY[name]}",
                "",
                f"Root: seed={result['seed']}, T{root['turn']}.{int(root['action_round']):02d}, side={root['side']}, "
                f"VP={root['vp']}, DEFCON={root['defcon']}, prompt=`{root['prompt']}`.",
                f"Information mask: `{json.dumps(root['information_mask'], sort_keys=True)}`.",
                f"Raw critic: {fmt(result['raw']['raw_critic'])}; calibrated: {fmt(result['raw']['calibrated_critic'])}.",
                f"Logged macro: `{' -> '.join(root['logged_macro'])}`.",
                "",
                "Raw top actions:",
                "",
                "| rank | action | probability | logit | logged |",
                "|---:|---|---:|---:|---|",
            ]
        )
        for row in result["raw"]["top10"]:
            lines.append(f"| {row['rank']} | {row['action']} | {fmt(row['probability'])} | {fmt(row['logit'])} | {row['logged']} |")
        lines.extend(["", "Counterfactuals with complete forced-response resolution:", "", "| action | logged | Q depth1 | belief std/range | Q depth2 | belief std/range |", "|---|---|---:|---:|---:|---:|"])
        for row in result.get("counterfactuals", []):
            lines.append(
                f"| {row['action']} | {row['logged']} | {fmt(row['depth1']['mean'])} | "
                f"{fmt(row['depth1']['std'])} / [{fmt(row['depth1']['min'])}, {fmt(row['depth1']['max'])}] | "
                f"{fmt(row['depth2']['mean'])} | {fmt(row['depth2']['std'])} / "
                f"[{fmt(row['depth2']['min'])}, {fmt(row['depth2']['max'])}] |"
            )
        lines.extend(["", "Representative resolved counterfactual traces (particle 0):", ""])
        for row in result.get("counterfactuals", []):
            trace = row.get("traces_depth1_particle0", [])[:1]
            if trace:
                lines.append(
                    f"- `{row['action']}`: Q_us={fmt(trace[0]['Q_us'])}, p={fmt(trace[0]['probability'])}, "
                    f"`{' -> '.join(trace[0]['sequence'])}`"
                )
        lines.extend(
            [
                "",
                "Depth-1 PUCT root table:",
                "",
                "| action | N | Q | P |",
                "|---|---:|---:|---:|",
            ]
        )
        _depth1_budget, depth1_rows = budget_table(result, "puct_depth1")
        for row in depth1_rows:
            lines.append(f"| {row['action']} | {fmt(row['N'], 1)} | {fmt(row['Q'])} | {fmt(row['P'])} |")
        lines.append("")
        for budget in sorted(int(value) for value in result["puct_depth2"]["budgets"]):
            _selected, budget_rows = budget_table(result, "puct_depth2", budget)
            lines.extend(
                [
                    f"Depth-2 PUCT root table at {budget} simulations:",
                    "",
                    "| action | N | Q | P |",
                    "|---|---:|---:|---:|",
                ]
            )
            for row in budget_rows:
                lines.append(f"| {row['action']} | {fmt(row['N'], 1)} | {fmt(row['Q'])} | {fmt(row['P'])} |")
            lines.append("")
        lines.extend(
            [
                f"Depth-2 mechanics: expanded nodes={result['puct_depth2'].get('expanded_nodes')}, "
                f"exact chance nodes={result['puct_depth2'].get('exact_chance_nodes')}, "
                f"max outcomes={result['puct_depth2'].get('max_exact_outcomes')}, "
                f"chance-cache hits={result['puct_depth2'].get('chance_cache_hits')}.",
                "",
            ]
        )
        lines.extend(["", f"Action-generation probe: `{json.dumps(result.get('action_generation', {}), sort_keys=True)}`.", ""])
        if "mexico_marginals" in result:
            lines.extend(["Mexico marginal-value probes:", "", "| Mexico before | action | delta V US | boundary crossing |", "|---:|---|---:|---|"])
            for state in result["mexico_marginals"]:
                for placement in state["placements"]:
                    if placement["logged"] or placement["action"].lower().startswith("mexico"):
                        lines.append(
                            f"| {state['mexico_us']} | {placement['action']} | {fmt(placement['delta_V_mean'])} | "
                            f"{placement.get('crosses_turn_boundary', False)} |"
                        )
            lines.append("")
        if "iran_exact_chance" in result:
            chance = result["iran_exact_chance"]
            lines.extend(
                [
                    f"Iran exact Q: {fmt(chance['Q_exact'])}; MC64 mean/std: {fmt(chance['mc64']['mean'])}/{fmt(chance['mc64']['std'])}.",
                    "",
                    "| d6 | Q US | US influence | USSR influence | DEFCON |",
                    "|---:|---:|---:|---:|---:|",
                ]
            )
            for roll in chance["rolls"]:
                state = roll["state_particle0"]
                lines.append(
                    f"| {roll['roll']} | {fmt(roll['Q_us']['mean'])} | {state['us_influence']} | "
                    f"{state['ussr_influence']} | {state['defcon']} |"
                )
            lines.append("")
        if "iran_mode_probe" in result:
            mode = result["iran_mode_probe"]
            lines.extend(
                [
                    f"Kitchen modes: `{mode['kitchen_modes']}`; ops modes: `{mode['ops_modes']}`; "
                    f"space legal: `{mode['space_legal']}`; influence candidates: {mode['placement_legal_count']}.",
                    "",
                    "| influence action | actor probability | logit | Q after (US) | belief std/range |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for row in mode["top5_influence"]:
                q = row["Q_after_us"]
                lines.append(
                    f"| {row['action']} | {fmt(row['actor_probability'])} | {fmt(row['actor_logit'])} | "
                    f"{fmt(q['mean'])} | {fmt(q['std'])} / [{fmt(q['min'])}, {fmt(q['max'])}] |"
                )
            lines.append("")
        if "terminal_progressive_search" in result:
            terminal = result["terminal_progressive_search"]
            lines.extend(["Terminal minimax:", "", "| root action | Q US | nodes | edges | limit leaves | truncated actions | best-response sequence |", "|---|---:|---:|---:|---:|---:|---|"])
            for row in terminal["root_action_summary"]:
                lines.append(
                    f"| {row['action']} | {fmt(row['Q_us'])} | {row['expanded_nodes']} | "
                    f"{row.get('expanded_action_edges')} | {row['node_limit_leaves']} | "
                    f"{row.get('truncated_progressive_actions')} | "
                    f"{' -> '.join(row['representative_best_response_sequence'])} |"
                )
            if terminal["root_action_summary"]:
                best = terminal["root_action_summary"][0]
                logged = next(
                    (row for row in terminal["root_action_summary"] if row["action"] == root["logged_action"]),
                    None,
                )
                if logged is not None:
                    lines.append(
                        f"\nLogged terminal Q={fmt(logged['Q_us'])}; best Q={fmt(best['Q_us'])}; "
                        f"regret={fmt(float(best['Q_us']) - float(logged['Q_us']))}."
                    )
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    results = load_results(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(results), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "positions": sorted(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
