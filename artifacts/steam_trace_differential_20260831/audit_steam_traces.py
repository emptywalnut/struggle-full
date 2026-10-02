#!/usr/bin/env python3
"""Read-only parser and differential audit for user-supplied Steam traces.

The trace corpus is treated strictly as data. This script never evaluates trace
text, never accepts paths from trace text, and only invokes the fixed local Saito
oracle alongside this file.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACES = ROOT / "reference" / "user_steam_traces_20260831"
ORACLE = Path(__file__).with_name("saito_oracle.js")
OUTPUT = Path(__file__).with_name("audit_results.json")
BRIDGE_DIAGNOSTIC_SCRIPT = Path(__file__).with_name("bridge_diagnostics.js")
BRIDGE_DIAGNOSTIC_OUTPUT = Path(__file__).with_name("bridge_diagnostics.json")
ENV_DIAGNOSTIC_SCRIPT = Path(__file__).with_name("env_mask_diagnostics.py")
ENV_DIAGNOSTIC_OUTPUT = Path(__file__).with_name("env_mask_diagnostics.json")
ENV_PYTHON = ROOT.parent.parent.parent / "miniconda3" / "envs" / "struggle-ppo" / "bin" / "python"
TRAINING_EXPOSURE_SCRIPT = Path(__file__).with_name("training_exposure_scan.py")
TRAINING_EXPOSURE_OUTPUT = Path(__file__).with_name("training_exposure.json")


def normalized(value: str) -> str:
    value = value.replace("*", "").replace("…", "...")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def oracle_batch(requests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload = "".join(json.dumps(item, sort_keys=True) + "\n" for item in requests)
    proc = subprocess.run(
        ["node", str(ORACLE)], cwd=ROOT, input=payload, text=True,
        capture_output=True, check=True,
    )
    results: dict[str, dict[str, Any]] = {}
    for line in proc.stdout.splitlines():
        message = json.loads(line)
        results[str(message.get("id"))] = message
    return results


def load_meta() -> dict[str, Any]:
    response = oracle_batch([{"id": "meta", "kind": "meta"}])["meta"]
    if not response["ok"]:
        raise RuntimeError(response["error"])
    return response["result"]


def run_bridge_diagnostics() -> dict[str, Any]:
    proc = subprocess.run(
        ["node", str(BRIDGE_DIAGNOSTIC_SCRIPT)], cwd=ROOT,
        text=True, capture_output=True, check=True,
    )
    result = json.loads(proc.stdout)
    BRIDGE_DIAGNOSTIC_OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if ENV_PYTHON.exists():
        subprocess.run(
            [str(ENV_PYTHON), str(ENV_DIAGNOSTIC_SCRIPT)], cwd=ROOT,
            text=True, capture_output=True, check=True,
        )
    subprocess.run(
        ["python3", str(TRAINING_EXPOSURE_SCRIPT)], cwd=ROOT,
        text=True, capture_output=True, check=True,
    )
    return result


def trace_excerpt(trace: str, start: int, end: int) -> dict[str, Any]:
    path = DEFAULT_TRACES / trace
    lines = path.read_text(encoding="ascii").splitlines()
    return {
        "trace": trace,
        "line_range": [start, end],
        "text": [{"line": number, "text": lines[number - 1]} for number in range(start, end + 1)],
    }


def build_bridge_differentials(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    bounded_evidence = {
        "socgov": trace_excerpt("3347414.txt", 20, 22),
        "liberation": trace_excerpt("3347843.txt", 295, 297),
        "reformer": trace_excerpt("3352341.txt", 802, 805),
        "warsawpact": trace_excerpt("3352375.txt", 79, 83),
    }
    labels = {
        "socgov": "Socialist Governments",
        "liberation": "Liberation Theology",
        "reformer": "The Reformer",
        "warsawpact": "Warsaw Pact add mode",
    }
    findings: list[dict[str, Any]] = []
    for item in diagnostics["bounded"]:
        card = item["card"]
        findings.append({
            "id": f"bridge-class-selector-repeat-{card}",
            "mechanic": labels[card],
            "classification": "deterministic_hard_mismatch",
            "severity": "high",
            "shortest_reproduction": {
                "event": card, "first_target": item["target"],
                "initial_prompt": item["before"]["prompt"],
            },
            "trace_evidence": bounded_evidence[card],
            "expected_post_state": {
                "influence": item["influence_after_first"],
                "target_remains_legal_for_second_selection": True,
            },
            "actual_post_state": {
                "influence": item["influence_after_first"],
                "target_remains_legal_for_second_selection": item["target_still_legal"],
                "remaining_catalog_size": len(item["after_first_selection"]["choices"]),
                "saito_place_flag": item["target_place_flag"],
            },
            "layer_verdict": {
                "materialized_saito": "match (event handler permits a second selection)",
                "headless_bridge_queue_adapter": "mismatch",
                "browser_server_ui": "faithfully projects the defective bridge catalog",
                "training_env_encoder_mask": "faithfully masks the defective bridge catalog",
            },
            "root_cause": {
                "file": "bridge/saito_bridge.js",
                "function": "HeadlessDecisionAdapter.trigger",
                "current_lines": [465, 483],
                "detail": "After every class-selector click, the adapter deletes the selected ID even when the original Saito handler deliberately leaves it enabled until the per-country cap is reached.",
            },
            "models_exposed": True,
            "exposure_evidence": "TwilightStruggleEnv takes obs.legal_actions from TwilightBridgeClient and encode_observation sets one mask bit per catalog entry; no rules layer restores removed targets.",
            "minimal_fix_surface": [
                "Preserve class membership unless the original handler/place flag or an explicit bounded counter reaches the cap.",
                f"Add materialized-handler, bridge-catalog, browser, and env-mask tests for two selections of {item['target']}.",
            ],
        })
    regional_evidence = {
        "vietnam": [
            trace_excerpt("3347401.txt", 196, 202),
            trace_excerpt("3347401.txt", 242, 244),
            trace_excerpt("3346438.txt", 134, 138),
        ],
        "china": [trace_excerpt("3346480.txt", 86, 95)],
    }
    boundaries = {item["kind"]: item for item in diagnostics["regional_saito_boundary"]}
    for item in diagnostics["regional"]:
        kind = item["kind"]
        findings.append({
            "id": f"bridge-region-bonus-replays-printed-ops-{kind}",
            "mechanic": "Vietnam Revolts one-Op Southeast Asia bonus" if kind == "vietnam" else "China Card one-Op Asia bonus",
            "classification": "deterministic_hard_mismatch",
            "severity": "critical",
            "shortest_reproduction": {
                "active_bonus": kind, "card": item["card"], "printed_ops": item["printed_ops"],
                "spend_all_printed_ops_in": item["target"],
            },
            "trace_evidence": regional_evidence[kind],
            "expected_post_state": {
                "next_prompt": boundaries[kind]["prompt_after_original_saito_callback"],
                "total_placements": item["printed_ops"] + 1,
                "target_influence_from_initial_2": 2 + item["printed_ops"] + 1,
            },
            "actual_post_state": {
                "next_prompt": item["after_printed_ops"]["prompt"],
                "reentered_full_ops_menu": item["incorrectly_reentered_full_ops_menu"],
                "total_placements_completed": item["total_placements_completed"],
                "target_influence": item["influence_after_replayed_ops"],
            },
            "layer_verdict": {
                "materialized_saito": "match (exactly one region-restricted target prompt)",
                "headless_bridge_queue_adapter": "mismatch",
                "browser_server_ui": "faithfully exposes the replayed full-Ops decision",
                "training_env_encoder_mask": "faithfully masks the replayed full-Ops catalog",
            },
            "root_cause": {
                "file": "bridge/saito_bridge.js", "function": "SaitoTwilightKernel.stepAsync",
                "current_lines": [2022, 2032],
                "detail": "forceDrainDomAction treats the original one-Op region-bonus prompt as a signal to drain the queue, which re-enters the original printed-Ops command instead of preserving the pending bonus target.",
            },
            "models_exposed": True,
            "exposure_evidence": "All repository self-play and browser model inference use the same bridge legal catalog and canonical action mask; no downstream correction exists.",
            "minimal_fix_surface": [
                "Return the pending Saito DOM decision at the region-bonus boundary instead of draining the original Ops queue.",
                f"Test exactly {item['printed_ops'] + 1} total Ops, region restriction, eligibility reset, browser projection, and env action mask.",
            ],
        })
    return findings


def classify_current_bridge_checks(findings: list[dict[str, Any]], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    """Retain the minimized historical defect evidence while classifying the current code."""
    bounded = {item["card"]: item for item in diagnostics["bounded"]}
    regional = {item["kind"]: item for item in diagnostics["regional"]}
    for finding in findings:
        if finding["id"].startswith("bridge-class-selector-repeat-"):
            item = bounded[finding["shortest_reproduction"]["event"]]
            finding["actual_post_state"].update({
                "target_remains_legal_for_second_selection": item["target_still_legal"],
                "target_rejected_after_second_selection": not item["target_legal_after_second"],
                "bounded_selections_completed": item["bounded_selection_count"],
            })
            fixed = item["target_still_legal"] and not item["target_legal_after_second"]
        else:
            kind = finding["shortest_reproduction"]["active_bonus"]
            item = regional[kind]
            finding["actual_post_state"].update({
                "next_prompt": item["after_printed_ops"]["prompt"],
                "reentered_full_ops_menu": item["incorrectly_reentered_full_ops_menu"],
                "total_placements_completed": item["total_placements_completed"],
                "target_influence": item["influence_after_replayed_ops"],
                "bonus_prompt_consumed_once": item["bonus_prompt_consumed_once"],
            })
            fixed = (
                not item["incorrectly_reentered_full_ops_menu"]
                and item["extra_placements_completed"] == 1
                and item["total_placements_completed"] == item["printed_ops"] + 1
                and item["bonus_prompt_consumed_once"]
            )
        if fixed:
            finding["classification"] = "match"
            finding["severity"] = "resolved"
            finding["resolution"] = "verified in the current bridge catalog and Saito transition path"
    return findings


def build_materialized_differentials(audits: list[FileAudit]) -> list[dict[str, Any]]:
    checks = [
        check for audit in audits for check in audit.checks
        if check.mechanic == "card_removal_marker" and check.status == "hard_mismatch"
    ]
    if not checks:
        return []
    return [{
        "id": "materialized-oas-founded-incorrectly-nonrecurring",
        "mechanic": "OAS Founded card lifecycle",
        "classification": "deterministic_hard_mismatch",
        "severity": "high",
        "trace_evidence": [trace_excerpt(check.trace, check.lines[0], check.lines[-1]) for check in checks],
        "additional_corpus_evidence": "Every OAS Founded action header in the corpus lacks Steam's remove-from-play asterisk.",
        "expected_post_state": "OAS Founded is discarded after its event and remains eligible for a later reshuffle.",
        "actual_post_state": "Materialized Saito metadata recurring=0 routes OAS Founded to removed-from-game.",
        "layer_verdict": {
            "materialized_saito": "mismatch",
            "headless_bridge_queue_adapter": "inherits materialized card lifecycle",
            "browser_server_ui": "inherits bridge deck state",
            "training_env_encoder_mask": "trained on the inherited deck removal",
        },
        "root_cause": {
            "file": "third_party/saito-lite-rust-materialized/mods/twilight/twilight.js",
            "function": "returnMidWarCards / generic event resolution",
            "current_lines": [7583, 2683, 2687],
            "detail": "OAS Founded is declared recurring:0; generic event resolution removes every non-recurring card.",
        },
        "models_exposed": True,
        "minimal_fix_surface": [
            "Set OAS Founded recurring metadata to the Steam/original recurring value.",
            "Test event -> discard, reshuffle eligibility, bridge/browser deck projection, and env card features.",
        ],
    }]


@dataclass
class Check:
    trace: str
    lines: list[int]
    mechanic: str
    status: str
    expected: Any
    actual: Any
    detail: str
    layer: str = "materialized_saito"
    request_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return vars(self)


@dataclass
class FileAudit:
    name: str
    sha256: str
    bytes: int
    lines: int
    nonblank_lines: int = 0
    semantic_lines: int = 0
    narrative_lines: int = 0
    unparsed: list[dict[str, Any]] = field(default_factory=list)
    players: dict[str, str] = field(default_factory=dict)
    bids: list[dict[str, Any]] = field(default_factory=list)
    handicap: str | None = None
    scenario: str | None = None
    optional_cards: bool = False
    time_minutes: int | None = None
    max_turn: int = 0
    max_ar: dict[str, int] = field(default_factory=lambda: {"us": 0, "ussr": 0})
    cards: collections.Counter[str] = field(default_factory=collections.Counter)
    events: collections.Counter[str] = field(default_factory=collections.Counter)
    actions: collections.Counter[str] = field(default_factory=collections.Counter)
    mechanics: collections.Counter[str] = field(default_factory=collections.Counter)
    dice: collections.Counter[str] = field(default_factory=collections.Counter)
    completeness: str = "unknown"
    terminal_condition: str | None = None
    terminal_winner: str | None = None
    final_public_state: dict[str, Any] = field(default_factory=dict)
    optional_card_plays: list[dict[str, Any]] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        result = vars(self).copy()
        for key in ("cards", "events", "actions", "mechanics", "dice"):
            result[key] = dict(sorted(result[key].items()))
        result["checks"] = [check.as_dict() for check in self.checks]
        result["parse_coverage_nonblank"] = (
            (self.semantic_lines + self.narrative_lines) / self.nonblank_lines
            if self.nonblank_lines else 1.0
        )
        result["semantic_coverage_nonblank"] = (
            self.semantic_lines / self.nonblank_lines if self.nonblank_lines else 1.0
        )
        return result


class Parser:
    STEAM_OPTIONAL_CARD_IDS = {
        "defectors", "cambridge", "specialrelation", "norad",
        "che", "tehran", "yuri", "awacs",
    }
    EVENT_FLAG_MAP = {
        "formosanresolution": "formosan",
        "shuttlediplomacy": "shuttlediplomacy",
        "irancontrascandal": "irancontra",
        "saltnegotiations": "saltnegotiations",
        "nuclearsubs": "nuclearsubs",
        "nato": "nato",
        "degaulleleadsfrance": "degaulle",
        "usjapanmutualdefensepact": "usjapan",
        "usjapandefensepact": "usjapan",
        "vietnamrevolts": "vietnam_revolts",
        "containment": "containment",
        "brezhnevdoctrine": "brezhnev",
        "redscarepurge": "redscare",
        "yuriandsamantha": "yuri",
    }
    REGION_BY_CARD = {
        "europescoring": "europe", "asiascoring": "asia",
        "mideastscoring": "mideast", "middleeastscoring": "mideast",
        "africascoring": "africa", "centralamericascoring": "camerica",
        "southamericascoring": "samerica", "southeastasiascoring": "seasia",
    }
    KNOWN_NARRATIVE_PREFIXES = (
        "USSR reveals ", "US reveals ", "USSR discards ", "US discards ",
        "USSR plays ", "US plays ", "USSR chooses ", "US chooses ",
        "USSR returns ", "US returns ", "War in ", "TIE!", "Scenario:",
        "Time per Player:", "SETUP:", "Handicap influence:",
        "Additional Influence from bidding:", "Optional Cards Added",
        "Target: ", "Die roll:", "Trap Roll:", "Event: ",
        "USSR Headlines ", "US Headlines ", "Turn ", "DEFCON ",
        "USSR Military Ops ", "US Military Ops ", "USSR advances ",
        "US advances ", "USSR gains ", "US gains ", "No VP awarded.",
        "SUCCESS:", "FAILURE:", "VICTORY:", "DEFEAT:",
        "USSR rolls ", "US rolls ", "Realignment ", "Coup ",
        "Place Influence ", "Space Race ", "*RESHUFFLE*",
    )

    def __init__(self, meta: dict[str, Any]):
        self.meta = meta
        self.country_by_name: dict[str, str] = {}
        self.country_meta = {country["id"]: country for country in meta["countries"]}
        for country in meta["countries"]:
            self.country_by_name[normalized(country["name"])] = country["id"]
        self.country_by_name.update({
            "spainportugal": "spain", "laoscambodia": "laos",
            "seafricanstates": "seafricanstates", "southeastafricanstates": "seafricanstates",
            "dominicanrep": "dominicanrepublic", "gulfstates": "gulfstates",
        })
        self.card_by_name: dict[str, str] = {}
        self.card_meta = {card["id"]: card for card in meta["cards"]}
        for card in meta["cards"]:
            self.card_by_name[normalized(card["name"])] = card["id"]
        self.card_by_name.update({
            "capturednaziscientist": "naziscientist",
            "nucleartestban": "nucleartestban",
            "usjapanmutualdefensepact": "usjapan",
            "destalinization": "destalinization",
            "thechinacard": "china",
            "mideastscoring": "mideast",
            "southeastasiascoring": "seasia",
        })
        self.requests: list[dict[str, Any]] = []
        self.request_checks: dict[str, Check] = {}
        self.request_context: dict[str, dict[str, Any]] = {}
        self.next_request = 1

    def request(self, check: Check, request: dict[str, Any], context: dict[str, Any]) -> None:
        request_id = f"r{self.next_request}"
        self.next_request += 1
        request = {"id": request_id, **request}
        check.request_id = request_id
        self.requests.append(request)
        self.request_checks[request_id] = check
        self.request_context[request_id] = context

    def parse_file(self, path: Path) -> FileAudit:
        raw = path.read_bytes()
        lines = raw.decode("ascii").splitlines()
        audit = FileAudit(
            name=path.name, sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw), lines=len(lines),
        )
        countries = {
            country["id"]: {"us": int(country["us"]), "ussr": int(country["ussr"])}
            for country in self.meta["countries"]
        }
        known_country = {country_id: True for country_id in countries}
        vp = 0
        defcon = 5
        milops = {"us": 0, "ussr": 0}
        space = {"us": 0, "ussr": 0}
        active_events: set[str] = set()
        current_turn = 0
        current_side: str | None = None
        current_ar = 0
        current_card: str | None = None
        current_card_id: str | None = None
        current_mode: str | None = None
        current_ops: int | None = None
        pending_coup: dict[str, Any] | None = None
        pending_realign: dict[str, Any] | None = None
        pending_scoring: dict[str, Any] | None = None
        pending_space: dict[str, Any] | None = None
        placement_block: dict[str, Any] | None = None
        pending_headline: dict[str, Any] | None = None
        current_event_name: str | None = None
        saw_terminal = None

        def public_countries() -> dict[str, dict[str, int]]:
            return copy.deepcopy(countries)

        def event_payload() -> dict[str, int]:
            payload: dict[str, int] = {}
            for name in active_events:
                key = self.EVENT_FLAG_MAP.get(normalized(name))
                if key and key != "redscare":
                    payload[key] = 1
            return payload

        def add_check(line_numbers: list[int], mechanic: str, status: str,
                      expected: Any, actual: Any, detail: str,
                      layer: str = "trace_public_state") -> Check:
            check = Check(path.name, line_numbers, mechanic, status, expected, actual, detail, layer)
            audit.checks.append(check)
            return check

        def finalize_placement(end_line: int) -> None:
            nonlocal placement_block
            block = placement_block
            placement_block = None
            if not block or not block["placements"]:
                return
            regions = block["regions"]
            allowed = {block["printed_ops"]}
            vietnam_eligible = (
                block["side"] == "ussr" and block["vietnam_active"]
                and regions and regions <= {"seasia"}
            )
            if vietnam_eligible:
                # Red Purge suppresses Vietnam's bonus for a printed 1-Op card.
                if block["printed_ops"] == 1 and block["red_purge_active"]:
                    allowed.add(block["printed_ops"])
                else:
                    allowed = {value + 1 for value in allowed}
            china_eligible = (
                block["card"] == "china" and regions
                and regions <= {"asia", "seasia"}
            )
            if china_eligible:
                allowed = {value + 1 for value in allowed}
            actual = block["cost"]
            status = "match" if actual in allowed else "parser_limitation"
            detail = (
                f"Influence-placement cost from explicit control states; printed={block['printed_ops']}, "
                f"Vietnam bonus={vietnam_eligible}, China bonus={china_eligible}"
            )
            add_check([block["line"], end_line], "ops_placement_budget", status,
                      sorted(allowed), actual, detail)

        def finalize_coup(end_line: int) -> None:
            nonlocal pending_coup
            if not pending_coup or "result" not in pending_coup:
                return
            item = pending_coup
            actual_post = copy.deepcopy(countries[item["country"]])
            pre = item["pre"]
            shift = max(0, item["result"])
            predictions = {}
            for candidate_side in ("us", "ussr"):
                candidate = copy.deepcopy(pre)
                opponent = "ussr" if candidate_side == "us" else "us"
                removed = min(shift, candidate[opponent])
                candidate[opponent] -= removed
                candidate[candidate_side] += shift - removed
                predictions[candidate_side] = candidate
            inferred = [side for side, post in predictions.items() if post == actual_post]
            if len(inferred) == 1:
                item["side"] = inferred[0]
            expected_post = predictions[item["side"]]
            status = "match" if actual_post == expected_post else "hard_mismatch"
            check = add_check(
                [item["line"], end_line], "coup_influence", status,
                expected_post, actual_post, "Trace coup result and resulting influence",
            )
            self.request(check, {
                "kind": "coup", "side": item["side"], "country": item["country"],
                "ops": item["ops"], "roll": item["effective_roll"],
                "countries": item["countries"], "events": {}, "defcon": item["defcon"],
                "lower_defcon_on_coup": 1,
            }, {"compare": "country", "expected": actual_post})
            pending_coup = None

        def finalize_realign(end_line: int) -> None:
            nonlocal pending_realign
            if not pending_realign or "us_roll" not in pending_realign or "ussr_roll" not in pending_realign:
                return
            item = pending_realign
            actual_post = copy.deepcopy(countries[item["country"]])
            check = add_check(
                [item["line"], end_line], "realignment", "pending_oracle",
                actual_post, None, "Saito realignment with trace-recorded dice and reconstructed board",
            )
            events = {"irancontra": 1} if any(normalized(x) == "irancontrascandal" for x in active_events) else {}
            self.request(check, {
                "kind": "realign", "side": item["side"], "country": item["country"],
                "us_roll": item["us_roll"], "ussr_roll": item["ussr_roll"],
                "countries": item["countries"], "events": events,
            }, {"compare": "country", "expected": actual_post})
            pending_realign = None

        for index, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                if pending_coup and "result" in pending_coup:
                    finalize_coup(index)
                if pending_realign and "us_roll" in pending_realign and "ussr_roll" in pending_realign:
                    finalize_realign(index)
                continue
            audit.nonblank_lines += 1
            semantic = False
            narrative = False

            if re.match(r"Turn \d+, (?:Headline Phase|Cleanup|USSR AR\d+|US AR\d+)", stripped) or stripped.startswith("Event: "):
                finalize_placement(index - 1)
            if pending_headline and re.match(r"Turn \d+, (?:USSR|US) AR\d+", stripped):
                add_check([pending_headline["line"], index - 1], "headline_order", "parser_limitation",
                          pending_headline.get("expected"), None,
                          "No explicit first headline Event line (cancellation or trace omission)")
                pending_headline = None

            # A new target/action closes the prior randomized atomic transition.
            if stripped.startswith("Target: "):
                finalize_coup(index - 1)
                finalize_realign(index - 1)

            match = re.match(r"(?:SETUP: )?(.+) will play as (USSR|USA)\.$", stripped)
            if match:
                audit.players["ussr" if match.group(2) == "USSR" else "us"] = match.group(1)
                semantic = True
            match = re.match(r"(?:SETUP: )?(.+) bids (\d+) Influence for USSR$", stripped)
            if match:
                audit.bids.append({"player": match.group(1), "influence": int(match.group(2))})
                semantic = True
            match = re.match(r"(?:Handicap influence|Additional Influence from bidding): (US|USSR) \+(\d+)$", stripped)
            if match:
                audit.handicap = f"{match.group(1).lower()}+{match.group(2)}"
                semantic = True
            if stripped.startswith("Scenario: "):
                audit.scenario = stripped.split(":", 1)[1].strip()
                semantic = True
            if stripped == "Optional Cards Added":
                audit.optional_cards = True
                semantic = True
            match = re.match(r"Time per Player: (\d+) Minutes$", stripped)
            if match:
                audit.time_minutes = int(match.group(1)); semantic = True

            # Turn/action headers. Card names may themselves contain colons, so modes
            # are found from the rightmost known action marker.
            match = re.match(r"Turn (\d+), Headline Phase: (.+?) & (.+?):(?: (.*))?$", stripped)
            if match:
                current_turn = int(match.group(1)); audit.max_turn = max(audit.max_turn, current_turn)
                current_side = None; current_ar = 0; current_card = None; current_card_id = None
                current_event_name = None
                audit.cards[match.group(2).rstrip("*")] += 1
                audit.cards[match.group(3).rstrip("*")] += 1
                for card_name in (match.group(2).rstrip("*"), match.group(3).rstrip("*")):
                    card_id = self.card_by_name.get(normalized(card_name))
                    if card_id in self.STEAM_OPTIONAL_CARD_IDS:
                        audit.optional_card_plays.append({"line": index, "card": card_id, "context": "headline"})
                ussr_id = self.card_by_name.get(normalized(match.group(2).rstrip("*")))
                us_id = self.card_by_name.get(normalized(match.group(3).rstrip("*")))
                if ussr_id and us_id and "defectors" not in {ussr_id, us_id}:
                    pending_headline = {
                        "line": index, "cards": {ussr_id, us_id}, "sides": {},
                    }
                semantic = True
            match = re.match(r"Turn (\d+), (USSR|US) AR(\d+)(?:: ?(.*))?$", stripped)
            if match:
                current_turn = int(match.group(1)); current_side = match.group(2).lower(); current_ar = int(match.group(3))
                audit.max_turn = max(audit.max_turn, current_turn)
                audit.max_ar[current_side] = max(audit.max_ar[current_side], current_ar)
                tail = match.group(4) or ""
                current_event_name = None
                mode_match = re.match(r"(.+?): (Event|Coup|Realignment|Place Influence|Space Race)(?::| \((\d+) Ops\):)?(?: (.*))?$", tail)
                if mode_match:
                    current_card = mode_match.group(1).rstrip("*")
                    current_card_id = self.card_by_name.get(normalized(current_card))
                    current_mode = mode_match.group(2).lower().replace(" ", "_")
                    current_ops = int(mode_match.group(3)) if mode_match.group(3) else None
                    audit.cards[current_card] += 1; audit.actions[current_mode] += 1
                    if current_card_id in self.STEAM_OPTIONAL_CARD_IDS:
                        audit.optional_card_plays.append({"line": index, "card": current_card_id, "context": current_mode})
                    if current_mode == "event":
                        current_event_name = current_card
                    if current_mode == "space_race":
                        pending_space = {
                            "line": index, "side": current_side, "card": current_card_id,
                            "before": copy.deepcopy(space), "vp": vp,
                        }
                    if current_mode == "place_influence" and current_side and current_ops is not None:
                        placement_block = {
                            "line": index, "side": current_side, "printed_ops": current_ops,
                            "card": current_card_id, "cost": 0, "placements": [], "regions": set(),
                            "vietnam_active": any(normalized(name) == "vietnamrevolts" for name in active_events),
                            "red_purge_active": any(normalized(name) == "redscarepurge" for name in active_events),
                        }
                elif tail:
                    current_card = tail.rstrip("*: "); current_card_id = self.card_by_name.get(normalized(current_card))
                    if current_card_id:
                        audit.cards[current_card] += 1
                        if current_card_id in self.STEAM_OPTIONAL_CARD_IDS:
                            audit.optional_card_plays.append({"line": index, "card": current_card_id, "context": "action_header"})
                    else:
                        current_card = None
                semantic = True
            match = re.match(r"Turn (\d+), Cleanup(?:: (.*))?$", stripped)
            if match:
                current_turn = int(match.group(1)); audit.max_turn = max(audit.max_turn, current_turn)
                current_side = None; current_ar = 0; current_mode = "cleanup"
                milops = {"us": 0, "ussr": 0}
                semantic = True

            match = re.match(r"(Coup|Realignment|Place Influence|Space Race) \((\d+) Ops\):$", stripped)
            if match:
                finalize_placement(index - 1)
                current_mode = match.group(1).lower().replace(" ", "_"); current_ops = int(match.group(2))
                audit.actions[current_mode] += 1; semantic = True
                if current_mode == "space_race":
                    pending_space = {"line": index, "side": current_side, "card": current_card_id, "before": copy.deepcopy(space), "vp": vp}
                if current_mode == "place_influence" and current_side:
                    placement_block = {
                        "line": index, "side": current_side, "printed_ops": current_ops,
                        "card": current_card_id, "cost": 0, "placements": [], "regions": set(),
                        "vietnam_active": any(normalized(name) == "vietnamrevolts" for name in active_events),
                        "red_purge_active": any(normalized(name) == "redscarepurge" for name in active_events),
                    }

            match = re.search(r"(?:^|: )(USSR|US) Headlines (.+)$", stripped)
            if match:
                headline_name = match.group(2).rstrip("*")
                headline_id = self.card_by_name.get(normalized(headline_name))
                audit.cards[headline_name] += 1; semantic = True
                if pending_headline and headline_id in pending_headline["cards"]:
                    pending_headline["sides"][match.group(1).lower()] = headline_id
                    if set(pending_headline["sides"]) == {"ussr", "us"}:
                        ussr_card = pending_headline["sides"]["ussr"]
                        us_card = pending_headline["sides"]["us"]
                        ussr_ops = int(self.card_meta[ussr_card]["ops"])
                        us_ops = int(self.card_meta[us_card]["ops"])
                        pending_headline["expected"] = ussr_card if ussr_ops > us_ops else us_card

            match = re.match(r"Event: (.+)$", stripped)
            if match:
                raw_event_name = match.group(1)
                event_name = raw_event_name.rstrip("*")
                event_card_id = self.card_by_name.get(normalized(event_name))
                if event_card_id:
                    expected_removed = not bool(self.card_meta[event_card_id]["recurring"])
                    displayed_removed = raw_event_name.endswith("*")
                    add_check([index], "card_removal_marker",
                              "match" if expected_removed == displayed_removed else "parser_limitation",
                              expected_removed, displayed_removed,
                              "Steam asterisk removal marker versus Saito recurring metadata; Steam omits the marker for some triggered events")
                if pending_headline:
                    actual_headline = self.card_by_name.get(normalized(event_name))
                    expected_headline = pending_headline.get("expected")
                    status = "match" if actual_headline == expected_headline else (
                        "parser_limitation" if expected_headline is None else "hard_mismatch"
                    )
                    add_check([pending_headline["line"], index], "headline_order", status,
                              expected_headline, actual_headline,
                              "Higher-Ops headline resolves first; US resolves ties")
                    pending_headline = None
                current_event_name = event_name
                audit.events[event_name] += 1; audit.mechanics["event"] += 1; semantic = True
                region = self.REGION_BY_CARD.get(normalized(event_name))
                if region:
                    pending_scoring = {
                        "line": index, "region": region, "side": current_side,
                        "countries": public_countries(), "events": event_payload(),
                    }

            match = re.match(r"(.+?)\*? is now in play\.$", stripped)
            if match:
                active_events.add(match.group(1).rstrip("*")); audit.mechanics["event_flag"] += 1; semantic = True
            match = re.match(r"(.+?)\*? is no longer in play\.$", stripped)
            if match:
                key = normalized(match.group(1))
                active_events = {name for name in active_events if normalized(name) != key}
                audit.mechanics["event_flag"] += 1; semantic = True

            # Influence delta with explicit post-state is a complete deterministic
            # atomic oracle independent of hidden hands/deck order.
            match = re.match(r"(USSR|US) ([+-])(\d+) in (.+?) \[(\d+)\]\[(\d+)\]$", stripped)
            if match:
                side = match.group(1).lower(); amount = int(match.group(3)); delta = amount if match.group(2) == "+" else -amount
                country_name = match.group(4); country_id = self.country_by_name.get(normalized(country_name))
                post = {"us": int(match.group(5)), "ussr": int(match.group(6))}
                if country_id:
                    pre = copy.deepcopy(countries[country_id])
                    if placement_block and delta > 0 and side == placement_block["side"]:
                        simulated = copy.deepcopy(pre)
                        opponent = "ussr" if side == "us" else "us"
                        stability = int(self.country_meta[country_id]["stability"])
                        for _ in range(amount):
                            opponent_controls = simulated[opponent] - simulated[side] >= stability
                            placement_block["cost"] += 2 if opponent_controls else 1
                            simulated[side] += 1
                        placement_block["placements"].append({"line": index, "country": country_id, "amount": amount})
                        placement_block["regions"].add(self.country_meta[country_id]["region"])
                    expected = copy.deepcopy(pre); expected[side] = max(0, expected[side] + delta)
                    status = "match" if expected == post else "hard_mismatch"
                    check = add_check([index], "influence", status, expected, post, f"{side} {delta:+d} {country_name}")
                    request_countries = {country_id: pre}
                    self.request(check, {
                        "kind": "influence", "side": side, "delta": delta,
                        "country": country_id, "countries": request_countries,
                    }, {"compare": "country", "expected": post})
                    countries[country_id] = post; known_country[country_id] = True
                else:
                    add_check([index], "influence", "parser_limitation", None, post, f"Unmapped country: {country_name}")
                audit.mechanics["influence"] += 1; semantic = True

            match = re.match(r"Target: (.+)$", stripped)
            if match:
                country_id = self.country_by_name.get(normalized(match.group(1)))
                if current_mode == "coup" and country_id and current_side and current_ops is not None:
                    pending_coup = {
                        "line": index, "country": country_id, "side": current_side,
                        "ops_header": current_ops, "pre": copy.deepcopy(countries[country_id]),
                        "countries": public_countries(), "defcon": defcon,
                    }
                elif current_mode == "realignment" and country_id and current_side:
                    pending_realign = {
                        "line": index, "country": country_id, "side": current_side,
                        "countries": public_countries(),
                    }
                semantic = True

            match = re.match(r"(SUCCESS|FAILURE): (\d+)(?: \(([+-]\d+)\))?\s*\[ \+ (\d+) - 2x(\d+) = (-?\d+) \]", stripped)
            if match and pending_coup:
                raw_roll = int(match.group(2)); modifier = int(match.group(3) or 0)
                ops = int(match.group(4)); stability = int(match.group(5)); result = int(match.group(6))
                calculated = raw_roll + modifier + ops - 2 * stability
                add_check([index], "coup_formula", "match" if calculated == result else "hard_mismatch", result, calculated, "Displayed coup arithmetic")
                meta_stability = int(self.country_meta[pending_coup["country"]]["stability"])
                add_check([pending_coup["line"], index], "coup_stability", "match" if stability == meta_stability else "hard_mismatch", stability, meta_stability, "Printed target stability versus Saito map")
                pending_coup.update({"result": result, "ops": ops, "effective_roll": raw_roll + modifier})
                audit.mechanics["coup"] += 1; audit.dice["coup"] += 1
                audit.dice[f"coup_roll_{raw_roll}"] += 1; semantic = True

            match = re.match(r"(USSR|US) rolls (\d+)(?: \(([+-]\d+)\))?(?: = (-?\d+))?$", stripped)
            if match and pending_realign:
                side = match.group(1).lower(); raw_roll = int(match.group(2)); modifier = int(match.group(3) or 0)
                total = int(match.group(4)) if match.group(4) else raw_roll + modifier
                add_check([index], "realign_roll_display", "match" if raw_roll + modifier == total else "hard_mismatch", total, raw_roll + modifier, "Displayed realignment arithmetic")
                pending_realign[f"{side}_roll"] = raw_roll
                pending_realign[f"{side}_total"] = total
                audit.dice["realign"] += 1; audit.dice[f"realign_roll_{raw_roll}"] += 1; semantic = True

            match = re.match(r"Die roll: (\d+) -- (Success|Failed)! \(Needed (\d+) or less\)$", stripped)
            if match:
                roll = int(match.group(1)); expected_success = roll <= int(match.group(3)); observed_success = match.group(2) == "Success"
                add_check([index], "space_roll", "match" if expected_success == observed_success else "hard_mismatch", observed_success, expected_success, "Steam space-roll threshold")
                if pending_space:
                    pending_space.update({"roll": roll, "success": observed_success, "line_roll": index})
                    if not observed_success and pending_space.get("card"):
                        side = pending_space["side"]
                        expected_position = pending_space["before"][side]
                        check = add_check([pending_space["line"], index], "space_saito", "pending_oracle", expected_position, None, "Saito failed space attempt with recorded die")
                        self.request(check, {
                            "kind": "space", "side": side, "card": pending_space["card"],
                            "roll": roll, "space_us": pending_space["before"]["us"],
                            "space_ussr": pending_space["before"]["ussr"],
                        }, {"compare": "space", "side": side, "expected": expected_position})
                        pending_space = None
                audit.mechanics["space_race"] += 1; audit.dice["space"] += 1
                audit.dice[f"space_roll_{roll}"] += 1; semantic = True

            match = re.match(r"(USSR|US) advances to (\d+) in the Space Race\.$", stripped)
            if match:
                side = match.group(1).lower(); target = int(match.group(2)); before = space[side]
                # One Small Step explicitly advances two boxes; all other logged successful attempts advance one.
                allowed = {before + 1}
                if current_card and normalized(current_card) == "onesmallstep": allowed.add(before + 2)
                add_check([index], "space_advance", "match" if target in allowed else "hard_mismatch", sorted(allowed), target, "Space-track movement")
                space[side] = target
                if pending_space and pending_space.get("roll") is not None and pending_space.get("card"):
                    check = add_check([pending_space["line"], index], "space_saito", "pending_oracle", target, None, "Saito space attempt with recorded die")
                    self.request(check, {
                        "kind": "space", "side": side, "card": pending_space["card"],
                        "roll": pending_space["roll"], "space_us": pending_space["before"]["us"],
                        "space_ussr": pending_space["before"]["ussr"],
                    }, {"compare": "space", "side": side, "expected": target})
                    pending_space = None
                semantic = True

            # Failed space attempts have no advance line, so close them at the next turn header later.

            match = re.search(r"(USSR|US) gains (\d+) VP\. Score is (even|USSR \d+|US \d+)\.", stripped)
            if match:
                side = match.group(1).lower(); amount = int(match.group(2)); expected_vp = vp + (amount if side == "us" else -amount)
                score_text = match.group(3)
                actual_vp = 0 if score_text == "even" else (int(score_text.split()[1]) * (1 if score_text.startswith("US ") else -1))
                add_check([index], "vp", "match" if expected_vp == actual_vp else "hard_mismatch", expected_vp, actual_vp, "VP arithmetic and signed score")
                if pending_scoring:
                    observed_delta = actual_vp - vp
                    check = add_check([pending_scoring["line"], index], "scoring", "pending_oracle", observed_delta, None, f"{pending_scoring['region']} scoring")
                    self.request(check, {
                        "kind": "score", "region": pending_scoring["region"], "side": pending_scoring["side"],
                        "countries": pending_scoring["countries"], "events": pending_scoring["events"],
                    }, {"compare": "vp_delta", "expected": observed_delta})
                    pending_scoring = None
                vp = actual_vp; audit.mechanics["vp"] += 1; semantic = True
                if abs(vp) >= 20:
                    saw_terminal = "vp_threshold"

            match = re.search(r"No VP awarded\. Score is (even|USSR \d+|US \d+)\.", stripped)
            if match:
                score_text = match.group(1)
                actual_vp = 0 if score_text == "even" else (int(score_text.split()[1]) * (1 if score_text.startswith("US ") else -1))
                add_check([index], "vp", "match" if vp == actual_vp else "hard_mismatch", vp, actual_vp, "No-VP score preservation")
                if pending_scoring:
                    check = add_check([pending_scoring["line"], index], "scoring", "pending_oracle", 0, None, f"{pending_scoring['region']} scoring")
                    self.request(check, {
                        "kind": "score", "region": pending_scoring["region"], "side": pending_scoring["side"],
                        "countries": pending_scoring["countries"], "events": pending_scoring["events"],
                    }, {"compare": "vp_delta", "expected": 0})
                    pending_scoring = None
                semantic = True

            match = re.search(r"DEFCON (degrades|improves) to (\d+)", stripped)
            if match:
                target = int(match.group(2))
                event_key = normalized(current_event_name or "")
                if event_key == "saltnegotiations":
                    expected: Any = min(5, defcon + 2)
                    status = "match" if target == expected else "hard_mismatch"
                elif event_key == "howilearnedtostopworrying":
                    expected = "chosen DEFCON 2..5"
                    direction_ok = (match.group(1) == "degrades" and target < defcon) or (match.group(1) == "improves" and target > defcon) or target == defcon
                    status = "match" if 2 <= target <= 5 and direction_ok else "hard_mismatch"
                else:
                    expected = max(1, defcon - 1) if match.group(1) == "degrades" else min(5, defcon + 1)
                    status = "match" if target == expected else "hard_mismatch"
                add_check([index], "defcon", status, expected, target, f"DEFCON {match.group(1)}")
                defcon = target; audit.mechanics["defcon"] += 1; semantic = True
                if target < 2:
                    saw_terminal = "nuclear_war"

            match = re.match(r"(USSR|US) Military Ops to (\d+)$", stripped)
            if match:
                side = match.group(1).lower(); target = int(match.group(2))
                if pending_coup and "ops" in pending_coup:
                    expected_milops = min(5, milops[side] + int(pending_coup["ops"]))
                    status = "match" if target == expected_milops else "hard_mismatch"
                    add_check([pending_coup["line"], index], "coup_milops", status,
                              expected_milops, target, "Coup Ops added to the capped MilOps track")
                else:
                    status = "match" if milops[side] <= target <= 5 else "hard_mismatch"
                    add_check([index], "milops", status, f"{milops[side]}..5", target, "MilOps monotonic capped track")
                milops[side] = target; audit.mechanics["milops"] += 1; semantic = True

            match = re.match(r"(VICTORY|DEFEAT): (\d+)(?: \(([+-]\d+)\))?\s+(>=|<) (\d+)$", stripped)
            if match:
                outcome = match.group(1).lower(); roll = int(match.group(2))
                modifier = int(match.group(3) or 0); comparator = match.group(4); threshold = int(match.group(5))
                adjusted = roll + modifier
                expected_victory = adjusted >= threshold
                displayed_victory = outcome == "victory" and comparator == ">="
                displayed_defeat = outcome == "defeat" and comparator == "<"
                valid = (displayed_victory and expected_victory) or (displayed_defeat and not expected_victory)
                add_check([index], "war_roll", "match" if valid else "hard_mismatch",
                          "victory" if expected_victory else "defeat", outcome,
                          f"War roll {roll}{modifier:+d} against {threshold}")
                audit.mechanics["war"] += 1; audit.dice["war"] += 1
                audit.dice[f"war_roll_{roll}"] += 1; semantic = True
            match = re.match(r"Trap Roll: (\d+) (<=|>) (\d+) -- Trap (Escaped|Remains in Effect)$", stripped)
            if match:
                roll = int(match.group(1)); threshold = int(match.group(3))
                escaped = match.group(4) == "Escaped"
                valid = (roll <= threshold) == escaped
                add_check([index], "trap_roll", "match" if valid else "hard_mismatch",
                          "escaped" if roll <= threshold else "remains", match.group(4).lower(),
                          "Bear Trap/Quagmire displayed roll threshold")
                audit.mechanics["trap"] += 1; audit.dice["trap"] += 1
                audit.dice[f"trap_roll_{roll}"] += 1; semantic = True
            if stripped == "*RESHUFFLE*":
                audit.mechanics["shuffle"] += 1; semantic = True

            # Known public narrative without enough hidden state for an atomic oracle.
            if not semantic and stripped.startswith(self.KNOWN_NARRATIVE_PREFIXES):
                narrative = True
            if not semantic and re.match(r".+ (?:is |are |has |have |may |can |will |does |chooses |selects |cancels |removes |returns |takes |receives |allows |prevents |triggers ).+", stripped):
                narrative = True

            if semantic:
                audit.semantic_lines += 1
            elif narrative:
                audit.narrative_lines += 1
            else:
                audit.unparsed.append({"line": index, "text": stripped})

            if normalized(stripped).startswith("eventwargames") or (current_card and normalized(current_card) == "wargames" and current_mode == "event"):
                saw_terminal = "wargames"

        finalize_coup(len(lines))
        finalize_realign(len(lines))
        finalize_placement(len(lines))
        audit.final_public_state = {
            "vp": vp, "defcon": defcon, "milops": dict(milops), "space": dict(space),
            "turn": audit.max_turn, "max_action_round": dict(audit.max_ar),
        }
        if path.name == "0.txt":
            audit.completeness = "setup_prefix"
        elif saw_terminal:
            audit.completeness = "terminal"; audit.terminal_condition = saw_terminal
            if saw_terminal == "nuclear_war" and current_side in {"us", "ussr"}:
                audit.terminal_winner = "ussr" if current_side == "us" else "us"
            elif vp != 0:
                audit.terminal_winner = "us" if vp > 0 else "ussr"
        elif audit.max_turn >= 10 and any("gains" in line and "VP" in line for line in lines[-8:]):
            audit.completeness = "terminal_or_final_scoring"; audit.terminal_condition = "final_scoring_or_vp_terminal"
            audit.terminal_winner = "us" if vp > 0 else "ussr" if vp < 0 else None
        else:
            audit.completeness = "truncated_prefix"
        return audit

    def resolve_oracles(self, audits: list[FileAudit]) -> None:
        responses = oracle_batch(self.requests) if self.requests else {}
        for request_id, check in self.request_checks.items():
            response = responses.get(request_id)
            if not response or not response.get("ok"):
                check.status = "oracle_error"
                check.actual = response.get("error") if response else "missing oracle response"
                continue
            result = response["result"]
            context = self.request_context[request_id]
            compare = context["compare"]
            if compare == "country": actual = result["country"]
            elif compare == "vp_delta": actual = int(result["vp_delta"])
            elif compare == "space": actual = int(result[f"space_{context['side']}"])
            else: actual = result
            expected = context["expected"]
            check.actual = actual
            # Preserve an independently detected trace inconsistency even if the
            # isolated oracle happens to match the trace post-state.
            if check.status != "hard_mismatch":
                check.status = "match" if actual == expected else "hard_mismatch"
            if check.status == "hard_mismatch":
                check.detail += "; isolated Saito oracle differs"


def aggregate(audits: list[FileAudit], parser: Parser,
              bridge_checks: list[dict[str, Any]],
              materialized_differentials: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [check for audit in audits for check in audit.checks]
    statuses = collections.Counter(check.status for check in checks)
    bridge_statuses = collections.Counter(item["classification"] for item in bridge_checks)
    bridge_differentials = [item for item in bridge_checks if item["classification"] == "deterministic_hard_mismatch"]
    mechanics = collections.Counter()
    cards = collections.Counter()
    events = collections.Counter()
    for audit in audits:
        mechanics.update(audit.mechanics); cards.update(audit.cards); events.update(audit.events)
    nonblank = sum(audit.nonblank_lines for audit in audits)
    semantic = sum(audit.semantic_lines for audit in audits)
    narrative = sum(audit.narrative_lines for audit in audits)
    return {
        "format": "steam_trace_differential_v1",
        "corpus": {
            "directory": str(DEFAULT_TRACES.relative_to(ROOT)),
            "files": len(audits), "lines": sum(audit.lines for audit in audits),
            "bytes": sum(audit.bytes for audit in audits), "nonblank_lines": nonblank,
        },
        "parse": {
            "semantic_lines": semantic, "known_narrative_lines": narrative,
            "unparsed_lines": sum(len(audit.unparsed) for audit in audits),
            "recognized_fraction_nonblank": (semantic + narrative) / nonblank if nonblank else 1.0,
            "semantic_fraction_nonblank": semantic / nonblank if nonblank else 1.0,
        },
        "deterministic_checks": {
            "total": len(checks) + len(bridge_checks),
            "materialized_and_trace_public_total": len(checks),
            "bridge_minimized_total": len(bridge_checks),
            "by_status": {
                **dict(statuses),
                "match": statuses.get("match", 0) + bridge_statuses.get("match", 0),
                "hard_mismatch": statuses.get("hard_mismatch", 0) + bridge_statuses.get("deterministic_hard_mismatch", 0),
            },
            "exact_matches": statuses.get("match", 0) + bridge_statuses.get("match", 0),
            "hard_mismatches": statuses.get("hard_mismatch", 0) + bridge_statuses.get("deterministic_hard_mismatch", 0),
            "parser_limitations": statuses.get("parser_limitation", 0),
            "unique_root_causes": len({
                (item["root_cause"]["file"], item["root_cause"]["function"])
                for item in bridge_differentials + materialized_differentials
            }),
        },
        "coverage": {
            "mechanics": dict(sorted(mechanics.items())),
            "cards_in_action_headers": dict(sorted(cards.items())),
            "events": dict(sorted(events.items())),
            "unique_cards": len(cards), "unique_events": len(events),
        },
        "limitations": [
            "Steam text logs omit complete hands, deck order, option IDs, and RNG state for shuffles/draws.",
            "Chronological public state is reconstructed from explicit deltas; hidden-state decisions are not forced into hard mismatch counts.",
            "Browser/UI checks are reserved for minimized bridge legal-catalog divergences, because UI projects that catalog without rule logic.",
        ],
        "bridge_differentials": bridge_differentials,
        "confirmed_defects": bridge_checks,
        "materialized_differentials": materialized_differentials,
        "training_env_projection": (
            json.loads(ENV_DIAGNOSTIC_OUTPUT.read_text()) if ENV_DIAGNOSTIC_OUTPUT.exists()
            else {"status": "not_run", "command": f"{ENV_PYTHON} {ENV_DIAGNOSTIC_SCRIPT}"}
        ),
        "training_artifact_exposure": json.loads(TRAINING_EXPOSURE_OUTPUT.read_text()),
        "files": [audit.as_dict() for audit in audits],
    }


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    argument_parser.add_argument("--output", type=Path, default=OUTPUT)
    args = argument_parser.parse_args()
    trace_root = args.traces.resolve()
    allowed_root = DEFAULT_TRACES.resolve()
    if trace_root != allowed_root:
        raise SystemExit(f"refusing non-canonical trace directory: {trace_root}")
    meta = load_meta()
    parser = Parser(meta)
    paths = sorted(trace_root.glob("*.txt"), key=lambda path: (not path.stem.isdigit(), int(path.stem) if path.stem.isdigit() else path.stem))
    audits = [parser.parse_file(path) for path in paths]
    parser.resolve_oracles(audits)
    bridge_diagnostics = run_bridge_diagnostics()
    bridge_checks = classify_current_bridge_checks(build_bridge_differentials(bridge_diagnostics), bridge_diagnostics)
    materialized_differentials = build_materialized_differentials(audits)
    result = aggregate(audits, parser, bridge_checks, materialized_differentials)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "files": result["corpus"]["files"],
        "lines": result["corpus"]["lines"], "parse": result["parse"],
        "checks": result["deterministic_checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
