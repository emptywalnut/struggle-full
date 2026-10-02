"""Safe registry and resident model cache for private browser play.

Only the curated repository-relative candidates below can cross the browser
policy boundary. HTTP callers never provide a checkpoint path.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from gymnasium import spaces

from struggle_ai.batched_inference import load_policy_state_model
from struggle_ai.bridge_client import TwilightBridgeClient
from struggle_ai.env import observation_space_for_spec
from struggle_ai.features import MAX_ACTIONS, make_spec


ROOT = Path(__file__).resolve().parents[1]
BASELINE_IDS = frozenset({"heuristic", "random"})


MODEL_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "phase3-u75-parent",
        "label": "Phase-3 u75 trained parent",
        "source": "runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819",
        "paths": {
            "shared": "runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075/policies/shared_policy/policy_state.pkl",
        },
        "provenance": "Retained current general parent; accepted raw actor checkpoint",
        "training": {"updates": 75},
    },
    {
        "id": "phase3-composed-p30u25",
        "label": "Phase-3 actor + p30u25 critic",
        "source": "runs/search_distill/search-scaling-phase3u75-parentactor-p30u25-composed-20260828",
        "paths": {
            "shared": "runs/search_distill/search-scaling-phase3u75-parentactor-p30u25-composed-20260828/checkpoints/composed/policies/shared_policy/policy_state.pkl",
        },
        "provenance": "Verified composition: retained Phase-3 actor and validated p30u25 value tower",
        "training": {"actor_updates": 75, "critic_updates": 25, "critic_examples": 15282},
    },
    {
        "id": "legacy-mixed-side-best",
        "label": "Selected PPO mixed-side best",
        "source": "runs/ppo/selected_policies/best_us57005_ussr66001",
        "paths": {
            "us": "runs/ppo/selected_policies/best_us57005_ussr66001/policies/us_policy/policy_state.pkl",
            "ussr": "runs/ppo/selected_policies/best_us57005_ussr66001/policies/ussr_policy/policy_state.pkl",
        },
        "provenance": "Manifest-selected best policy for each side from the completed 100k-game PPO run",
        "training": {
            "us": {"steps": 12650523, "episodes": 57005},
            "ussr": {"steps": 14919451, "episodes": 66001},
        },
    },
    {
        "id": "transformer-riskshape10-selected",
        "label": "Transformer risk-shaping selected",
        "source": "runs/ppo/selected_policies/transformer_history_riskshape10_best_20260626-042440",
        "paths": {
            "us": "runs/ppo/selected_policies/transformer_history_riskshape10_best_20260626-042440/policies/us_policy/policy_state.pkl",
            "ussr": "runs/ppo/selected_policies/transformer_history_riskshape10_best_20260626-042440/policies/ussr_policy/policy_state.pkl",
        },
        "provenance": "Side-specific evaluation selection recorded in manifest.json",
        "training": {
            "us": {"steps": 46529, "episodes": 100},
            "ussr": {"steps": 88614, "episodes": 200},
        },
    },
    {
        "id": "transformer-defconfilter5-selected",
        "label": "Transformer DEFCON-filter selected",
        "source": "runs/ppo/selected_policies/transformer_history_defconfilter5_side_selected_20260627-1110",
        "paths": {
            "us": "runs/ppo/selected_policies/transformer_history_defconfilter5_side_selected_20260627-1110/policies/us_policy/policy_state.pkl",
            "ussr": "runs/ppo/selected_policies/transformer_history_defconfilter5_side_selected_20260627-1110/policies/ussr_policy/policy_state.pkl",
        },
        "provenance": "Side-specific Elo selection after checkpoint-200 evaluation",
        "training": {
            "us": {"episodes": 100, "elo": 1592.1548483114607, "eval_games": 20},
            "ussr": {"episodes": 200, "elo": 1585.9344455642222, "eval_games": 30},
        },
    },
)


AUDIT_REJECTIONS: tuple[dict[str, str], ...] = (
    {
        "id": "mcts-best-retention",
        "source": "runs/mcts/belief-chance-sideheads-usfocus-retention-gpu0-48h-20260802-221726/best_checkpoint.json",
        "reason": "promotion metadata points to a checkpoint directory removed by repository cleanup",
    },
    {
        "id": "mcts-best-tanhretention",
        "source": "runs/mcts/belief-chance-sideheads-usfocus-tanhretention-gpu0-48h-20260803-021850/best_checkpoint.json",
        "reason": "promotion metadata points to a checkpoint directory removed by repository cleanup",
    },
    {
        "id": "mcts-best-topk4",
        "source": "runs/mcts/belief-chance-sideheads-usfocus-topk4-gpu0-48h-20260803-063914/best_checkpoint.json",
        "reason": "best-checkpoint metadata is retained, but the referenced MCTS checkpoint payload is absent",
    },
    {
        "id": "p30u25-value-only",
        "source": "runs/search_distill/search-scaling-phase3u75-valuetower-weighted-p30-u25-gpu2-20260827/checkpoints/updates-000025",
        "reason": "retained as a critic/value candidate, not independently promoted as a raw browser actor",
    },
    {
        "id": "smoke-unified-bc",
        "source": "runs/warmup_distill/smoke_unified_bc/checkpoints/bc-samples-00000032",
        "reason": "32-sample integration smoke artifact is not a promoted gameplay policy",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preferred_device(preference: str | None = None) -> str:
    """Choose the configured device, preferring the CUDA device with most free memory."""

    import torch

    requested = str(preference or os.environ.get("STRUGGLE_AI_DEVICE", "auto")).strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda:"):
        index = int(requested.split(":", 1)[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"requested model device is unavailable: {requested}")
        return requested
    if requested != "auto":
        raise RuntimeError(f"unsupported STRUGGLE_AI_DEVICE: {requested}")
    if not torch.cuda.is_available():
        return "cpu"
    candidates: list[tuple[int, int]] = []
    for index in range(torch.cuda.device_count()):
        try:
            free, _total = torch.cuda.mem_get_info(index)
        except Exception:
            free = 0
        candidates.append((int(free), index))
    return f"cuda:{max(candidates)[1]}"


class BrowserModelRegistry:
    """Validated repository model catalog with lazy, per-side resident loading."""

    def __init__(
        self,
        *,
        root: Path = ROOT,
        candidates: tuple[dict[str, Any], ...] = MODEL_CANDIDATES,
        device_preference: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.device_preference = device_preference
        self.entries: dict[str, dict[str, Any]] = {}
        self.rejected: list[dict[str, str]] = [dict(item) for item in AUDIT_REJECTIONS]
        self.models: dict[tuple[str, str], tuple[Any, str]] = {}
        self.path_cache: dict[tuple[Path, str], tuple[Any, dict[str, Any]]] = {}
        self.load_counts: Counter[str] = Counter()
        self.cards: list[dict[str, Any]] = []
        self.countries: list[dict[str, Any]] = []
        self.observation_space: Any = None
        self.action_space: Any = spaces.Discrete(MAX_ACTIONS)
        self._initialize_schema()
        for candidate in candidates:
            self._discover(candidate)

    def _initialize_schema(self) -> None:
        with TwilightBridgeClient() as bridge:
            self.cards = bridge.cards()
            self.countries = bridge.countries()
        feature_spec = make_spec(self.cards, self.countries)
        self.observation_space = observation_space_for_spec(feature_spec)

    def _safe_error(self, error: BaseException) -> str:
        return str(error).replace(str(self.root), "<repository>")

    def _candidate_paths(self, candidate: dict[str, Any]) -> dict[str, Path]:
        raw = candidate["paths"]
        if "shared" in raw:
            path = (self.root / raw["shared"]).resolve()
            return {"us": path, "ussr": path}
        return {side: (self.root / raw[side]).resolve() for side in ("us", "ussr")}

    def _discover(self, candidate: dict[str, Any]) -> None:
        model_id = str(candidate["id"])
        paths = self._candidate_paths(candidate)
        try:
            if any(self.root not in path.parents for path in paths.values()):
                raise RuntimeError("candidate resolves outside repository root")
            reports: dict[str, dict[str, Any]] = {}
            hashes: dict[str, str] = {}
            for side, path in paths.items():
                if not path.is_file():
                    raise FileNotFoundError("checkpoint policy payload is missing")
                model, report = load_policy_state_model(
                    path,
                    device="cpu",
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                )
                reports[side] = report
                hashes[side] = _sha256(path)
                del model
                gc.collect()
            architectures = {report["architecture"] for report in reports.values()}
            if len(architectures) != 1:
                raise RuntimeError(f"side architecture mismatch: {sorted(architectures)}")
            self.entries[model_id] = {
                "id": model_id,
                "kind": "model",
                "label": candidate["label"],
                "source": candidate["source"],
                "provenance": candidate["provenance"],
                "training": candidate.get("training", {}),
                "architecture": next(iter(architectures)),
                "action_space_n": MAX_ACTIONS,
                "compatibility": {side: reports[side]["migration"]["mode"] for side in ("us", "ussr")},
                "inference_environment": {side: reports[side]["environment"] for side in ("us", "ussr")},
                "sha256": hashes,
                "readiness": {
                    side: {"status": "available", "device": None, "error": None}
                    for side in ("us", "ussr")
                },
                "_paths": paths,
            }
        except Exception as exc:
            self.rejected.append(
                {
                    "id": model_id,
                    "source": str(candidate.get("source", model_id)),
                    "reason": f"incompatible checkpoint: {self._safe_error(exc)}",
                }
            )

    @staticmethod
    def baseline_entries() -> list[dict[str, Any]]:
        return [
            {
                "id": "heuristic",
                "kind": "baseline",
                "label": "Scripted heuristic",
                "architecture": "scripted",
                "provenance": "struggle_ai.policies.choose_heuristic_action",
                "readiness": {side: {"status": "ready", "device": "cpu", "error": None} for side in ("us", "ussr")},
            },
            {
                "id": "random",
                "kind": "baseline",
                "label": "Random legal",
                "architecture": "random_legal",
                "provenance": "Uniform selection from the authoritative legal-action list",
                "readiness": {side: {"status": "ready", "device": "cpu", "error": None} for side in ("us", "ussr")},
            },
        ]

    def public_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in entry.items() if not key.startswith("_")}

    def catalog(self) -> dict[str, Any]:
        return {
            "policies": self.baseline_entries() + [self.public_entry(entry) for entry in self.entries.values()],
            "rejected": list(self.rejected),
            "device_preference": str(self.device_preference or os.environ.get("STRUGGLE_AI_DEVICE", "auto")),
        }

    def contains(self, model_id: str) -> bool:
        return model_id in self.entries

    def load(self, model_id: str, side: str) -> dict[str, Any]:
        if model_id not in self.entries:
            raise ValueError(f"unknown registered model ID: {model_id}")
        if side not in {"us", "ussr"}:
            raise ValueError(f"invalid model side: {side}")
        entry = self.entries[model_id]
        ready = entry["readiness"][side]
        if ready["status"] == "ready":
            return self.public_entry(entry)
        ready.update({"status": "loading", "error": None})
        try:
            desired = preferred_device(self.device_preference)
        except Exception as exc:
            safe_error = self._safe_error(exc)
            ready.update({"status": "error", "device": None, "error": safe_error})
            raise RuntimeError(f"failed to select a device for model {model_id}: {safe_error}") from exc
        path = entry["_paths"][side]
        fallback_error: str | None = None
        try:
            model, device = self._load_path(path, desired)
        except Exception as exc:
            if desired == "cpu":
                safe_error = self._safe_error(exc)
                ready.update({"status": "error", "device": "cpu", "error": safe_error})
                raise RuntimeError(f"failed to load model {model_id} for {side} on CPU: {safe_error}") from exc
            fallback_error = self._safe_error(exc)
            try:
                model, device = self._load_path(path, "cpu")
            except Exception as cpu_exc:
                safe_cpu_error = self._safe_error(cpu_exc)
                ready.update({"status": "error", "device": None, "error": safe_cpu_error})
                raise RuntimeError(
                    f"failed to load model {model_id} for {side} on {desired} ({fallback_error}) and CPU ({safe_cpu_error})"
                ) from cpu_exc
        self.models[(model_id, side)] = (model, device)
        ready.update({"status": "ready", "device": device, "error": None})
        if fallback_error:
            ready["device_fallback"] = f"{desired} failed; loaded the same model on CPU: {fallback_error}"
        return self.public_entry(entry)

    def _load_path(self, path: Path, device: str) -> tuple[Any, str]:
        cache_key = (path, device)
        cached = self.path_cache.get(cache_key)
        if cached is None:
            cached = load_policy_state_model(
                path,
                device=device,
                observation_space=self.observation_space,
                action_space=self.action_space,
            )
            self.path_cache[cache_key] = cached
            self.load_counts[str(path.relative_to(self.root))] += 1
        return cached[0], device

    def model_for(self, model_id: str, side: str) -> tuple[Any, str]:
        model = self.models.get((model_id, side))
        if model is None:
            raise RuntimeError(f"registered model {model_id} is not loaded for {side}; request loading first")
        return model

    def environment_for(self, model_id: str, side: str) -> dict[str, Any]:
        if model_id not in self.entries:
            raise ValueError(f"unknown registered model ID: {model_id}")
        return dict(self.entries[model_id]["inference_environment"][side])
