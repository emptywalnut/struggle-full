from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from struggle_ai.mcts import SelectiveSearchDecision


SEARCH_REPLAY_VERSION = 2
SUPPORTED_SEARCH_REPLAY_VERSIONS = (1, SEARCH_REPLAY_VERSION)


@dataclass(frozen=True)
class SearchReplayMetadata:
    side: str
    trigger_categories: tuple[str, ...]
    legal_actions: tuple[dict[str, Any], ...]
    hierarchical_path: tuple[dict[str, str], ...]
    raw_action: int
    search_action: int
    search_depth: int
    simulation_budget: int
    root_q_gap: float
    action_flipped: bool
    root_value_target: float | None = None
    source_game: str | None = None
    source_step: int | None = None
    q_advantage_over_raw: float = 0.0
    max_meaningful_depth_reached: int = 0
    max_turn_delta: int = 0
    cross_turn_leaf_evaluations: int = 0
    terminal_evaluations: int = 0
    evidence_gate_rejected: bool = False
    hierarchy_index: int = 0
    hierarchy_level: str = "root"
    teacher_action_visits: float = 0.0
    total_visits: float = 0.0
    teacher_visit_share: float = 0.0
    atomic_horizon_cutoffs: int = 0
    node_budget_cutoffs: int = 0
    sampled_large_chance_nodes: int = 0
    behavior_policy: str = "offline_search"
    complete_game: bool = False
    game_winner: str | None = None
    terminal_reason: str | None = None
    behavior_action: int | None = None
    behavior_temperature: float = 0.0
    behavior_path: tuple[dict[str, str], ...] = ()
    convergence_status: str | None = None
    convergence_configs: tuple[str, ...] = ()
    convergence_rank_correlation: float | None = None
    convergence_q_gap: float | None = None


@dataclass
class SearchReplayExample:
    observation: dict[str, np.ndarray]
    raw_policy: np.ndarray
    visit_counts: np.ndarray
    search_policy: np.ndarray
    q_values: np.ndarray
    metadata: SearchReplayMetadata

    @property
    def legal_count(self) -> int:
        return int(np.asarray(self.observation["action_mask"]).sum())


def padded(values: np.ndarray, size: int) -> np.ndarray:
    result = np.zeros((size,), dtype=np.float32)
    source = np.asarray(values, dtype=np.float32).reshape(-1)
    result[: min(size, len(source))] = source[:size]
    return result


def example_from_decision(
    observation: dict[str, np.ndarray],
    legal_actions: list[dict[str, Any]],
    side: str,
    decision: SelectiveSearchDecision,
    *,
    search_depth: int,
    root_value_target: float | None = None,
    source_game: str | None = None,
    source_step: int | None = None,
    behavior_policy: str = "offline_search",
    behavior_temperature: float = 0.0,
    convergence_status: str | None = None,
    convergence_configs: tuple[str, ...] = (),
    convergence_rank_correlation: float | None = None,
    convergence_q_gap: float | None = None,
) -> SearchReplayExample:
    output_size = int(np.asarray(observation["action_mask"]).shape[0])
    legal_count = int(np.asarray(observation["action_mask"]).sum())
    if legal_count != len(legal_actions):
        raise ValueError(f"legal mask/action mismatch: mask={legal_count} actions={len(legal_actions)}")
    if not decision.searched:
        raise ValueError("search replay requires an executed search")
    metadata = SearchReplayMetadata(
        side=str(side),
        trigger_categories=tuple(decision.triggers),
        legal_actions=tuple(json_safe_action(action) for action in legal_actions),
        hierarchical_path=tuple(decision.hierarchical_path),
        raw_action=int(decision.raw_action),
        search_action=int(decision.teacher_action if decision.teacher_action is not None else decision.action),
        search_depth=int(decision.meaningful_depth or search_depth),
        simulation_budget=int(decision.simulations),
        root_q_gap=float(decision.root_q_gap),
        action_flipped=bool(
            (decision.teacher_action if decision.teacher_action is not None else decision.action)
            != decision.raw_action
        ),
        root_value_target=None if root_value_target is None else float(root_value_target),
        source_game=source_game,
        source_step=source_step,
        q_advantage_over_raw=float(decision.q_advantage_over_raw),
        max_meaningful_depth_reached=int(decision.max_meaningful_depth_reached),
        max_turn_delta=int(decision.max_turn_delta),
        cross_turn_leaf_evaluations=int(decision.cross_turn_leaf_evaluations),
        terminal_evaluations=int(decision.terminal_evaluations),
        evidence_gate_rejected=bool(decision.evidence_gate_rejected),
        hierarchy_index=0,
        hierarchy_level="root",
        teacher_action_visits=float(
            decision.visits[decision.teacher_action if decision.teacher_action is not None else decision.action]
        ),
        total_visits=float(decision.visits.sum()),
        teacher_visit_share=float(
            decision.visits[decision.teacher_action if decision.teacher_action is not None else decision.action]
            / max(float(decision.visits.sum()), 1e-12)
        ),
        atomic_horizon_cutoffs=int(decision.atomic_horizon_cutoffs),
        node_budget_cutoffs=int(decision.node_budget_cutoffs),
        sampled_large_chance_nodes=int(decision.sampled_large_chance_nodes),
        behavior_policy=str(behavior_policy),
        behavior_action=int(decision.action),
        behavior_temperature=float(behavior_temperature),
        behavior_path=tuple(decision.behavior_path or decision.hierarchical_path),
        convergence_status=convergence_status,
        convergence_configs=tuple(convergence_configs),
        convergence_rank_correlation=convergence_rank_correlation,
        convergence_q_gap=convergence_q_gap,
    )
    return SearchReplayExample(
        observation={key: np.asarray(value, dtype=np.float32).copy() for key, value in observation.items()},
        raw_policy=padded(decision.raw_policy, output_size),
        visit_counts=padded(decision.visits, output_size),
        search_policy=padded(decision.search_policy, output_size),
        q_values=padded(decision.q_root, output_size),
        metadata=metadata,
    )


def examples_from_decision(
    observation: dict[str, np.ndarray],
    legal_actions: list[dict[str, Any]],
    side: str,
    decision: SelectiveSearchDecision,
    *,
    search_depth: int,
    root_value_target: float | None = None,
    source_game: str | None = None,
    source_step: int | None = None,
    behavior_policy: str = "offline_search",
    behavior_temperature: float = 0.0,
    convergence_status: str | None = None,
    convergence_configs: tuple[str, ...] = (),
    convergence_rank_correlation: float | None = None,
    convergence_q_gap: float | None = None,
) -> list[SearchReplayExample]:
    examples = [
        example_from_decision(
            observation,
            legal_actions,
            side,
            decision,
            search_depth=search_depth,
            root_value_target=root_value_target,
            source_game=source_game,
            source_step=source_step,
            behavior_policy=behavior_policy,
            behavior_temperature=behavior_temperature,
            convergence_status=convergence_status,
            convergence_configs=convergence_configs,
            convergence_rank_correlation=convergence_rank_correlation,
            convergence_q_gap=convergence_q_gap,
        )
    ]
    for hierarchy_index, target in enumerate(decision.hierarchical_targets, start=1):
        output_size = int(np.asarray(target.observation["action_mask"]).shape[0])
        legal_count = int(np.asarray(target.observation["action_mask"]).sum())
        if legal_count != len(target.legal_actions):
            raise ValueError(
                f"hierarchical legal mask/action mismatch: mask={legal_count} actions={len(target.legal_actions)}"
            )
        q_advantage = float(
            target.q_values[target.search_action] - target.q_values[target.raw_action]
        )
        metadata = SearchReplayMetadata(
            side=target.side,
            trigger_categories=tuple(decision.triggers),
            legal_actions=tuple(json_safe_action(action) for action in target.legal_actions),
            hierarchical_path=tuple(target.path),
            raw_action=int(target.raw_action),
            search_action=int(target.search_action),
            search_depth=int(decision.meaningful_depth or search_depth),
            simulation_budget=int(decision.simulations),
            root_q_gap=q_advantage,
            action_flipped=bool(target.search_action != target.raw_action),
            source_game=source_game,
            source_step=source_step,
            q_advantage_over_raw=q_advantage,
            max_meaningful_depth_reached=int(decision.max_meaningful_depth_reached),
            max_turn_delta=int(decision.max_turn_delta),
            cross_turn_leaf_evaluations=int(decision.cross_turn_leaf_evaluations),
            terminal_evaluations=int(decision.terminal_evaluations),
            evidence_gate_rejected=bool(decision.evidence_gate_rejected),
            hierarchy_index=hierarchy_index,
            hierarchy_level=target.level,
            teacher_action_visits=float(target.visits[target.search_action]),
            total_visits=float(target.visits.sum()),
            teacher_visit_share=float(
                target.visits[target.search_action] / max(float(target.visits.sum()), 1e-12)
            ),
            atomic_horizon_cutoffs=int(decision.atomic_horizon_cutoffs),
            node_budget_cutoffs=int(decision.node_budget_cutoffs),
            sampled_large_chance_nodes=int(decision.sampled_large_chance_nodes),
            root_value_target=None if root_value_target is None else float(root_value_target),
            behavior_policy=str(behavior_policy),
            behavior_action=None,
            behavior_temperature=float(behavior_temperature),
            behavior_path=tuple(decision.behavior_path or decision.hierarchical_path),
            convergence_status=convergence_status,
            convergence_configs=tuple(convergence_configs),
            convergence_rank_correlation=convergence_rank_correlation,
            convergence_q_gap=convergence_q_gap,
        )
        examples.append(
            SearchReplayExample(
                observation={
                    key: np.asarray(value, dtype=np.float32).copy()
                    for key, value in target.observation.items()
                },
                raw_policy=padded(target.raw_policy, output_size),
                visit_counts=padded(target.visits, output_size),
                search_policy=padded(target.search_policy, output_size),
                q_values=padded(target.q_values, output_size),
                metadata=metadata,
            )
        )
    return examples


def with_terminal_outcome(
    examples: Iterable[SearchReplayExample],
    *,
    winner: str,
    terminal_reason: str | None,
) -> list[SearchReplayExample]:
    """Attach auditable acting-side outcome targets after a complete game."""

    if winner not in {"us", "ussr", "draw"}:
        raise ValueError(f"cannot assign a terminal target for winner={winner!r}")
    completed: list[SearchReplayExample] = []
    for example in examples:
        target = None
        if example.metadata.hierarchy_index == 0:
            target = 0.0 if winner == "draw" else (1.0 if winner == example.metadata.side else -1.0)
        completed.append(
            replace(
                example,
                metadata=replace(
                    example.metadata,
                    root_value_target=target,
                    complete_game=True,
                    game_winner=winner,
                    terminal_reason=terminal_reason,
                ),
            )
        )
    return completed


class CompleteGameReplayTransaction:
    """Buffer search labels until a non-truncated game reaches a real terminal."""

    def __init__(self, writer: "SearchReplayWriter") -> None:
        self.writer = writer
        self.examples: list[SearchReplayExample] = []
        self.invalid_reason: str | None = None
        self.partial_decisions_rejected = 0

    def add_decision(self, examples: Iterable[SearchReplayExample], decision: SelectiveSearchDecision) -> None:
        if decision.atomic_horizon_cutoffs or decision.node_budget_cutoffs:
            self.partial_decisions_rejected += 1
            return
        self.examples.extend(examples)

    def invalidate(self, reason: str) -> None:
        self.invalid_reason = self.invalid_reason or str(reason)
        self.examples.clear()

    def commit(self, *, winner: str, terminal_reason: str | None) -> int:
        if self.invalid_reason is not None:
            return 0
        completed = with_terminal_outcome(
            self.examples,
            winner=winner,
            terminal_reason=terminal_reason,
        )
        for example in completed:
            self.writer.add(example)
        self.examples.clear()
        return len(completed)


class DiskBackedCompleteGameReplayTransaction:
    """Stage one game's labels on disk until a real terminal is known.

    Deep PUCT games can produce thousands of large observation tensors.  The
    in-memory transaction is convenient for short diagnostics, but retaining
    a full game can consume tens of GiB in long parallel collections.  This
    variant preserves the same fail-closed semantics while bounding resident
    memory to the staging writer's shard size.
    """

    def __init__(
        self,
        writer: "SearchReplayWriter",
        staging_dir: Path | str,
        *,
        shard_size: int = 32,
    ) -> None:
        self.writer = writer
        self.staging_dir = Path(staging_dir).expanduser().resolve()
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        self.staging_writer = SearchReplayWriter(
            self.staging_dir,
            shard_size=max(1, int(shard_size)),
        )
        self.invalid_reason: str | None = None
        self.partial_decisions_rejected = 0
        self._closed = False

    def add_decision(
        self,
        examples: Iterable[SearchReplayExample],
        decision: SelectiveSearchDecision,
    ) -> None:
        if decision.atomic_horizon_cutoffs or decision.node_budget_cutoffs:
            self.partial_decisions_rejected += 1
            return
        if self.invalid_reason is not None:
            return
        for example in examples:
            self.staging_writer.add(example)

    def invalidate(self, reason: str) -> None:
        self.invalid_reason = self.invalid_reason or str(reason)
        self._discard_staging()

    def commit(self, *, winner: str, terminal_reason: str | None) -> int:
        if self.invalid_reason is not None:
            return 0
        if winner not in {"us", "ussr", "draw"}:
            raise ValueError(f"cannot assign a terminal target for winner={winner!r}")
        manifest = self._close_staging()
        count = 0
        for example in iter_search_replay((manifest,)):
            target = None
            if example.metadata.hierarchy_index == 0:
                target = 0.0 if winner == "draw" else (1.0 if winner == example.metadata.side else -1.0)
            self.writer.add(
                replace(
                    example,
                    metadata=replace(
                        example.metadata,
                        root_value_target=target,
                        complete_game=True,
                        game_winner=winner,
                        terminal_reason=terminal_reason,
                    ),
                )
            )
            count += 1
        self._discard_staging()
        return count

    def _close_staging(self) -> Path:
        if not self._closed:
            manifest = self.staging_writer.close()
            self._closed = True
            return manifest
        return self.staging_dir / "manifest.json"

    def _discard_staging(self) -> None:
        self.staging_writer.buffer.clear()
        self.staging_writer.example_keys.clear()
        self.staging_writer.shards.clear()
        self._closed = True
        shutil.rmtree(self.staging_dir, ignore_errors=True)


def json_safe_action(action: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in action.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            result[str(key)] = value
        elif isinstance(value, (list, tuple)):
            result[str(key)] = [item for item in value if item is None or isinstance(item, (str, int, float, bool))]
    return result


class SearchReplayWriter:
    def __init__(self, output_dir: Path | str, *, shard_size: int = 128) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = max(1, int(shard_size))
        self.buffer: list[SearchReplayExample] = []
        self.shards: list[dict[str, Any]] = []
        self.total_examples = 0
        self.example_keys: set[tuple[str | None, int | None, str, int]] = set()
        manifest = self.output_dir / "manifest.json"
        if manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if int(payload.get("version", 0)) not in SUPPORTED_SEARCH_REPLAY_VERSIONS:
                raise ValueError(f"unsupported search replay version in {manifest}")
            self.shards = list(payload.get("shards", []))
            self.total_examples = int(payload.get("examples", 0))
            for example in iter_search_replay((manifest,)):
                self.example_keys.add(self._key(example))

    @staticmethod
    def _key(example: SearchReplayExample) -> tuple[str | None, int | None, str, int]:
        metadata = example.metadata
        return metadata.source_game, metadata.source_step, metadata.side, metadata.hierarchy_index

    def add(self, example: SearchReplayExample) -> None:
        key = self._key(example)
        if key in self.example_keys:
            return
        self.example_keys.add(key)
        self.buffer.append(example)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        shard_index = 1 + max(
            (-1, *(int(Path(str(shard["path"])).stem.rsplit("-", 1)[-1]) for shard in self.shards))
        )
        path = self.output_dir / f"search-replay-{shard_index:06d}.npz"
        observation_keys = tuple(sorted(self.buffer[0].observation))
        for example in self.buffer:
            if tuple(sorted(example.observation)) != observation_keys:
                raise ValueError("observation keys changed within a search replay shard")
        arrays: dict[str, np.ndarray] = {
            f"obs__{key}": np.stack([example.observation[key] for example in self.buffer])
            for key in observation_keys
        }
        arrays.update(
            {
                "raw_policy": np.stack([example.raw_policy for example in self.buffer]),
                "visit_counts": np.stack([example.visit_counts for example in self.buffer]),
                "search_policy": np.stack([example.search_policy for example in self.buffer]),
                "q_values": np.stack([example.q_values for example in self.buffer]),
                "metadata_json": np.asarray(
                    [json.dumps(asdict(example.metadata), sort_keys=True, separators=(",", ":")) for example in self.buffer]
                ),
            }
        )
        np.savez_compressed(path, **arrays)
        count = len(self.buffer)
        self.shards.append({"path": path.name, "examples": count})
        self.total_examples += count
        self.buffer.clear()
        self._write_manifest()

    def close(self) -> Path:
        self.flush()
        return self._write_manifest()

    def _write_manifest(self) -> Path:
        path = self.output_dir / "manifest.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "version": SEARCH_REPLAY_VERSION,
            "format": "compressed_npz_shards",
            "examples": self.total_examples,
            "shard_size": self.shard_size,
            "shards": self.shards,
        }
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
        return path

    def __enter__(self) -> "SearchReplayWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()


def resolve_manifest(path: Path | str) -> Path:
    resolved = Path(path).expanduser().resolve()
    return resolved / "manifest.json" if resolved.is_dir() else resolved


def iter_search_replay(
    paths: Iterable[Path | str],
    *,
    metadata_predicate: Callable[[SearchReplayMetadata], bool] | None = None,
) -> Iterator[SearchReplayExample]:
    for raw_path in paths:
        manifest_path = resolve_manifest(raw_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) not in SUPPORTED_SEARCH_REPLAY_VERSIONS:
            raise ValueError(f"unsupported search replay version in {manifest_path}")
        for shard in payload.get("shards", []):
            shard_path = manifest_path.parent / str(shard["path"])
            with np.load(shard_path, allow_pickle=False) as arrays:
                materialized = {key: arrays[key] for key in arrays.files}
                observation_keys = sorted(
                    key[5:] for key in materialized if key.startswith("obs__")
                )
                count = int(materialized["search_policy"].shape[0])
                for index in range(count):
                    metadata_payload = json.loads(str(materialized["metadata_json"][index]))
                    metadata_payload["trigger_categories"] = tuple(metadata_payload["trigger_categories"])
                    metadata_payload["legal_actions"] = tuple(metadata_payload["legal_actions"])
                    metadata_payload["hierarchical_path"] = tuple(metadata_payload["hierarchical_path"])
                    metadata_payload.setdefault("behavior_policy", "offline_search")
                    metadata_payload.setdefault("complete_game", False)
                    metadata_payload.setdefault("game_winner", None)
                    metadata_payload.setdefault("terminal_reason", None)
                    metadata_payload.setdefault("behavior_action", None)
                    metadata_payload.setdefault("behavior_temperature", 0.0)
                    metadata_payload.setdefault("behavior_path", ())
                    metadata_payload["behavior_path"] = tuple(metadata_payload["behavior_path"])
                    metadata_payload.setdefault("convergence_status", None)
                    metadata_payload.setdefault("terminal_evaluations", 0)
                    metadata_payload.setdefault("evidence_gate_rejected", False)
                    metadata_payload.setdefault("convergence_configs", ())
                    metadata_payload["convergence_configs"] = tuple(metadata_payload["convergence_configs"])
                    metadata_payload.setdefault("convergence_rank_correlation", None)
                    metadata_payload.setdefault("convergence_q_gap", None)
                    metadata = SearchReplayMetadata(**metadata_payload)
                    if metadata_predicate is not None and not metadata_predicate(metadata):
                        continue
                    yield SearchReplayExample(
                        observation={
                            key: materialized[f"obs__{key}"][index].copy()
                            for key in observation_keys
                        },
                        raw_policy=materialized["raw_policy"][index].copy(),
                        visit_counts=materialized["visit_counts"][index].copy(),
                        search_policy=materialized["search_policy"][index].copy(),
                        q_values=materialized["q_values"][index].copy(),
                        metadata=metadata,
                    )


def _load_search_replay_shard(
    shard_path: Path,
    *,
    metadata_predicate: Callable[[SearchReplayMetadata], bool] | None = None,
) -> list[SearchReplayExample]:
    examples: list[SearchReplayExample] = []
    with np.load(shard_path, allow_pickle=False) as arrays:
        # NpzFile is lazy: every arrays[key] call decompresses that member.
        # Materialize once before iterating rows to avoid O(rows * members)
        # repeated decompression.
        materialized = {key: arrays[key] for key in arrays.files}
        observation_keys = sorted(key[5:] for key in materialized if key.startswith("obs__"))
        count = int(materialized["search_policy"].shape[0])
        for index in range(count):
            metadata_payload = json.loads(str(materialized["metadata_json"][index]))
            metadata_payload["trigger_categories"] = tuple(metadata_payload["trigger_categories"])
            metadata_payload["legal_actions"] = tuple(metadata_payload["legal_actions"])
            metadata_payload["hierarchical_path"] = tuple(metadata_payload["hierarchical_path"])
            metadata_payload.setdefault("behavior_policy", "offline_search")
            metadata_payload.setdefault("complete_game", False)
            metadata_payload.setdefault("game_winner", None)
            metadata_payload.setdefault("terminal_reason", None)
            metadata_payload.setdefault("behavior_action", None)
            metadata_payload.setdefault("behavior_temperature", 0.0)
            metadata_payload.setdefault("behavior_path", ())
            metadata_payload["behavior_path"] = tuple(metadata_payload["behavior_path"])
            metadata_payload.setdefault("convergence_status", None)
            metadata_payload.setdefault("terminal_evaluations", 0)
            metadata_payload.setdefault("evidence_gate_rejected", False)
            metadata_payload.setdefault("convergence_configs", ())
            metadata_payload["convergence_configs"] = tuple(metadata_payload["convergence_configs"])
            metadata_payload.setdefault("convergence_rank_correlation", None)
            metadata_payload.setdefault("convergence_q_gap", None)
            metadata = SearchReplayMetadata(**metadata_payload)
            if metadata_predicate is not None and not metadata_predicate(metadata):
                continue
            examples.append(
                SearchReplayExample(
                    observation={
                        key: materialized[f"obs__{key}"][index].copy()
                        for key in observation_keys
                    },
                    raw_policy=materialized["raw_policy"][index].copy(),
                    visit_counts=materialized["visit_counts"][index].copy(),
                    search_policy=materialized["search_policy"][index].copy(),
                    q_values=materialized["q_values"][index].copy(),
                    metadata=metadata,
                )
            )
    return examples


def load_search_replay(
    paths: Iterable[Path | str],
    *,
    max_examples: int | None = None,
    workers: int = 1,
    metadata_predicate: Callable[[SearchReplayMetadata], bool] | None = None,
) -> list[SearchReplayExample]:
    if int(workers) <= 1:
        result: list[SearchReplayExample] = []
        for example in iter_search_replay(paths, metadata_predicate=metadata_predicate):
            result.append(example)
            if max_examples is not None and len(result) >= max_examples:
                break
        return result

    shard_paths: list[Path] = []
    for raw_path in paths:
        manifest_path = resolve_manifest(raw_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) not in SUPPORTED_SEARCH_REPLAY_VERSIONS:
            raise ValueError(f"unsupported search replay version in {manifest_path}")
        shard_paths.extend(
            manifest_path.parent / str(shard["path"])
            for shard in payload.get("shards", [])
        )
    result: list[SearchReplayExample] = []
    # executor.map preserves manifest/shard order, keeping seeded training
    # reproducible while overlapping compressed NPZ decoding.
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        for shard_examples in executor.map(
            lambda shard_path: _load_search_replay_shard(
                shard_path,
                metadata_predicate=metadata_predicate,
            ),
            shard_paths,
        ):
            result.extend(shard_examples)
            if max_examples is not None and len(result) >= max_examples:
                del result[max_examples:]
                break
    return result
