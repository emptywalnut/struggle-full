#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from scripts.eval_puct_agent_pair_parallel import elo_delta, summary
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from eval_puct_agent_pair_parallel import elo_delta, summary


@dataclass(frozen=True)
class Agent:
    label: str
    checkpoint: Path
    mode: str
    stage1: bool = True
    note: str = ""

    @property
    def identity(self) -> str:
        return f"adaptive:{self.label}/{self.mode}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive matched PUCT Elo evaluation.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--elo-path", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-label", default="accepted-parent")
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--workers-per-job", type=int, default=16)
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="Maximum concurrent evaluation jobs assigned to each GPU.",
    )
    parser.add_argument("--replicas-per-checkpoint", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=96_000_000)
    parser.add_argument("--screen-games", type=int, default=50)
    parser.add_argument("--confirm-games", type=int, default=100)
    parser.add_argument("--max-games", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--max-episode-steps", type=int, default=1_200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_agents(path: Path) -> tuple[list[Agent], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    agents = []
    for item in payload["agents"]:
        checkpoint = Path(str(item["checkpoint"])).expanduser().resolve()
        if not (checkpoint / "rllib_checkpoint.json").is_file():
            raise FileNotFoundError(f"invalid checkpoint for {item['label']}: {checkpoint}")
        for mode in item.get("modes", ["raw"]):
            if mode not in {"raw", "d2", "d4"}:
                raise ValueError(f"unsupported mode {mode!r}")
            agents.append(
                Agent(
                    label=str(item["label"]),
                    checkpoint=checkpoint,
                    mode=mode,
                    stage1=bool(item.get("stage1", True)),
                    note=str(item.get("note") or ""),
                )
            )
    identities = [agent.identity for agent in agents]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate checkpoint/mode agent identity")
    return agents, payload


def mode_args(mode: str) -> list[str]:
    if mode == "raw":
        return ["--no-search", "--no-candidate-search", "--no-opponent-search"]
    if mode == "d2":
        return [
            "--no-search",
            "--candidate-search",
            "--no-opponent-search",
            "--meaningful-depth", "2",
            "--initial-simulations", "32",
            "--medium-simulations", "32",
            "--max-simulations", "32",
            "--belief-particles", "32",
            "--max-expanded-nodes", "512",
        ]
    return [
        "--no-search",
        "--candidate-search",
        "--no-opponent-search",
        "--meaningful-depth", "4",
        "--initial-simulations", "128",
        "--medium-simulations", "128",
        "--max-simulations", "128",
        "--belief-particles", "32",
        "--max-expanded-nodes", "8192",
        "--max-atomic-steps", "1024",
    ]


def stage_schedule(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    targets = [int(args.screen_games), int(args.confirm_games), int(args.max_games)]
    if targets != sorted(set(targets)) or any(target % 2 for target in targets):
        raise ValueError("screen/confirm/max games must be distinct increasing even values")
    schedule = []
    previous = 0
    for target in targets:
        games = target - previous
        schedule.append((target, previous // 2, games // 2))
        previous = target
    return schedule


def stage_dir(output_dir: Path, agent: Agent, target_games: int) -> Path:
    return output_dir / "evaluations" / agent.label / agent.mode / f"to-{target_games:03d}"


def eval_command(
    args: argparse.Namespace,
    agent: Agent,
    output: Path,
    *,
    seed_offset: int,
    seed_pairs: int,
    gpu: str,
) -> list[str]:
    return [
        sys.executable,
        "scripts/eval_puct_agent_pair_parallel.py",
        "--candidate-checkpoint", str(agent.checkpoint),
        "--opponent-checkpoint", str(args.reference_checkpoint),
        "--candidate-label", agent.identity,
        "--opponent-label", f"adaptive:{args.reference_label}/raw",
        "--output-dir", str(output),
        "--seed-pairs", str(seed_pairs),
        "--seed-base", str(int(args.seed_base) + seed_offset),
        "--workers", str(min(int(args.workers_per_job), seed_pairs * 2)),
        "--gpus", gpu,
        "--replicas-per-checkpoint", str(args.replicas_per_checkpoint),
        "--max-inference-batch-size", "32",
        "--max-inference-delay-ms", "3",
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--max-episode-steps", str(args.max_episode_steps),
        *mode_args(agent.mode),
    ]


def run_jobs(args: argparse.Namespace, jobs: list[tuple[Agent, int, int, int]]) -> None:
    gpus = [value.strip() for value in str(args.gpus).split(",") if value.strip()]
    jobs_per_gpu = max(1, int(args.jobs_per_gpu))
    capacity = min(int(args.max_concurrent), len(gpus) * jobs_per_gpu)
    pending = list(jobs)
    running: list[
        tuple[subprocess.Popen[str], Any, Agent, int, Path, str, int, int]
    ] = []
    failures: dict[tuple[str, int], int] = {}
    while pending or running:
        while pending and len(running) < capacity:
            agent, target, seed_offset, seed_pairs = pending.pop(0)
            output = stage_dir(args.output_dir, agent, target)
            if (output / "results.json").is_file():
                continue
            output.mkdir(parents=True, exist_ok=True)
            gpu_counts = {
                value: sum(item[5] == value for item in running)
                for value in gpus
            }
            gpu = min(
                (value for value in gpus if gpu_counts[value] < jobs_per_gpu),
                key=lambda value: (gpu_counts[value], gpus.index(value)),
            )
            command = eval_command(
                args, agent, output, seed_offset=seed_offset, seed_pairs=seed_pairs, gpu=gpu
            )
            (output / "command.json").write_text(
                json.dumps(command, indent=2) + "\n", encoding="utf-8"
            )
            if args.dry_run:
                print(json.dumps({"event": "dry_run", "agent": agent.identity, "command": command}))
                continue
            log = (output / "run.log").open("a", encoding="utf-8")
            env = dict(os.environ)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[name] = "1"
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append(
                (process, log, agent, target, output, gpu, seed_offset, seed_pairs)
            )
            print(json.dumps({"event": "eval_started", "agent": agent.identity, "target_games": target, "gpu": gpu, "pid": process.pid}), flush=True)
        if args.dry_run:
            return
        time.sleep(2.0)
        survivors = []
        for process, log, agent, target, output, gpu, seed_offset, seed_pairs in running:
            status = process.poll()
            if status is None:
                survivors.append(
                    (process, log, agent, target, output, gpu, seed_offset, seed_pairs)
                )
                continue
            log.close()
            if status != 0:
                key = (agent.identity, target)
                failures[key] = failures.get(key, 0) + 1
                if failures[key] <= 3:
                    pending.insert(0, (agent, target, seed_offset, seed_pairs))
                    print(
                        json.dumps(
                            {
                                "event": "eval_retry",
                                "agent": agent.identity,
                                "target_games": target,
                                "attempt": failures[key],
                                "exit_status": status,
                            }
                        ),
                        flush=True,
                    )
                    continue
                raise RuntimeError(
                    f"evaluation failed repeatedly for {agent.identity}; "
                    f"inspect {output / 'run.log'}"
                )
            print(json.dumps({"event": "eval_complete", "agent": agent.identity, "target_games": target}), flush=True)
        running = survivors


def aggregate_agent(
    args: argparse.Namespace,
    agent: Agent,
    *,
    max_target: int | None = None,
) -> dict[str, Any] | None:
    records: list[dict[str, Any]] = []
    completed_targets = []
    for target, _, _ in stage_schedule(args):
        if max_target is not None and target > max_target:
            break
        path = stage_dir(args.output_dir, agent, target) / "results.json"
        if not path.is_file():
            break
        result = json.loads(path.read_text(encoding="utf-8"))
        records.extend(result["records"])
        completed_targets.append(target)
    if not records:
        return None
    records.sort(key=lambda record: (int(record["seed"]), str(record["orientation"])))
    seeds = {int(record["seed"]) for record in records}
    summary_args = SimpleNamespace(
        bootstrap_samples=int(args.bootstrap_samples),
        seed_base=int(args.seed_base),
        seed_pairs=len(seeds),
    )
    result = summary(summary_args, records)
    result.update(
        {
            "agent_id": agent.identity,
            "label": agent.label,
            "mode": agent.mode,
            "checkpoint": str(agent.checkpoint),
            "note": agent.note,
            "completed_targets": completed_targets,
        }
    )
    return result


def contender_ids(results: list[dict[str, Any]], target_games: int) -> set[str]:
    selected: set[str] = set()
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if int(result["games"]) != int(target_games):
            continue
        by_mode.setdefault(str(result["mode"]), []).append(result)
    for mode_results in by_mode.values():
        ordered = sorted(mode_results, key=lambda item: float(item["score"]), reverse=True)
        if not ordered:
            continue
        best = ordered[0]
        best_low = float(best["paired_bootstrap_ci95_low"])
        if target_games >= 100:
            runner_up_high = (
                float(ordered[1]["paired_bootstrap_ci95_high"])
                if len(ordered) > 1
                else 0.5
            )
            if best_low > 0.5 and runner_up_high < best_low:
                continue
        for rank, result in enumerate(ordered):
            side_floor = min(
                float(result["by_candidate_side"][side]["score"]) for side in ("us", "ussr")
            )
            overlaps = float(result["paired_bootstrap_ci95_high"]) >= best_low
            near_leader = float(result["score"]) >= float(best["score"]) - (0.12 if target_games == 50 else 0.08)
            mandatory_screen_rank = rank < 2 and target_games == 50
            if mandatory_screen_rank or (overlaps and near_leader and side_floor >= 0.25):
                selected.add(str(result["agent_id"]))
    return selected


def update_existing_elo(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    active_ids: set[str],
) -> None:
    state = json.loads(args.elo_path.read_text(encoding="utf-8"))
    campaign_id = args.output_dir.name
    adaptive = state.setdefault("adaptive_evaluations", {})
    campaigns = adaptive.setdefault("campaigns", {})
    campaign = campaigns.setdefault(
        campaign_id,
        {
            "created_at": time.time(),
            "reference": f"adaptive:{args.reference_label}/raw",
            "reference_checkpoint": str(args.reference_checkpoint),
            "manifest": manifest,
            "entries": {},
        },
    )
    campaign["updated_at"] = time.time()
    campaign["entries"] = {
        result["agent_id"]: {
            **result,
            "status": (
                "confirmatory" if int(result["games"]) >= int(args.confirm_games)
                else "preliminary" if result["agent_id"] in active_ids
                else "superseded"
            ),
        }
        for result in results
    }
    anchor = f"adaptive:{args.reference_label}/raw"
    boards = state.setdefault("leaderboards", {})
    for board_name in ("bot", "us", "ussr"):
        board = boards.setdefault(board_name, {"ratings": {}, "games": {}})
        board.setdefault("ratings", {})
        board.setdefault("games", {})
    boards["bot"]["ratings"][f"{anchor}:shared_policy"] = 1500.0
    boards["us"]["ratings"][f"{anchor}:us_policy"] = 1500.0
    boards["ussr"]["ratings"][f"{anchor}:ussr_policy"] = 1500.0
    for result in results:
        identity = str(result["agent_id"])
        games = int(result["games"])
        boards["bot"]["ratings"][f"{identity}:shared_policy"] = 1500.0 + float(result["elo_delta"])
        boards["bot"]["games"][f"{identity}:shared_policy"] = games
        for side in ("us", "ussr"):
            side_result = result["by_candidate_side"][side]
            boards[side]["ratings"][f"{identity}:{side}_policy"] = 1500.0 + elo_delta(float(side_result["score"]))
            boards[side]["games"][f"{identity}:{side}_policy"] = int(side_result["games"])
    atomic_json(args.elo_path, state)


def write_report(args: argparse.Namespace, results: list[dict[str, Any]], active_ids: set[str]) -> None:
    ordered = sorted(results, key=lambda item: (float(item["elo_delta"]), min(float(item["by_candidate_side"][side]["score"]) for side in ("us", "ussr"))), reverse=True)
    lines = [
        "# Adaptive PUCT Elo Evaluation",
        "",
        f"Reference: `{args.reference_label}/raw` at 1500 Elo.",
        "",
        "| Agent | Games | Score | US | USSR | Elo | Paired 95% CI | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        f"| adaptive:{args.reference_label}/raw | baseline | 50.0% | 50.0% | 50.0% | +0.0 | n/a | reference |",
    ]
    for result in ordered:
        status = "confirmatory" if int(result["games"]) >= int(args.confirm_games) else "preliminary" if result["agent_id"] in active_ids else "superseded"
        lines.append(
            "| {agent_id} | {games} | {score:.1%} | {us:.1%} | {ussr:.1%} | {elo:+.1f} | {low:.1%}-{high:.1%} | {status} |".format(
                agent_id=result["agent_id"], games=result["games"], score=result["score"],
                us=result["by_candidate_side"]["us"]["score"], ussr=result["by_candidate_side"]["ussr"]["score"],
                elo=result["elo_delta"], low=result["paired_bootstrap_ci95_low"], high=result["paired_bootstrap_ci95_high"], status=status,
            )
        )
    (args.output_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def persist(args: argparse.Namespace, manifest: dict[str, Any], agents: list[Agent], active_ids: set[str]) -> list[dict[str, Any]]:
    results = [result for agent in agents if (result := aggregate_agent(args, agent)) is not None]
    update_existing_elo(args, manifest, results, active_ids)
    write_report(args, results, active_ids)
    atomic_json(args.output_dir / "state.json", {"updated_at": time.time(), "active_ids": sorted(active_ids), "results": results})
    return results


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.elo_path = args.elo_path.expanduser().resolve()
    args.reference_checkpoint = args.reference_checkpoint.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.elo_path.is_file():
        raise FileNotFoundError("--elo-path must be an existing project leaderboard")
    if not (args.reference_checkpoint / "rllib_checkpoint.json").is_file():
        raise FileNotFoundError(f"invalid reference checkpoint: {args.reference_checkpoint}")
    backup = args.elo_path.with_name(f"{args.elo_path.stem}.pre-{args.output_dir.name}{args.elo_path.suffix}")
    if not args.dry_run and not backup.exists():
        shutil.copy2(args.elo_path, backup)
    agents, manifest = load_agents(args.manifest)
    shutil.copy2(args.manifest, args.output_dir / "manifest.json")
    active = {agent.identity for agent in agents if agent.stage1}
    for target, seed_offset, seed_pairs in stage_schedule(args):
        jobs = [
            (agent, target, seed_offset, seed_pairs)
            for agent in agents
            if agent.identity in active and not (stage_dir(args.output_dir, agent, target) / "results.json").is_file()
        ]
        jobs.sort(key=lambda item: ({"raw": 0, "d2": 1, "d4": 2}[item[0].mode], item[0].label))
        run_jobs(args, jobs)
        if args.dry_run:
            return
        results = persist(args, manifest, agents, active)
        if target < int(args.max_games):
            stage_results = [
                result
                for agent in agents
                if (
                    result := aggregate_agent(
                        args,
                        agent,
                        max_target=target,
                    )
                )
                is not None
            ]
            active = contender_ids(stage_results, target)
            persist(args, manifest, agents, active)
    atomic_json(args.output_dir / "complete.json", {"completed_at": time.time(), "active_ids": sorted(active)})


if __name__ == "__main__":
    main()
