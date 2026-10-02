#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from scripts.run_puct_multiday_campaign import (
        apply_compatibility_adaptation,
        atomic_json,
        checkpoint_from_log,
        distillation_checkpoints,
        eval_command,
        evaluation_summary,
        log_wandb,
    )
except ModuleNotFoundError:
    from run_puct_multiday_campaign import (  # type: ignore[no-redef]
        apply_compatibility_adaptation,
        atomic_json,
        checkpoint_from_log,
        distillation_checkpoints,
        eval_command,
        evaluation_summary,
        log_wandb,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add checkpoint sweep and policy/value compatibility diagnostics to a completed cycle."
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--compatibility-gpu", default="5")
    parser.add_argument("--sweep-seed-pairs", type=int, default=4)
    parser.add_argument("--eval-seed-pairs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--recover-unreported-cycle",
        action="store_true",
        help=(
            "Recover a cycle whose durable training/evaluation stages completed but whose "
            "non-gating diagnostic stage failed before cycle_report.json was written."
        ),
    )
    return parser.parse_args()


def run_logged(name: str, command: list[str], cycle_dir: Path, cuda: str) -> None:
    marker = cycle_dir / f".{name}.complete"
    if marker.is_file():
        return
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda
    with (cycle_dir / f"{name}.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "postprocess_stage_start", "command": command}) + "\n")
        handle.flush()
        subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    marker.write_text("complete\n", encoding="utf-8")


def _required_recovery_stages(cycle_dir: Path) -> list[str]:
    required = ["collect", "distill", "ppo", "eval_raw", "eval_d2"]
    if (cycle_dir / "evaluation" / "large_raw" / "results.json").is_file() or (
        cycle_dir / "evaluation" / "large_d2" / "results.json"
    ).is_file():
        required.extend(["eval_large_raw", "eval_large_d2"])
    return required


def recover_cycle_report(
    campaign_dir: Path,
    cycle_dir: Path,
    cycle: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_path = campaign_dir / "campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("cycle", -1)) != cycle - 1:
        raise SystemExit(
            f"recovery requires campaign state at cycle {cycle - 1}, "
            f"found {state.get('cycle')}"
        )
    missing = [
        stage for stage in _required_recovery_stages(cycle_dir)
        if not (cycle_dir / f".{stage}.complete").is_file()
    ]
    if missing:
        raise SystemExit(f"cannot recover cycle {cycle}; missing durable stages: {missing}")

    cycle_config = json.loads((cycle_dir / "cycle_config.json").read_text(encoding="utf-8"))
    campaign_config = json.loads(
        (campaign_dir / "campaign_config.json").read_text(encoding="utf-8")
    )
    accepted = Path(str(cycle_config["accepted_parent"])).resolve()
    candidate = Path(checkpoint_from_log(cycle_dir / "ppo.log")).resolve()
    raw = evaluation_summary(cycle_dir / "evaluation" / "raw")
    d2 = evaluation_summary(cycle_dir / "evaluation" / "d2")
    side_floor = min(
        float(summary["by_candidate_side"][side]["score"])
        for summary in (raw, d2)
        for side in ("us", "ussr")
    )
    working_pass = (
        float(raw["score"]) >= float(campaign_config.get("working_raw_floor", 0.45))
        and float(d2["score"]) >= float(campaign_config.get("working_d2_floor", 0.45))
        and side_floor >= float(campaign_config.get("working_side_floor", 0.30))
    )

    large = None
    large_raw_path = cycle_dir / "evaluation" / "large_raw" / "results.json"
    large_d2_path = cycle_dir / "evaluation" / "large_d2" / "results.json"
    if large_raw_path.is_file() and large_d2_path.is_file():
        large = {
            "raw": evaluation_summary(cycle_dir / "evaluation" / "large_raw"),
            "d2": evaluation_summary(cycle_dir / "evaluation" / "large_d2"),
        }
    promotion_candidate = bool(
        large
        and float(large["raw"]["score"]) >= float(campaign_config.get("promotion_score", 0.55))
        and float(large["d2"]["score"]) >= float(campaign_config.get("promotion_score", 0.55))
        and float(large["raw"]["paired_bootstrap_ci95_low"])
        >= float(campaign_config.get("promotion_ci_low", 0.50))
        and float(large["d2"]["paired_bootstrap_ci95_low"])
        >= float(campaign_config.get("promotion_ci_low", 0.50))
        and min(
            float(large[mode]["by_candidate_side"][side]["score"])
            for mode in ("raw", "d2")
            for side in ("us", "ussr")
        ) >= float(campaign_config.get("promotion_side_floor", 0.45))
    )
    diagnostic_dir = cycle_dir / "frozen_diagnostics"
    diagnostic_manifest = diagnostic_dir / "manifest.json"
    completed_positions: list[str] = []
    if diagnostic_manifest.is_file():
        completed_positions = sorted(
            json.loads(diagnostic_manifest.read_text(encoding="utf-8")).get("results", {})
        )
    report = {
        "cycle": cycle,
        "source_checkpoint": str(Path(str(cycle_config["source_checkpoint"])).resolve()),
        "candidate_checkpoint": str(candidate),
        "accepted_parent": str(accepted),
        "working_pass": working_pass,
        "promotion_candidate": promotion_candidate,
        "raw": raw,
        "d2": d2,
        "large": large,
        "checkpoint_sweep": None,
        "policy_value_compatibility": None,
        "collection": json.loads(
            (cycle_dir / "puct_selfplay" / "collection_summary.json").read_text(encoding="utf-8")
        ),
        "distillation": json.loads(
            (cycle_dir / "search_student" / "distillation_report.json").read_text(encoding="utf-8")
        ),
        "frozen_diagnostics": {
            "status": "partial_failure",
            "completed_positions": completed_positions,
            "missing_positions": sorted(set([
                "olympics_t1", "olympics_t3", "olympics_t5", "destalinization_t1",
                "europe_scoring_t2", "mexico_t5_04", "mexico_t5_05", "iran_coup_t7",
                "final_scoring_t10",
            ]) - set(completed_positions)),
            "failure_stage": "final_scoring_t10.terminal_progressive_search",
            "log": str(cycle_dir / "frozen_diagnostics.log"),
        },
        "recovered_after_diagnostic_failure": True,
    }
    atomic_json(cycle_dir / "cycle_report.json", report)
    state["cycle"] = cycle
    # The recovered report is only a staging point. Keep the accepted parent active
    # until the checkpoint sweep and compatibility gate finish successfully.
    state["working_checkpoint"] = str(accepted)
    if promotion_candidate:
        state.setdefault("promotion_candidates", []).append(
            str(cycle_dir / "cycle_report.json")
        )
    state.setdefault("cycles", []).append({
        "cycle": cycle,
        "report": str(cycle_dir / "cycle_report.json"),
        "working_pass": working_pass,
        "promotion_candidate": promotion_candidate,
    })
    atomic_json(state_path, state)
    return report, state


def main() -> None:
    args = parse_args()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    cycle_dir = campaign_dir / "cycles" / f"cycle-{args.cycle:04d}"
    report_path = cycle_dir / "cycle_report.json"
    state_path = campaign_dir / "campaign_state.json"
    if not report_path.is_file():
        if not args.recover_unreported_cycle:
            raise SystemExit(f"cycle is not durably complete: {report_path}")
        report, state = recover_cycle_report(campaign_dir, cycle_dir, args.cycle)
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state["cycle"]) != int(args.cycle):
        raise SystemExit(
            f"postprocess requires latest completed cycle {args.cycle}, state has {state['cycle']}"
        )
    accepted = Path(str(report["accepted_parent"])).resolve()
    distill_report_path = cycle_dir / "search_student" / "distillation_report.json"
    distill_report = json.loads(distill_report_path.read_text(encoding="utf-8"))
    ppo_candidate = Path(str(report["candidate_checkpoint"])).resolve()
    checkpoint_args = [
        item
        for label, checkpoint in distillation_checkpoints(distill_report)
        for item in ("--checkpoint", f"{label}={checkpoint}")
    ]
    checkpoint_args.extend(["--checkpoint", f"ppo_final={ppo_candidate}"])
    cycle_seed = int(
        json.loads((cycle_dir / "cycle_config.json").read_text(encoding="utf-8"))["seed_base"]
    )
    sweep_dir = cycle_dir / "checkpoint_sweep"
    run_logged(
        "checkpoint_sweep",
        [
            sys.executable,
            "scripts/sweep_puct_checkpoints.py",
            "--parent-checkpoint", str(accepted),
            *checkpoint_args,
            "--output-dir", str(sweep_dir),
            "--distillation-report", str(distill_report_path),
            "--distill-log", str(cycle_dir / "distill.log"),
            "--final-label", "ppo_final",
            "--seed-pairs", str(args.sweep_seed_pairs),
            "--seed-base", str(cycle_seed + 400_000),
            "--workers", str(min(args.workers, args.sweep_seed_pairs * 2)),
            "--gpus", str(args.gpus),
        ],
        cycle_dir,
        args.gpus,
    )
    sweep = json.loads((sweep_dir / "checkpoint_sweep.json").read_text(encoding="utf-8"))
    candidate = Path(str(sweep["selection"]["selected_checkpoint"])).resolve()

    if candidate != ppo_candidate:
        eval_args = SimpleNamespace(search_workers=args.workers, gpus=args.gpus)
        selected_root = cycle_dir / "evaluation" / "selected_checkpoint"
        run_logged(
            "eval_selected_raw",
            eval_command(
                eval_args, candidate, accepted, selected_root / "raw",
                search=False, seed_pairs=args.eval_seed_pairs, seed_base=cycle_seed + 500_000,
            ),
            cycle_dir,
            args.gpus,
        )
        run_logged(
            "eval_selected_d2",
            eval_command(
                eval_args, candidate, accepted, selected_root / "d2",
                search=True, seed_pairs=args.eval_seed_pairs, seed_base=cycle_seed + 600_000,
            ),
            cycle_dir,
            args.gpus,
        )
        raw = evaluation_summary(selected_root / "raw")
        d2 = evaluation_summary(selected_root / "d2")
        report["large"] = None
        report["promotion_candidate"] = False
    else:
        raw = report["raw"]
        d2 = report["d2"]

    compatibility_dir = cycle_dir / "policy_value_compatibility"
    run_logged(
        "policy_value_compatibility",
        [
            sys.executable,
            "scripts/run_policy_value_compatibility.py",
            "--parent-checkpoint", str(accepted),
            "--child-checkpoint", str(candidate),
            "--output-dir", str(compatibility_dir),
            "--belief-particles", "32",
            "--budget", "32",
            "--gpu", str(args.compatibility_gpu),
            "--seed", str(cycle_seed + 450_000),
        ],
        cycle_dir,
        args.compatibility_gpu,
    )
    compatibility = json.loads(
        (compatibility_dir / "compatibility_report.json").read_text(encoding="utf-8")
    )

    config = json.loads((campaign_dir / "campaign_config.json").read_text(encoding="utf-8"))
    side_floor = min(
        float(summary["by_candidate_side"][side]["score"])
        for summary in (raw, d2)
        for side in ("us", "ussr")
    )
    working_pass = (
        float(raw["score"]) >= float(config.get("working_raw_floor", 0.45))
        and float(d2["score"]) >= float(config.get("working_d2_floor", 0.45))
        and side_floor >= float(config.get("working_side_floor", 0.30))
    )
    report.update(
        {
            "candidate_checkpoint": str(candidate),
            "raw": raw,
            "d2": d2,
            "working_pass": working_pass,
            "checkpoint_sweep": sweep,
            "policy_value_compatibility": compatibility,
            "postprocessed": True,
        }
    )
    atomic_json(report_path, report)
    for item in state["cycles"]:
        if int(item["cycle"]) == int(args.cycle):
            item["working_pass"] = working_pass
            item["promotion_candidate"] = bool(report["promotion_candidate"])
    state["working_checkpoint"] = str(candidate if working_pass else accepted)
    selection = sweep["selection"]
    if selection.get("intermediate_clearly_dominates_final") and selection.get("selected_update"):
        state["overtraining_intermediate_wins"] = int(
            state.get("overtraining_intermediate_wins", 0)
        ) + 1
        state.setdefault("overtraining_selected_updates", []).append(
            int(selection["selected_update"])
        )
    else:
        state["overtraining_intermediate_wins"] = 0
    apply_compatibility_adaptation(state, compatibility)
    atomic_json(state_path, state)
    wandb_args = SimpleNamespace(
        wandb_mode=str(config.get("wandb_mode", "online")),
        wandb_project=str(config.get("wandb_project", "twilight-struggle-puct-campaign")),
        wandb_run_name=config.get("wandb_run_name"),
        output_dir=campaign_dir,
    )
    log_wandb(
        wandb_args,
        state,
        {
            "campaign/cycle": int(args.cycle),
            "campaign/postprocessed": 1,
            "campaign/working_pass": int(working_pass),
            "sweep/selected_update": selection.get("selected_update") or -1,
            "sweep/intermediate_dominates": int(
                bool(selection.get("intermediate_clearly_dominates_final"))
            ),
            **{
                f"sweep/{label}/{mode}_score": float(record[mode]["score"])
                for label, record in sweep["checkpoints"].items()
                for mode in ("raw", "d2")
            },
            **{
                f"compatibility/{name}": float(value)
                for name, value in compatibility["agreement_with_parent_search"].items()
            },
        },
    )
    (cycle_dir / ".postprocess.complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps({
        "event": "cycle_postprocess_complete",
        "cycle": args.cycle,
        "candidate": str(candidate),
        "working_pass": working_pass,
        "selection": selection,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
