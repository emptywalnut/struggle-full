from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.run_puct_selfplay_iteration import diagnostic_gate, eval_command, parse_args


def test_diagnostic_gate_requires_matching_explicit_approval(tmp_path) -> None:
    child = (tmp_path / "child").resolve()
    child.mkdir()

    passed, evidence = diagnostic_gate(None, child)
    assert not passed
    assert "no diagnostic" in evidence["reason"]

    approval = tmp_path / "pass.json"
    approval.write_text(
        json.dumps({"passed": True, "child_checkpoint": str(child)}) + "\n",
        encoding="utf-8",
    )
    passed, evidence = diagnostic_gate(approval, child)
    assert passed
    assert evidence["checkpoint_matches"]

    approval.write_text(
        json.dumps({"passed": True, "child_checkpoint": str(tmp_path / "other")}) + "\n",
        encoding="utf-8",
    )
    passed, evidence = diagnostic_gate(approval, child)
    assert not passed
    assert not evidence["checkpoint_matches"]


def test_eval_command_uses_parallel_dual_gpu_evaluator() -> None:
    args = SimpleNamespace(
        parent_checkpoint=Path("/tmp/parent"),
        eval_seed_pairs=20,
        seed_base=10,
        parallel_search_workers=32,
        parallel_inference_gpus="4,5",
        inference_replicas_per_gpu=2,
        max_inference_batch_size=64,
        max_inference_delay_ms=4.0,
        eval_inference_replicas_per_checkpoint=1,
    )

    command = eval_command(
        args,
        Path("/tmp/child"),
        Path("/tmp/output"),
        search=True,
        depth=2,
        simulations=32,
    )

    assert command[1] == "scripts/eval_puct_agent_pair_parallel.py"
    assert command[command.index("--workers") + 1] == "32"
    assert command[command.index("--gpus") + 1] == "4,5"
    assert command[command.index("--replicas-per-checkpoint") + 1] == "1"
    assert command[-1] == "--search"


def test_iteration_defaults_use_conservative_joint_distillation(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_puct_selfplay_iteration.py",
            "--parent-checkpoint",
            str(tmp_path / "parent"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    args = parse_args()

    assert args.distill_parameter_scope == "policy_value_heads"
    assert args.distill_batch_size == 128
    assert args.distill_lr == 2e-6
    assert args.lambda_value == 0.1
    assert args.lambda_search_q == 0.02
    assert args.lambda_anchor == 0.3
    assert args.min_value_examples_per_batch == 32
    assert args.separate_value_batch_size == 64
    assert args.distill_require_flip
    assert args.distill_min_q_advantage == 0.02
    assert args.distill_min_teacher_visit_share == 0.20
    assert args.distill_search_policy_mix == 0.5
    assert args.eval_inference_replicas_per_checkpoint == 1
