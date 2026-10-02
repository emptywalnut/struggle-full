from argparse import Namespace
from pathlib import Path

from scripts.generate_search_replay import algorithm_args


def test_algorithm_args_forwards_value_state_tower(tmp_path: Path) -> None:
    args = Namespace(
        checkpoint=tmp_path,
        num_gpus=1.0,
        hidden=256,
        value_state_tower=True,
        history_layers=2,
        card_history_layers=2,
        history_attention_heads=4,
        history_dropout=0.05,
        graph_layers=2,
        graph_neighbor_hops=2,
        policy_temperature=1.0,
    )

    assert algorithm_args(args).value_state_tower is True
