from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_puct_tournament.py"
SPEC = importlib.util.spec_from_file_location("summarize_puct_tournament", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_matches_excludes_incomplete_side_swap_blocks(tmp_path: Path) -> None:
    result_path = tmp_path / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "candidate": {"label": "search"},
                "opponent": {"label": "raw"},
                "records": [
                    {"seed": 1, "orientation": "candidate_us", "candidate_score": 1.0},
                    {"seed": 1, "orientation": "candidate_ussr", "candidate_score": 0.0},
                    {"seed": 2, "orientation": "candidate_us", "candidate_score": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    agents, matches, incomplete = MODULE.load_matches([result_path])

    assert agents == ["raw", "search"]
    assert len(matches) == 2
    assert {match["seed"] for match in matches} == {1}
    assert incomplete == [
        {
            "source": str(result_path.resolve()),
            "seed": 2,
            "orientations": ["candidate_us"],
            "record_count": 1,
        }
    ]
