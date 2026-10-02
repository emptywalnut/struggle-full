from __future__ import annotations

import math
from typing import Any

import numpy as np


def ordinal_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def visited_q_top(visits: np.ndarray, q_values: np.ndarray) -> tuple[int, float]:
    visits = np.asarray(visits, dtype=np.float64)
    q_values = np.asarray(q_values, dtype=np.float64)
    indices = np.flatnonzero(visits > 0.0)
    if not len(indices):
        return int(np.argmax(q_values)), 0.0
    ordered = indices[np.argsort(q_values[indices], kind="stable")]
    gap = (
        float(q_values[ordered[-1]] - q_values[ordered[-2]])
        if len(ordered) > 1
        else math.inf
    )
    return int(ordered[-1]), gap


def compare_q_rankings(
    left_visits: np.ndarray,
    left_q: np.ndarray,
    right_visits: np.ndarray,
    right_q: np.ndarray,
    *,
    min_rank_correlation: float,
    meaningful_q_gap: float,
) -> dict[str, Any]:
    left_visits = np.asarray(left_visits, dtype=np.float64)
    right_visits = np.asarray(right_visits, dtype=np.float64)
    left_q = np.asarray(left_q, dtype=np.float64)
    right_q = np.asarray(right_q, dtype=np.float64)
    if not (
        left_visits.shape == right_visits.shape == left_q.shape == right_q.shape
    ):
        return {"comparable": False, "stable": False, "reason": "array_shape_mismatch"}
    common = (left_visits > 0.0) & (right_visits > 0.0)
    indices = np.flatnonzero(common)
    if len(indices) < 2:
        return {"comparable": False, "stable": False, "reason": "insufficient_shared_actions"}
    rank_left = ordinal_ranks(left_q[indices])
    rank_right = ordinal_ranks(right_q[indices])
    correlation = float(np.corrcoef(rank_left, rank_right)[0, 1])
    left_top, left_gap = visited_q_top(left_visits, left_q)
    right_top, right_gap = visited_q_top(right_visits, right_q)
    cross_gap = float(abs(right_q[right_top] - right_q[left_top]))
    near_tie = bool(min(left_gap, right_gap, cross_gap) < meaningful_q_gap)
    stable = bool(
        correlation >= min_rank_correlation
        and (left_top == right_top or cross_gap < meaningful_q_gap)
    )
    return {
        "comparable": True,
        "stable": stable,
        "near_tie": near_tie,
        "rank_correlation": correlation,
        "shared_actions": int(len(indices)),
        "left_top": left_top,
        "right_top": right_top,
        "left_q_gap": left_gap,
        "right_q_gap": right_gap,
        "cross_q_gap_at_right": cross_gap,
        "q_linf": float(np.max(np.abs(left_q[indices] - right_q[indices]))),
    }
