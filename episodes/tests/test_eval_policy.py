"""Offline tests for eval_policy's seed parsing and summary (no machine)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_policy import format_summary, parse_seeds, summarize, wilson

STAGES = ["reached", "grasped", "lifted"]


def record(seed: int, highest: str | None, success: bool) -> dict:
    reached = {s: False for s in STAGES}
    times = {s: None for s in STAGES}
    if highest is not None:
        for index, s in enumerate(STAGES[: STAGES.index(highest) + 1]):
            reached[s] = True
            times[s] = 10.0 * (index + 1)
    return {
        "seed": seed,
        "verdict": {
            "success": success,
            "success_once": success,
            "stages_reached": reached,
            "stage_times_sec": times,
            "highest_stage": highest,
        },
        "run": {"state": "completed", "error": "", "chunks_inferred": 12, "duration_sec": 60.0},
    }


def test_parse_seeds_range_and_list() -> None:
    assert parse_seeds("3000-3002") == [3000, 3001, 3002]
    assert parse_seeds("1, 5,7-8") == [1, 5, 7, 8]
    with pytest.raises(ValueError):
        parse_seeds(" ")


def test_wilson_matches_known_values() -> None:
    lo, hi = wilson(10, 10)
    assert lo == pytest.approx(0.722, abs=1e-3) and hi == 1.0
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.237, abs=1e-3) and hi == pytest.approx(0.763, abs=1e-3)


def test_summary_counts_stages_and_failures() -> None:
    records = [
        record(1, "lifted", True),
        record(2, "grasped", False),
        record(3, None, False),
    ]
    summary = summarize(records, STAGES)
    assert summary["success"] == 1 and summary["n"] == 3
    assert summary["stage_counts"] == {"reached": 2, "grasped": 2, "lifted": 1}
    assert summary["stage_time_sec_median"] == {"reached": 10.0, "grasped": 20.0, "lifted": 30.0}
    assert [f["seed"] for f in summary["failures"]] == [2, 3]
    text = format_summary(summary, records)
    assert "success 1/3" in text and "| 2 | no | grasped |" in text
