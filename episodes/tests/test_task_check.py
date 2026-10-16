"""Offline tests for the task checker: no simulator, synthetic snapshots.

These are the "predicate audit" every generated checker should pass: false at
reset, true after a correct episode, false after a wrong one, and tolerant to
small perturbations of the final state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_check import (
    PLACE_PAD_HEIGHT_M,
    PickTask,
    Snapshot,
)

BLOCK = "pick_block"
H = 0.0313
Z0 = H / 2.0
PAD = (0.40, 0.10, PLACE_PAD_HEIGHT_M / 2.0)
PAD_REST_Z = PAD[2] + PLACE_PAD_HEIGHT_M / 2.0 + H / 2.0


def snap(
    t: float,
    block: tuple[float, float, float],
    gripper: tuple[float, float, float] | None,
    holding: bool,
) -> Snapshot:
    return Snapshot(t_sec=t, props={BLOCK: block}, gripper_position=gripper, holding=holding)


def pick_task(pad: bool = False) -> PickTask:
    return PickTask(
        prop=BLOCK,
        initial_z_m=Z0,
        minimum_lift_m=0.10,
        prop_height_m=H,
        pad_position_m=PAD if pad else None,
        hold_samples=3,
    )


def test_reset_state_is_not_success() -> None:
    history = [snap(t, (0.3, 0.0, Z0), (0.2, 0.0, 0.3), False) for t in range(4)]
    verdict = pick_task().check(history)
    assert not verdict.success
    assert not verdict.success_once
    assert verdict.highest_stage is None


def test_successful_pick_reaches_every_stage_in_order() -> None:
    history = [
        snap(0, (0.3, 0.0, Z0), (0.2, 0.0, 0.3), False),
        snap(1, (0.3, 0.0, Z0), (0.3, 0.0, Z0 + 0.05), False),  # reached
        snap(2, (0.3, 0.0, Z0), (0.3, 0.0, Z0), True),  # grasped
        snap(3, (0.3, 0.0, Z0 + 0.12), (0.3, 0.0, Z0 + 0.12), True),  # lifted
        snap(4, (0.3, 0.0, Z0 + 0.20), (0.3, 0.0, Z0 + 0.20), True),
        snap(5, (0.3, 0.0, Z0 + 0.20), (0.3, 0.0, Z0 + 0.20), True),
    ]
    verdict = pick_task().check(history)
    assert verdict.success and verdict.success_once
    assert verdict.highest_stage == "lifted"
    assert verdict.stage_times_sec == {"reached": 1, "grasped": 2, "lifted": 3}


def test_single_frame_lift_does_not_count() -> None:
    """A bounce: the block clears the threshold once, then drops."""
    history = [
        snap(0, (0.3, 0.0, Z0), (0.3, 0.0, Z0), True),
        snap(1, (0.3, 0.0, Z0 + 0.11), (0.3, 0.0, Z0 + 0.11), True),
        snap(2, (0.3, 0.0, Z0), (0.3, 0.0, Z0 + 0.11), False),
        snap(3, (0.3, 0.0, Z0), (0.3, 0.0, Z0 + 0.11), False),
    ]
    verdict = pick_task().check(history)
    assert not verdict.success and not verdict.success_once
    assert verdict.stages_reached["lifted"]
    assert verdict.highest_stage == "lifted"


def test_lift_then_drop_separates_once_from_end() -> None:
    held = [snap(t, (0.3, 0.0, Z0 + 0.15), (0.3, 0.0, Z0 + 0.15), True) for t in range(3)]
    dropped = [snap(t, (0.3, 0.0, Z0), (0.3, 0.0, Z0 + 0.15), False) for t in range(3, 6)]
    verdict = pick_task().check(held + dropped)
    assert verdict.success_once and not verdict.success


def test_place_requires_release_and_rest_on_pad() -> None:
    task = pick_task(pad=True)
    over = [snap(t, (PAD[0], PAD[1], 0.15), (PAD[0], PAD[1], 0.15), True) for t in range(3)]
    still_held = over + [
        snap(t, (PAD[0], PAD[1], PAD_REST_Z), (PAD[0], PAD[1], PAD_REST_Z), True)
        for t in range(3, 6)
    ]
    assert not task.check(still_held).success
    assert task.check(still_held).highest_stage == "over_pad"
    released = over + [
        snap(t, (PAD[0], PAD[1], PAD_REST_Z), (PAD[0], PAD[1], 0.10), False)
        for t in range(3, 6)
    ]
    assert task.check(released).success
    assert task.check(released).highest_stage == "placed"


@pytest.mark.parametrize("dx_m, expected", [(0.003, True), (0.03, True), (0.06, False)])
def test_place_tolerance(dx_m: float, expected: bool) -> None:
    """A few mm off center still counts; well off the 8 cm pad does not."""
    task = pick_task(pad=True)
    history = [
        snap(t, (PAD[0] + dx_m, PAD[1], PAD_REST_Z), (PAD[0], PAD[1], 0.10), False)
        for t in range(3)
    ]
    assert task.check(history).success is expected


def test_hovering_block_is_not_placed() -> None:
    task = pick_task(pad=True)
    history = [
        snap(t, (PAD[0], PAD[1], PAD_REST_Z + 0.03), (PAD[0], PAD[1], 0.10), False)
        for t in range(3)
    ]
    assert not task.check(history).success


def test_check_rejects_empty_history() -> None:
    with pytest.raises(ValueError):
        pick_task().check([])
