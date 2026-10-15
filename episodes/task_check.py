"""Task success checker shared by the oracle (episode labelling) and the policy eval.

The checker is a pure function of simulator state, so the same code decides
whether a scripted episode is kept for training and whether a trained policy
succeeded on a rollout. Keep it sparse (final-state predicates over object
poses and gripper state) and never let the controller decide its own outcome.

Pick:            reached -> grasped -> lifted            (success = lifted, held)
Pick and place:  reached -> grasped -> lifted -> over_pad -> placed
                                                         (success = placed, held)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Vec3 = tuple[float, float, float]

# The gripper origin must be this close in xy to the block, and no higher than
# this above it, to count as "reached". Generous on purpose: this is a stage
# marker for partial credit, not the success test.
REACH_XY_M = 0.03
REACH_ABOVE_M = 0.10
# A final-state predicate must hold for this many consecutive samples so a
# bouncing or slipping block is not scored as success (SimplerEnv uses 5 frames).
DEFAULT_HOLD_SAMPLES = 3
# Pad geometry, mirrored from oracle_episode so the checker has no oracle import.
PLACE_PAD_WIDTH_M = 0.08
PLACE_PAD_HEIGHT_M = 0.002
PLACE_REST_TOLERANCE_M = 0.01


@dataclass(frozen=True)
class Snapshot:
    """One sample of the simulator state, in the world frame, in meters."""

    t_sec: float
    props: Mapping[str, Vec3]
    gripper_position: Vec3 | None
    holding: bool
    firm_grasp: bool | None = None
    gripper_meta: Mapping[str, Any] = field(default_factory=dict)


def xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def block_on_pad(
    block_position: Sequence[float],
    pad_position: Sequence[float],
    block_height_m: float,
) -> bool:
    """True when the block's center is over the pad and it rests flat on it."""
    half = PLACE_PAD_WIDTH_M / 2.0
    rest_z = pad_position[2] + PLACE_PAD_HEIGHT_M / 2.0 + block_height_m / 2.0
    return (
        abs(block_position[0] - pad_position[0]) <= half
        and abs(block_position[1] - pad_position[1]) <= half
        and abs(block_position[2] - rest_z) <= PLACE_REST_TOLERANCE_M
    )


Predicate = Callable[[Snapshot], bool]


@dataclass(frozen=True)
class Stage:
    name: str
    predicate: Predicate


@dataclass(frozen=True)
class Verdict:
    success: bool
    success_once: bool
    stages_reached: Mapping[str, bool]
    stage_times_sec: Mapping[str, float | None]
    highest_stage: str | None
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "success_once": self.success_once,
            "stages_reached": dict(self.stages_reached),
            "stage_times_sec": dict(self.stage_times_sec),
            "highest_stage": self.highest_stage,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class TaskCheck:
    """Ordered stages; the last stage is the success condition."""

    stages: Sequence[Stage]
    hold_samples: int = DEFAULT_HOLD_SAMPLES

    @property
    def final(self) -> Stage:
        return self.stages[-1]

    def check(self, history: Sequence[Snapshot]) -> Verdict:
        if not history:
            raise ValueError("check needs at least one snapshot")
        reached: dict[str, bool] = {}
        times: dict[str, float | None] = {}
        for stage in self.stages:
            first = next(
                (s.t_sec for s in history if stage.predicate(s)), None
            )
            reached[stage.name] = first is not None
            times[stage.name] = first
        highest = next(
            (stage.name for stage in reversed(self.stages) if reached[stage.name]),
            None,
        )
        final_flags = [self.final.predicate(s) for s in history]
        n = self.hold_samples
        success = len(final_flags) >= n and all(final_flags[-n:])
        success_once = any(
            all(final_flags[i : i + n]) for i in range(len(final_flags) - n + 1)
        )
        return Verdict(
            success=success,
            success_once=success_once,
            stages_reached=reached,
            stage_times_sec=times,
            highest_stage=highest,
            samples=len(history),
        )


@dataclass(frozen=True)
class PickTask:
    """'pick up the block', optionally followed by placing it on a pad.

    `initial_z_m` is the block's resting height when the episode started;
    `pad_position_m` switches on the place stages.
    """

    prop: str
    initial_z_m: float
    minimum_lift_m: float
    prop_height_m: float
    pad_position_m: Vec3 | None = None
    hold_samples: int = DEFAULT_HOLD_SAMPLES
    reach_xy_m: float = REACH_XY_M
    reach_above_m: float = REACH_ABOVE_M
    extra: Mapping[str, Any] = field(default_factory=dict)

    def _block(self, s: Snapshot) -> Vec3:
        return s.props[self.prop]

    def reached(self, s: Snapshot) -> bool:
        if s.gripper_position is None:
            return False
        block = self._block(s)
        return (
            xy_distance(s.gripper_position, block) <= self.reach_xy_m
            and 0.0 <= s.gripper_position[2] - block[2] <= self.reach_above_m
        )

    def grasped(self, s: Snapshot) -> bool:
        return s.holding

    def lifted(self, s: Snapshot) -> bool:
        return s.holding and self._block(s)[2] >= self.initial_z_m + self.minimum_lift_m

    def over_pad(self, s: Snapshot) -> bool:
        assert self.pad_position_m is not None
        return s.holding and xy_distance(self._block(s), self.pad_position_m) <= (
            PLACE_PAD_WIDTH_M / 2.0
        )

    def placed(self, s: Snapshot) -> bool:
        assert self.pad_position_m is not None
        return not s.holding and block_on_pad(
            self._block(s), self.pad_position_m, self.prop_height_m
        )

    def task_check(self) -> TaskCheck:
        stages = [
            Stage("reached", self.reached),
            Stage("grasped", self.grasped),
            Stage("lifted", self.lifted),
        ]
        if self.pad_position_m is not None:
            stages += [Stage("over_pad", self.over_pad), Stage("placed", self.placed)]
        return TaskCheck(stages=stages, hold_samples=self.hold_samples)

    def check(self, history: Sequence[Snapshot]) -> Verdict:
        return self.task_check().check(history)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": "pick_and_place" if self.pad_position_m is not None else "pick",
            "prop": self.prop,
            "initial_z_m": self.initial_z_m,
            "minimum_lift_m": self.minimum_lift_m,
            "prop_height_m": self.prop_height_m,
            "pad_position_m": list(self.pad_position_m) if self.pad_position_m else None,
            "hold_samples": self.hold_samples,
            "stages": [stage.name for stage in self.task_check().stages],
        }


async def snapshot(
    world: Any,
    gripper: Any,
    motion: Any,
    gripper_name: str,
    props: Sequence[str],
    t_sec: float,
) -> Snapshot:
    """Read one Snapshot through the Viam APIs the oracle already uses."""
    positions: dict[str, Vec3] = {}
    for prop in props:
        measured = await world.do_command({"command": "get_prop_pose", "prop": prop})
        x, y, z = (float(v) for v in measured["pose"]["position"])
        positions[prop] = (x, y, z)
    pose = await motion.get_pose(gripper_name, "world")
    gripper_position = (
        pose.pose.x / 1000.0,
        pose.pose.y / 1000.0,
        pose.pose.z / 1000.0,
    )
    status = await gripper.is_holding_something()
    meta = dict(status.meta) if status.meta else {}
    firm = meta.get("firm_grasp")
    return Snapshot(
        t_sec=t_sec,
        props=positions,
        gripper_position=gripper_position,
        holding=bool(status.is_holding_something),
        firm_grasp=bool(firm) if isinstance(firm, bool) else None,
        gripper_meta=meta,
    )
