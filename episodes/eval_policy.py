"""Score a trained policy on seeded rollouts with the checker that labelled its training data.

For each seed: home the arm, place the block (and pad) exactly where
oracle_episode.py would for that seed, run the arm-policy module's `execute`
command, sample simulator state while it runs, and score the history with
task_check.PickTask. The same checker decides which scripted episodes are kept
for training, so data filtering and evaluation cannot drift apart.

    uv run eval_policy.py <machine address> --seeds 3000-3009 --max-chunks 20
    uv run eval_policy.py <machine address> --seeds 3000-3009 --place-prop place_pad

Pick eval seeds that batch_episodes.py did not use. 10 seeds is a smoke test
(a Wilson 95% interval at n=10 spans about 45 points); compare two models with
25 to 50 seeds each. Writes runs/eval/<label>/seed-<seed>.json and summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from viam.components.arm import Arm, JointPositions
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.services.motion import MotionClient

from oracle_episode import (
    HOME_JOINT_DEGREES,
    PICK_YAW_RANGE_DEG,
    PickConfig,
    connect,
    resolve_pick_region,
    sample_place_pose,
    sample_prop_pose,
)
from task_check import DEFAULT_HOLD_SAMPLES, PickTask, Snapshot, Verdict, snapshot

DEFAULT_SEEDS = "3000-3009"
DEFAULT_MAX_CHUNKS = 20
DEFAULT_POLL_SEC = 0.5
DEFAULT_TIMEOUT_SEC = 600.0
GRIPPER_OPEN_SETTLE_SEC = 1.0
POST_RUN_SETTLE_SEC = 0.5


def parse_seeds(value: str) -> list[int]:
    """'3000-3009' or '3000,3002,3004' or a mix."""
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            seeds.extend(range(lo, hi + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("no seeds given")
    return seeds


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


async def prepare_scene(
    machine: Any,
    world: Any,
    arm: Any,
    gripper: Any,
    config: PickConfig,
    seed: int,
) -> tuple[PickTask, dict[str, Any]]:
    """Home the arm and place the props for `seed` the way the oracle does.

    Returns the task checker for this scene and a JSON-friendly scene record."""
    await arm.move_to_joint_positions(JointPositions(values=list(HOME_JOINT_DEGREES)))
    await gripper.open()
    await asyncio.sleep(GRIPPER_OPEN_SETTLE_SEC)
    region_min, region_max = await resolve_pick_region(machine, config.arm_name)
    config = PickConfig(
        **{
            **config.__dict__,
            "pick_region_min_m": region_min,
            "pick_region_max_m": region_max,
        }
    )
    position, orientation = sample_prop_pose(
        seed=seed,
        region_min=config.pick_region_min_m,
        region_max=config.pick_region_max_m,
        yaw_range_deg=PICK_YAW_RANGE_DEG,
        prop_width=config.prop_width_m,
        prop_depth=config.prop_depth_m,
        prop_height=config.prop_height_m,
    )
    applied = await world.do_command(
        {
            "command": "set_prop_pose",
            "prop": config.prop_name,
            "position": list(position),
            "orientation_wxyz": list(orientation),
        }
    )
    target_position = tuple(float(v) for v in applied["position"])
    scene: dict[str, Any] = {
        "seed": seed,
        "prop": config.prop_name,
        "prop_pose": {
            "position": list(applied["position"]),
            "orientation_wxyz": list(applied["orientation_wxyz"]),
        },
    }
    pad_position: tuple[float, float, float] | None = None
    if config.place_prop:
        pad_position, pad_orientation = sample_place_pose(seed, config, target_position)
        applied_pad = await world.do_command(
            {
                "command": "set_prop_pose",
                "prop": config.place_prop,
                "position": list(pad_position),
                "orientation_wxyz": list(pad_orientation),
            }
        )
        pad_position = tuple(float(v) for v in applied_pad["position"])
        scene["pad"] = config.place_prop
        scene["pad_pose"] = {
            "position": list(applied_pad["position"]),
            "orientation_wxyz": list(applied_pad["orientation_wxyz"]),
        }
    task = PickTask(
        prop=config.prop_name,
        initial_z_m=target_position[2],
        minimum_lift_m=config.minimum_lift_m,
        prop_height_m=config.prop_height_m,
        pad_position_m=pad_position,
    )
    return task, scene


def _snapshot_record(s: Snapshot) -> dict[str, Any]:
    return {
        "t_sec": round(s.t_sec, 3),
        "props": {name: [round(v, 4) for v in pos] for name, pos in s.props.items()},
        "gripper_position": (
            [round(v, 4) for v in s.gripper_position] if s.gripper_position else None
        ),
        "holding": s.holding,
        "firm_grasp": s.firm_grasp,
    }


async def rollout(
    policy: Any,
    world: Any,
    gripper: Any,
    motion: Any,
    config: PickConfig,
    task: PickTask,
    command: Mapping[str, Any],
    *,
    poll_sec: float,
    timeout_sec: float,
    stop_on_success: bool,
) -> tuple[Verdict, list[Snapshot], dict[str, Any]]:
    """Run one `execute` and sample state until it returns, times out, or succeeds."""
    props = [config.prop_name]
    t0 = time.monotonic()
    history: list[Snapshot] = []
    checker = task.task_check()
    run = asyncio.ensure_future(policy.do_command(dict(command)))
    stopped_for: str | None = None
    while not run.done():
        history.append(
            await snapshot(
                world, gripper, motion, config.gripper_name, props, time.monotonic() - t0
            )
        )
        elapsed = time.monotonic() - t0
        if stop_on_success and checker.check(history).success:
            stopped_for = "success"
        elif elapsed > timeout_sec:
            stopped_for = "timeout"
        if stopped_for is not None:
            await policy.do_command({"command": "stop"})
            break
        await asyncio.sleep(poll_sec)
    result: Mapping[str, Any] = {}
    error = ""
    try:
        result = await asyncio.wait_for(run, timeout=60.0)
    except Exception as exc:  # the module reports its own failure state too
        error = f"{type(exc).__name__}: {exc}"
    # Let the scene settle after the last chunk, then sample the hold window.
    await asyncio.sleep(POST_RUN_SETTLE_SEC)
    for _ in range(task.hold_samples):
        history.append(
            await snapshot(
                world, gripper, motion, config.gripper_name, props, time.monotonic() - t0
            )
        )
        await asyncio.sleep(poll_sec)
    status = await policy.do_command({"command": "status"})
    inference_ms = [float(v) for v in status.get("inference_ms") or []]
    run_record = {
        "state": result.get("state") or status.get("state"),
        "stopped_for": stopped_for,
        "error": error or str(result.get("error") or status.get("last_error") or ""),
        "chunks_inferred": int(status.get("chunks_inferred") or 0),
        "rejected_chunks": int(status.get("rejected_chunks") or 0),
        "inference_ms_median": (
            round(statistics.median(inference_ms)) if inference_ms else None
        ),
        "duration_sec": round(time.monotonic() - t0, 1),
    }
    return checker.check(history), history, run_record


def summarize(records: Sequence[Mapping[str, Any]], stages: Sequence[str]) -> dict[str, Any]:
    n = len(records)
    k = sum(bool(r["verdict"]["success"]) for r in records)
    k_once = sum(bool(r["verdict"]["success_once"]) for r in records)
    lo, hi = wilson(k, n)
    stage_counts = {
        s: sum(bool(r["verdict"]["stages_reached"].get(s)) for r in records) for s in stages
    }
    stage_time_med = {}
    for s in stages:
        times = [
            r["verdict"]["stage_times_sec"][s]
            for r in records
            if r["verdict"]["stage_times_sec"].get(s) is not None
        ]
        stage_time_med[s] = round(statistics.median(times), 1) if times else None
    durations = [r["run"]["duration_sec"] for r in records]
    return {
        "n": n,
        "success": k,
        "success_once": k_once,
        "success_rate": round(k / n, 3) if n else None,
        "wilson95": [round(lo, 3), round(hi, 3)],
        "stage_counts": stage_counts,
        "stage_time_sec_median": stage_time_med,
        "duration_sec_median": round(statistics.median(durations), 1) if durations else None,
        "failures": [
            {
                "seed": r["seed"],
                "highest_stage": r["verdict"]["highest_stage"],
                "state": r["run"]["state"],
                "error": r["run"]["error"][:80],
            }
            for r in records
            if not r["verdict"]["success"]
        ],
    }


def format_summary(summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        (
            f"success {summary['success']}/{summary['n']} "
            f"(95% CI {summary['wilson95'][0]:.2f}-{summary['wilson95'][1]:.2f}); "
            f"success at any point {summary['success_once']}/{summary['n']}"
        ),
        "stage reached: "
        + ", ".join(f"{s} {c}" for s, c in summary["stage_counts"].items()),
        "median time to stage (s): "
        + ", ".join(f"{s} {t}" for s, t in summary["stage_time_sec_median"].items()),
        "",
        "| seed | success | highest stage | chunks | duration s | state | error |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        v, run = r["verdict"], r["run"]
        lines.append(
            f"| {r['seed']} | {'YES' if v['success'] else 'no'} | {v['highest_stage']} | "
            f"{run['chunks_inferred']} | {run['duration_sec']} | {run['state']} | "
            f"{run['error'][:40]} |"
        )
    return "\n".join(lines)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.output_dir)
    config = PickConfig(
        prop_name=args.prop,
        arm_name=args.arm,
        gripper_name=args.gripper,
        place_prop=args.place_prop or None,
    )
    command: dict[str, Any] = {
        "command": "execute",
        "max_chunks": args.max_chunks,
        "progress_prop": args.prop,
    }
    if args.prompt:
        command["prompt"] = args.prompt
    if args.action_prefix_steps is not None:
        command["action_prefix_steps"] = args.action_prefix_steps
    if args.conditioner_algorithm:
        command["conditioner_algorithm"] = args.conditioner_algorithm
    if args.action_smoothing_window is not None:
        command["action_smoothing_window"] = args.action_smoothing_window

    records: list[dict[str, Any]] = []
    async with await connect(
        args.address,
        viam_config_path=args.viam_config,
        disable_webrtc=args.disable_webrtc,
    ) as machine:
        policy = Generic.from_robot(machine, args.policy)
        world = Generic.from_robot(machine, args.world)
        gripper = Gripper.from_robot(machine, args.gripper)
        motion = MotionClient.from_robot(machine, args.motion)
        arm = Arm.from_robot(machine, args.arm)
        stages: list[str] = []
        for seed in seeds:
            path = out_dir / f"seed-{seed}.json"
            if path.exists() and not args.rerun:
                records.append(json.loads(path.read_text(encoding="utf-8")))
                print(f"seed {seed}: cached")
                continue
            task, scene = await prepare_scene(machine, world, arm, gripper, config, seed)
            stages = [stage.name for stage in task.task_check().stages]
            verdict, history, run = await rollout(
                policy,
                world,
                gripper,
                motion,
                config,
                task,
                command,
                poll_sec=args.poll_sec,
                timeout_sec=args.timeout_sec,
                stop_on_success=not args.run_all_chunks,
            )
            record = {
                "seed": seed,
                "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "scene": scene,
                "task": task.to_dict(),
                "command": command,
                "verdict": verdict.to_dict(),
                "run": run,
                "history": [_snapshot_record(s) for s in history],
            }
            _write_json(path, record)
            records.append(record)
            print(
                f"seed {seed}: {'SUCCESS' if verdict.success else 'fail'} "
                f"(stage {verdict.highest_stage}, {run['chunks_inferred']} chunks, "
                f"{run['duration_sec']} s)"
            )
        if not stages and records:
            stages = records[0]["task"]["stages"]
    summary = summarize(records, stages)
    summary["seeds"] = seeds
    summary["command"] = command
    _write_json(out_dir / "summary.json", summary)
    print(format_summary(summary, records))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("address", help="machine address from the Viam CONNECT tab")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="range 'a-b' and/or comma list")
    parser.add_argument("--policy", default="policy", help="arm-policy resource name")
    parser.add_argument("--world", default="sim-world")
    parser.add_argument("--gripper", default="pick-gripper")
    parser.add_argument("--motion", default="builtin")
    parser.add_argument("--prop", default="pick_block")
    parser.add_argument("--arm", default="pick-arm")
    parser.add_argument("--place-prop", default="", help="pad prop name; enables the place stages")
    parser.add_argument("--prompt", default="", help="override the module's configured prompt")
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS)
    parser.add_argument("--action-prefix-steps", type=int)
    parser.add_argument("--conditioner-algorithm", choices=("quintic", "toppra"))
    parser.add_argument("--action-smoothing-window", type=int)
    parser.add_argument("--poll-sec", type=float, default=DEFAULT_POLL_SEC)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument(
        "--run-all-chunks",
        action="store_true",
        help="keep executing after success is held (default stops the run early)",
    )
    parser.add_argument("--rerun", action="store_true", help="ignore cached seed files")
    parser.add_argument(
        "--output-dir",
        default=f"runs/eval/{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument("--viam-config", type=Path, default=Path("/etc/viam.json"))
    parser.add_argument("--disable-webrtc", action="store_true")
    args = parser.parse_args()
    summary = asyncio.run(_main(args))
    return 0 if summary["n"] and summary["success"] == summary["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
