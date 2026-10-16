"""Audit the task checker on the live simulator before generating a dataset.

Three tests, all scored by task_check.PickTask exactly as the oracle and
eval_policy.py use it:

  reset      the fresh scene must not count as success
  oracle     one noise-free oracle episode must count as success
  wrong      a deliberately wrong ending must not count: for pick, open the
             gripper after the lift; for place, move the block 6 cm off the
             pad (and a 3 mm nudge must still pass)

    uv run check_task.py <machine address> [--seed 3000] [--place-prop place_pad]

Exit code 0 when every test passes. Run tests/test_task_check.py for the
offline predicate tests that need no machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from viam.components.arm import Arm
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.services.motion import MotionClient

from eval_policy import prepare_scene
from oracle_episode import PickConfig, connect, execute_pick, resolve_pick_region
from task_check import PickTask, Snapshot, snapshot

SETTLE_SEC = 1.0
SAMPLE_SEC = 0.2
OFF_PAD_M = 0.06
NUDGE_M = 0.003


async def _sample(
    world: Any, gripper: Any, motion: Any, config: PickConfig, task: PickTask
) -> list[Snapshot]:
    history: list[Snapshot] = []
    for index in range(task.hold_samples):
        history.append(
            await snapshot(
                world, gripper, motion, config.gripper_name, [config.prop_name], float(index)
            )
        )
        await asyncio.sleep(SAMPLE_SEC)
    return history


async def _move_block(world: Any, prop: str, position: tuple[float, float, float]) -> None:
    current = await world.do_command({"command": "get_prop_pose", "prop": prop})
    await world.do_command(
        {
            "command": "set_prop_pose",
            "prop": prop,
            "position": list(position),
            "orientation_wxyz": list(current["pose"]["orientation_wxyz"]),
        }
    )
    await asyncio.sleep(SETTLE_SEC)


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    results: dict[str, Any] = {}
    config = PickConfig(
        prop_name=args.prop,
        arm_name=args.arm,
        gripper_name=args.gripper,
        place_prop=args.place_prop or None,
        noise=False,
    )
    async with await connect(
        args.address, viam_config_path=args.viam_config, disable_webrtc=args.disable_webrtc
    ) as machine:
        world = Generic.from_robot(machine, args.world)
        gripper = Gripper.from_robot(machine, args.gripper)
        motion = MotionClient.from_robot(machine, args.motion)
        arm = Arm.from_robot(machine, args.arm)

        task, scene = await prepare_scene(machine, world, arm, gripper, config, args.seed)
        verdict = task.check(await _sample(world, gripper, motion, config, task))
        results["reset"] = {
            "pass": not verdict.success and verdict.highest_stage is None,
            "verdict": verdict.to_dict(),
        }

        region_min, region_max = await resolve_pick_region(machine, args.arm)
        oracle_config = PickConfig(
            **{**config.__dict__, "pick_region_min_m": region_min, "pick_region_max_m": region_max}
        )
        result = await execute_pick(world, gripper, motion, arm, args.seed, oracle_config)
        oracle_verdict = result["verdict"]
        results["oracle"] = {
            "pass": result["outcome"] == "success" and bool(oracle_verdict and oracle_verdict["success"]),
            "outcome": result["outcome"],
            "verdict": oracle_verdict,
        }

        pad = result["task"]["pad_position_m"]
        if pad is None:
            # Pick: dropping the block after the lift must fail the check.
            await gripper.open()
            await asyncio.sleep(SETTLE_SEC)
            verdict = task.check(await _sample(world, gripper, motion, config, task))
            results["wrong"] = {"pass": not verdict.success, "verdict": verdict.to_dict()}
        else:
            placed_task = PickTask(**{**task.__dict__, "pad_position_m": tuple(pad)})
            rest = await world.do_command({"command": "get_prop_pose", "prop": args.prop})
            rest_z = float(rest["pose"]["position"][2])
            await _move_block(world, args.prop, (pad[0] + NUDGE_M, pad[1], rest_z))
            nudged = placed_task.check(await _sample(world, gripper, motion, config, placed_task))
            await _move_block(world, args.prop, (pad[0] + OFF_PAD_M, pad[1], rest_z))
            off = placed_task.check(await _sample(world, gripper, motion, config, placed_task))
            results["wrong"] = {
                "pass": nudged.success and not off.success,
                "nudged_3mm": nudged.to_dict(),
                "off_pad_6cm": off.to_dict(),
            }
    results["all_pass"] = all(v["pass"] for k, v in results.items() if k != "all_pass")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("address")
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--world", default="sim-world")
    parser.add_argument("--gripper", default="pick-gripper")
    parser.add_argument("--motion", default="builtin")
    parser.add_argument("--prop", default="pick_block")
    parser.add_argument("--arm", default="pick-arm")
    parser.add_argument("--place-prop", default="")
    parser.add_argument("--viam-config", type=Path, default=Path("/etc/viam.json"))
    parser.add_argument("--disable-webrtc", action="store_true")
    args = parser.parse_args()
    results = asyncio.run(_main(args))
    for name in ("reset", "oracle", "wrong"):
        print(f"{name}: {'PASS' if results[name]['pass'] else 'FAIL'}")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if results["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
