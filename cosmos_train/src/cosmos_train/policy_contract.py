"""Validate a prepared six-joint dataset and package its runtime policy contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

JOINT_COUNT = 6
GRIPPER_CONVENTION = "0=open, 1=closed"


def _merge_missing_metadata(
    policy: dict[str, Any], overlay: dict[str, Any], overlay_path: Path
) -> dict[str, Any]:
    merged = dict(policy)
    for key, value in overlay.items():
        if key in merged and merged[key] != value:
            raise ValueError(
                f"{overlay_path}: metadata overlay conflicts with dataset field {key!r}"
            )
        merged[key] = value
    return merged


def load_policy_metadata(
    dataset: Path, *, metadata_overlay: Path | None = None
) -> dict[str, Any]:
    policy_path = dataset / "meta/cosmos_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if metadata_overlay is not None:
        overlay = json.loads(metadata_overlay.read_text(encoding="utf-8"))
        if not isinstance(overlay, dict):
            raise TypeError(
                f"{metadata_overlay}: metadata overlay must be a JSON object"
            )
        policy = _merge_missing_metadata(policy, overlay, metadata_overlay)

    embodiment = policy.get("embodiment")
    if not isinstance(embodiment, dict):
        raise TypeError(f"{policy_path}: embodiment must be an object")
    arm_model = embodiment.get("arm_model")
    gripper_model = embodiment.get("gripper_model")
    gripper_mode = embodiment.get("gripper_mode")
    joint_names = embodiment.get("joint_names")
    if not isinstance(arm_model, str) or not arm_model:
        raise ValueError(f"{policy_path}: embodiment.arm_model must be non-empty")
    if not isinstance(gripper_model, str) or not gripper_model:
        raise ValueError(f"{policy_path}: embodiment.gripper_model must be non-empty")
    if gripper_mode not in ("binary", "position"):
        raise ValueError(f"{policy_path}: embodiment.gripper_mode is invalid")
    if (
        not isinstance(joint_names, list)
        or len(joint_names) != JOINT_COUNT
        or any(not isinstance(name, str) or not name for name in joint_names)
        or len(set(joint_names)) != JOINT_COUNT
    ):
        raise ValueError(
            f"{policy_path}: embodiment must declare {JOINT_COUNT} unique joint names"
        )

    gripper_units = "binary" if gripper_mode == "binary" else "normalized"
    expected_action = {
        "key": "action",
        "type": "absolute_joint_target",
        "shape": [JOINT_COUNT + 1],
        "names": [*joint_names, "gripper"],
        "units": [*["rad"] * JOINT_COUNT, gripper_units],
        "fps": policy.get("fps"),
        "gripper_convention": GRIPPER_CONVENTION,
    }
    if policy.get("native_action_schema") != expected_action:
        raise ValueError(
            f"{policy_path}: native action schema does not match the declared embodiment"
        )
    expected_framework = {
        "framework_loader": "DROIDLeRobotDataset:joint_pos",
        "framework_action_space": "joint_pos",
        "framework_joint_count": JOINT_COUNT,
        "framework_use_state": True,
    }
    mismatches = {
        name: (policy.get(name), expected)
        for name, expected in expected_framework.items()
        if policy.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"{policy_path}: framework contract mismatch: {mismatches}")
    if not isinstance(policy.get("fps"), (int, float)) or policy["fps"] <= 0:
        raise ValueError(f"{policy_path}: fps must be positive")
    return policy


def build_runtime_contract(
    dataset: Path,
    *,
    action_chunk_size: int = 32,
    domain_name: str = "droid_lerobot",
    metadata_overlay: Path | None = None,
) -> dict[str, Any]:
    if (
        isinstance(action_chunk_size, bool)
        or not isinstance(action_chunk_size, int)
        or action_chunk_size <= 0
    ):
        raise ValueError("action_chunk_size must be a positive integer")
    if not domain_name:
        raise ValueError("domain_name must not be empty")
    policy = load_policy_metadata(dataset, metadata_overlay=metadata_overlay)
    video_keys = policy.get("video_keys")
    if (
        not isinstance(video_keys, list)
        or not video_keys
        or any(not isinstance(key, str) or not key for key in video_keys)
    ):
        raise ValueError("dataset policy video_keys must be a non-empty string list")
    camera_aliases = {
        key: key.replace("observation.images.", "observation/", 1) for key in video_keys
    }
    return {
        "schema_version": 3,
        "embodiment": policy["embodiment"],
        "action_schema": policy["native_action_schema"],
        "observation_schema": {
            "joint_names": policy["embodiment"]["joint_names"],
            "joint_units": "rad",
            "gripper_units": (
                "binary"
                if policy["embodiment"]["gripper_mode"] == "binary"
                else "normalized"
            ),
            "gripper_convention": GRIPPER_CONVENTION,
            "camera_keys": video_keys,
            "camera_aliases": camera_aliases,
        },
        "framework": {
            "loader": policy["framework_loader"],
            "action_space": policy["framework_action_space"],
            "joint_count": policy["framework_joint_count"],
            "use_state": policy["framework_use_state"],
        },
        "transform": {
            "action_normalization": None,
            "viewpoint": "concat_view",
            "resolution": "480",
            "format_prompt_as_json": True,
            "append_viewpoint_info": True,
            "append_duration_fps_timestamps": True,
            "append_resolution_info": True,
            "append_idle_frames": False,
            "camera_keys": video_keys,
            "camera_aliases": camera_aliases,
        },
        "action_chunk_size": action_chunk_size,
        "conditioning_fps": float(policy["fps"]),
        "domain_name": domain_name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--action-chunk-size", type=int, default=32)
    parser.add_argument("--domain-name", default="droid_lerobot")
    parser.add_argument(
        "--metadata-overlay",
        type=Path,
        help="JSON fields to add when pairing a legacy export whose dataset metadata predates the explicit embodiment contract",
    )
    args = parser.parse_args()
    contract = build_runtime_contract(
        args.dataset,
        action_chunk_size=args.action_chunk_size,
        domain_name=args.domain_name,
        metadata_overlay=args.metadata_overlay,
    )
    if args.output is not None:
        args.output.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(contract, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
