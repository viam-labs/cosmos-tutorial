"""Adapt the captured LeRobot v3 dataset to Cosmos Framework's DROID loader."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DROID_ROOT_NAME = "droid_lerobot_20260115_no_noops"
DROID_VIDEO_KEYS = (
    "observation.images.wrist_image_left",
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
)
COSMOS_VIDEO_ALIASES = {
    DROID_VIDEO_KEYS[0]: "observation.images.wrist",
    DROID_VIDEO_KEYS[1]: "observation.images.fixed",
    DROID_VIDEO_KEYS[2]: "observation.images.fixed",
}
COSMOS_FIXED_2_KEY = "observation.images.fixed_2"
LIVE_LITE6_VIDEO_ALIASES = {
    DROID_VIDEO_KEYS[0]: "observation.images.realsense_cam",
    DROID_VIDEO_KEYS[1]: "observation.images.camera_1",
    DROID_VIDEO_KEYS[2]: "observation.images.camera_1",
}
BOOKKEEPING_FEATURES = (
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def _fixed_list(values: np.ndarray) -> pa.Array:
    return pa.array(values.tolist(), type=pa.list_(pa.float32(), values.shape[1]))


def _matrix(table: pa.Table, name: str, width: int) -> np.ndarray:
    values = np.asarray(table[name].to_pylist(), dtype=np.float32)
    if values.shape != (table.num_rows, width) or not np.isfinite(values).all():
        raise ValueError(
            f"{name} must have shape ({table.num_rows}, {width}) and finite values"
        )
    return values


def _column(table: pa.Table, name: str) -> np.ndarray:
    values = np.asarray(table[name].to_pylist(), dtype=np.float32)
    if values.shape != (table.num_rows,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must have {table.num_rows} finite scalar values")
    return values


def _stats(values: np.ndarray) -> dict[str, Any]:
    values64 = values.astype(np.float64)
    return {
        "min": values64.min(axis=0).tolist(),
        "max": values64.max(axis=0).tolist(),
        "mean": values64.mean(axis=0).tolist(),
        "std": values64.std(axis=0).tolist(),
        "count": [len(values64)],
        "q01": np.quantile(values64, 0.01, axis=0).tolist(),
        "q10": np.quantile(values64, 0.10, axis=0).tolist(),
        "q50": np.quantile(values64, 0.50, axis=0).tolist(),
        "q90": np.quantile(values64, 0.90, axis=0).tolist(),
        "q99": np.quantile(values64, 0.99, axis=0).tolist(),
    }


def _tasks(source: Path) -> list[str]:
    table = pq.read_table(source / "meta/tasks.parquet")
    rows = sorted(table.to_pylist(), key=lambda row: row["task_index"])
    tasks = [row["task"] for row in rows]
    if (
        not tasks
        or any(not isinstance(task, str) or not task for task in tasks)
        or len(set(tasks)) != len(tasks)
        or [row["task_index"] for row in rows] != list(range(len(rows)))
    ):
        raise ValueError("source tasks must be unique with contiguous task indexes")
    return tasks


def _single_task(source: Path) -> str:
    tasks = _tasks(source)
    if len(tasks) != 1:
        raise ValueError(f"source must contain exactly one task, got {tasks}")
    return tasks[0]


def _training_feature(width: int, names: list[str]) -> dict[str, Any]:
    return {"dtype": "float32", "shape": [width], "names": names}


def _bookkeeping_feature(name: str) -> dict[str, Any]:
    dtype = "float32" if name == "timestamp" else "int64"
    return {"dtype": dtype, "shape": [1], "names": None}


def _canonical_video_feature(feature: dict[str, Any]) -> dict[str, Any]:
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError(
            f"video feature must have [height, width, channels], got {shape}"
        )
    source_info = feature.get("info")
    if not isinstance(source_info, dict):
        source_info = {}
    info: dict[str, Any] = {
        "has_audio": bool(source_info.get("has_audio", False)),
        "video.is_depth_map": bool(
            source_info.get(
                "video.is_depth_map", source_info.get("is_depth_map", False)
            )
        ),
    }
    for key in (
        "video.channels",
        "video.codec",
        "video.fps",
        "video.height",
        "video.pix_fmt",
        "video.width",
    ):
        if key in source_info:
            info[key] = source_info[key]
    return {
        "dtype": "video",
        "shape": shape,
        "names": ["height", "width", "channels"],
        "info": info,
    }


def _source_contract(
    source: Path, info: dict[str, Any], profile: str
) -> tuple[str, dict[str, Any], dict[str, str]]:
    features = info["features"]
    policy_path = source / "meta/cosmos_policy.json"
    selected = profile
    if selected == "auto":
        selected = "cosmos-policy" if policy_path.is_file() else "lite6-live"
    if selected == "cosmos-policy":
        if not policy_path.is_file():
            raise ValueError("cosmos-policy source is missing meta/cosmos_policy.json")
        policy = json.loads(policy_path.read_text())
        if policy.get("format") != "lerobot-v3" or policy.get("fps") != info.get("fps"):
            raise ValueError(
                "source is not the expected LeRobot v3 Cosmos Policy dataset"
            )
        if policy.get("action_space") != "joint_absolute":
            raise ValueError("source canonical action_space must be 'joint_absolute'")
        aliases = dict(COSMOS_VIDEO_ALIASES)
        if COSMOS_FIXED_2_KEY in features:
            aliases[DROID_VIDEO_KEYS[2]] = COSMOS_FIXED_2_KEY
    elif selected == "lite6-live":
        fps = info.get("fps")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or float(fps) <= 0.0
        ):
            raise ValueError("lite6-live source fps must be finite and positive")
        for key in ("observation.state", "action"):
            feature = features.get(key)
            if not isinstance(feature, dict) or feature.get("shape") != [7]:
                raise ValueError(f"lite6-live source {key} must have shape [7]")
            names = feature.get("names")
            if not isinstance(names, list) or names[-1:] != ["gripper"]:
                raise ValueError(f"lite6-live source {key} must end with gripper")
        task = _single_task(source)
        joint_names = [f"joint{index}" for index in range(1, 7)]
        policy = {
            "format": "lerobot-v3",
            "fps": fps,
            "task": task,
            "action_space": "joint_absolute",
            "embodiment": {
                "arm_model": "lite6",
                "gripper_model": "lite6_gripper",
                "gripper_mode": "binary",
                "joint_names": joint_names,
            },
            "native_action_schema": {
                "key": "action",
                "type": "absolute_joint_target",
                "shape": [7],
                "names": [*joint_names, "gripper"],
                "units": [*["rad"] * 6, "binary"],
                "fps": fps,
                "gripper_convention": "0=open, 1=closed",
            },
            "dataset_variant": "viam-live-height-inferred-gripper",
            "video_keys": list(LIVE_LITE6_VIDEO_ALIASES.values()),
            "auxiliary_video_keys": {},
        }
        aliases = LIVE_LITE6_VIDEO_ALIASES
    else:
        raise ValueError(f"unknown source profile {profile!r}")
    return selected, policy, aliases


def prepare(source: Path, output: Path, source_profile: str = "auto") -> dict[str, Any]:
    """Create the minimal six-joint schema consumed by the DROID joint_pos path."""
    if output.name != DROID_ROOT_NAME:
        raise ValueError(f"output directory must be named {DROID_ROOT_NAME!r}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    info = json.loads((source / "meta/info.json").read_text())
    features = info.get("features")
    if not isinstance(features, dict):
        raise TypeError("source info metadata is missing features")
    profile, policy, video_aliases = _source_contract(source, info, source_profile)
    tasks = _tasks(source)
    source_video_keys = sorted(
        name
        for name, feature in features.items()
        if isinstance(name, str)
        and isinstance(feature, dict)
        and feature.get("dtype") == "video"
    )
    used_video_keys = set(video_aliases.values())
    missing_video_features = used_video_keys.difference(source_video_keys)
    if missing_video_features:
        raise ValueError(
            f"source is missing video features: {sorted(missing_video_features)}"
        )
    dropped_auxiliary_video_keys = sorted(set(source_video_keys) - used_video_keys)

    embodiment = policy.get("embodiment")
    if not isinstance(embodiment, dict):
        raise TypeError("source policy metadata is missing embodiment")
    joint_names = embodiment.get("joint_names")
    if (
        not isinstance(joint_names, list)
        or len(joint_names) != 6
        or any(not isinstance(name, str) or not name for name in joint_names)
        or len(set(joint_names)) != 6
    ):
        raise ValueError("source embodiment must declare six unique joint_names")
    gripper_units = (
        "binary" if embodiment.get("gripper_mode") == "binary" else "normalized"
    )
    expected_schema = {
        "key": "action.joint_absolute" if profile == "cosmos-policy" else "action",
        "type": "absolute_joint_target",
        "shape": [7],
        "names": [*joint_names, "gripper"],
        "units": [*["rad"] * 6, gripper_units],
        "fps": info["fps"],
        "gripper_convention": "0=open, 1=closed",
    }
    if policy.get("native_action_schema") != expected_schema:
        raise ValueError("source native_action_schema does not match its embodiment")

    if profile == "cosmos-policy":
        required = {
            "source.joint_position_rad",
            "source.joint_target_rad",
            "source.gripper_command",
            "source.gripper_state",
        }
    else:
        required = {"observation.state", "action"}
    missing = required.difference(features)
    if missing:
        raise ValueError(f"source is missing features: {sorted(missing)}")

    temporary = output.with_name(f".{output.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"partial output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        shutil.copytree(source / "meta", temporary / "meta", dirs_exist_ok=True)
        batches: dict[str, list[np.ndarray]] = {}
        for source_file in sorted((source / "data").glob("chunk-*/*.parquet")):
            table = pq.read_table(source_file)
            if profile == "cosmos-policy":
                joint_state = _matrix(table, "source.joint_position_rad", 6)
                joint_action = _matrix(table, "source.joint_target_rad", 6)
                gripper_action = _column(table, "source.gripper_command")
                gripper_state = _column(table, "source.gripper_state")
            else:
                source_state = _matrix(table, "observation.state", 7)
                source_action = _matrix(table, "action", 7)
                joint_state, gripper_state = source_state[:, :6], source_state[:, 6]
                joint_action, gripper_action = source_action[:, :6], source_action[:, 6]
            if embodiment.get("gripper_mode") == "binary" and (
                not np.isin(gripper_state, (0.0, 1.0)).all()
                or not np.isin(gripper_action, (0.0, 1.0)).all()
            ):
                raise ValueError("binary gripper state/action must contain only 0 or 1")
            action = np.column_stack((joint_action, gripper_action)).astype(np.float32)
            state = np.column_stack((joint_state, gripper_state)).astype(np.float32)
            vectors = {
                "observation.state": state,
                "observation.state.joint_positions": joint_state,
                "observation.state.gripper_position": gripper_state[:, None],
                "action": action,
                "action.joint_position": joint_action,
            }
            for name, values in vectors.items():
                batches.setdefault(name, []).append(values)

            columns = {
                name: table[name]
                for name in BOOKKEEPING_FEATURES
                if name in table.column_names
            }
            columns.update(
                {
                    "observation.state": _fixed_list(state),
                    "observation.state.joint_positions": _fixed_list(joint_state),
                    "observation.state.gripper_position": pa.array(gripper_state),
                    "action": _fixed_list(action),
                    "action.joint_position": _fixed_list(joint_action),
                }
            )
            destination = temporary / source_file.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pydict(columns), destination, compression="snappy"
            )

        if not batches:
            raise ValueError("source contains no data parquet files")
        alias_features = {
            alias: _canonical_video_feature(features[original])
            for alias, original in video_aliases.items()
        }
        for alias, original in video_aliases.items():
            shutil.copytree(source / "videos" / original, temporary / "videos" / alias)
        generated_features = {
            "observation.state": _training_feature(7, [*joint_names, "gripper"]),
            "observation.state.joint_positions": _training_feature(6, joint_names),
            "observation.state.gripper_position": _training_feature(1, ["gripper"]),
            "action": _training_feature(7, [*joint_names, "gripper"]),
            "action.joint_position": _training_feature(6, joint_names),
        }
        info["features"] = {
            **{
                name: _bookkeeping_feature(name)
                for name in BOOKKEEPING_FEATURES
                if name in features
            },
            **generated_features,
            **alias_features,
        }
        info["robot_type"] = str(embodiment["arm_model"])
        info["total_tasks"] = len(tasks)
        for episode_file in sorted(
            (temporary / "meta/episodes").glob("chunk-*/*.parquet")
        ):
            table = pq.read_table(episode_file)
            columns = {
                name: table[name]
                for name in table.column_names
                if not name.startswith("videos/") and not name.startswith("stats/")
            }
            chunk_index = int(episode_file.parent.name.removeprefix("chunk-"))
            file_index = int(episode_file.stem.removeprefix("file-"))
            columns["meta/episodes/chunk_index"] = pa.array(
                np.full(table.num_rows, chunk_index, dtype=np.int64)
            )
            columns["meta/episodes/file_index"] = pa.array(
                np.full(table.num_rows, file_index, dtype=np.int64)
            )
            for alias, original in video_aliases.items():
                for suffix in (
                    "chunk_index",
                    "file_index",
                    "from_timestamp",
                    "to_timestamp",
                ):
                    columns[f"videos/{alias}/{suffix}"] = table[
                        f"videos/{original}/{suffix}"
                    ]
            pq.write_table(
                pa.Table.from_pydict(columns), episode_file, compression="snappy"
            )
        (temporary / "meta/info.json").write_text(
            json.dumps(info, indent=2, sort_keys=True) + "\n"
        )

        source_stats = json.loads((source / "meta/stats.json").read_text())
        stats = {
            name: source_stats[name]
            for name in BOOKKEEPING_FEATURES
            if name in source_stats
        }
        for name, values in batches.items():
            stats[name] = _stats(np.concatenate(values))
        (temporary / "meta/stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n"
        )

        native_schema = dict(expected_schema)
        native_schema["key"] = "action"
        policy.update(
            action_space="joint_absolute",
            action_key="action",
            state_key="observation.state",
            video_keys=list(DROID_VIDEO_KEYS),
            auxiliary_video_keys={},
            dropped_auxiliary_video_keys=dropped_auxiliary_video_keys,
            tasks=tasks,
            framework_loader="DROIDLeRobotDataset:joint_pos",
            framework_action_space="joint_pos",
            framework_joint_count=6,
            framework_use_state=True,
            native_action_schema=native_schema,
            prepared_source_profile=profile,
        )
        (temporary / "meta/cosmos_policy.json").write_text(
            json.dumps(policy, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "output": str(output),
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "action_dim": 7,
        "joint_count": 6,
        "source_profile": profile,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source-profile",
        choices=("auto", "cosmos-policy", "lite6-live"),
        default="auto",
        help="input schema profile; auto uses cosmos_policy metadata when present",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source, args.output, source_profile=args.source_profile),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
