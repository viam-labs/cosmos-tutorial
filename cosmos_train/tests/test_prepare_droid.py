import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_train.policy_contract import build_runtime_contract
from cosmos_train.prepare_droid import DROID_ROOT_NAME, prepare

JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
EMBODIMENT = {
    "arm_model": "lite6",
    "gripper_model": "lite6_gripper",
    "gripper_mode": "binary",
    "joint_names": JOINT_NAMES,
}
ACTION_SCHEMA = {
    "key": "action.joint_absolute",
    "type": "absolute_joint_target",
    "shape": [7],
    "names": [*JOINT_NAMES, "gripper"],
    "units": [*["rad"] * 6, "binary"],
    "fps": 20,
    "gripper_convention": "0=open, 1=closed",
}

CANONICAL_VIDEO_KEYS = (
    "observation.images.fixed",
    "observation.images.wrist",
)
LIVE_VIDEO_KEYS = (
    "observation.images.camera_1",
    "observation.images.realsense_cam",
)
COSMOSWRITER_MODALITIES = (
    "rgb",
    "depth",
    "edges",
    "segmentation",
    "shaded_seg",
)
COSMOSWRITER_VIDEO_KEYS = tuple(
    f"auxiliary.cosmoswriter.{role}.{modality}"
    for role in ("fixed", "wrist")
    for modality in COSMOSWRITER_MODALITIES
)


def _fixed(values: list[list[float]]) -> pa.Array:
    return pa.array(values, type=pa.list_(pa.float32(), len(values[0])))


def _source_dataset(
    root: Path, embodiment: dict = EMBODIMENT
) -> tuple[np.ndarray, np.ndarray]:
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    for video_key in (*CANONICAL_VIDEO_KEYS, *COSMOSWRITER_VIDEO_KEYS):
        video = root / "videos" / video_key / "chunk-000/file-000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video-placeholder")

    joint_state = np.asarray(
        [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06], [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]],
        dtype=np.float32,
    )
    joint_target = joint_state + np.float32(0.5)
    table = pa.Table.from_pydict(
        {
            "source.joint_position_rad": _fixed(joint_state.tolist()),
            "source.joint_target_rad": _fixed(joint_target.tolist()),
            "source.gripper_command": pa.array([0.0, 1.0], type=pa.float32()),
            "source.gripper_state": pa.array([0.0, 1.0], type=pa.float32()),
            "observation.state": _fixed([[0.0] * 7, [0.0] * 7]),
            "action": _fixed([[0.0] * 7, [0.0] * 7]),
            "timestamp": pa.array([0.0, 0.05], type=pa.float32()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
            "episode_index": pa.array([0, 0], type=pa.int64()),
            "index": pa.array([0, 1], type=pa.int64()),
            "task_index": pa.array([0, 1], type=pa.int64()),
        }
    )
    pq.write_table(table, root / "data/chunk-000/file-000.parquet")

    episode_columns: dict[str, pa.Array] = {"episode_index": pa.array([0])}
    for video_key in (*CANONICAL_VIDEO_KEYS, *COSMOSWRITER_VIDEO_KEYS):
        episode_columns.update(
            {
                f"videos/{video_key}/chunk_index": pa.array([0]),
                f"videos/{video_key}/file_index": pa.array([0]),
                f"videos/{video_key}/from_timestamp": pa.array([0.0]),
                f"videos/{video_key}/to_timestamp": pa.array([0.1]),
            }
        )
    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pydict(episode_columns), episode_path)

    vector = lambda width: {"dtype": "float32", "shape": [width]}
    video = {"dtype": "video", "shape": [16, 16, 3], "info": {}}
    info = {
        "fps": 20,
        "total_episodes": 1,
        "total_frames": 2,
        "total_tasks": 2,
        "features": {
            "source.joint_position_rad": vector(6),
            "source.joint_target_rad": vector(6),
            "source.gripper_command": vector(1),
            "source.gripper_state": vector(1),
            "observation.state": vector(7),
            "action": vector(7),
            "timestamp": vector(1),
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            **{
                video_key: video
                for video_key in (*CANONICAL_VIDEO_KEYS, *COSMOSWRITER_VIDEO_KEYS)
            },
        },
    }
    action_schema = {
        **ACTION_SCHEMA,
        "names": [*embodiment["joint_names"], "gripper"],
        "units": [
            *["rad"] * 6,
            "binary" if embodiment["gripper_mode"] == "binary" else "normalized",
        ],
    }
    policy = {
        "format": "lerobot-v3",
        "fps": 20,
        "task": "Pick up the block",
        "tasks": ["Pick up the block", "Center"],
        "action_space": "joint_absolute",
        "embodiment": embodiment,
        "native_action_schema": action_schema,
        "dataset_variant": "direct_dataset_with_cosmoswriter",
        "video_keys": list(CANONICAL_VIDEO_KEYS),
        "auxiliary_video_keys": {
            role: {
                modality: f"auxiliary.cosmoswriter.{role}.{modality}"
                for modality in COSMOSWRITER_MODALITIES
            }
            for role in ("fixed", "wrist")
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text("{}")
    pd.DataFrame(
        {"task_index": [0, 1]},
        index=pd.Index(["Pick up the block", "Center"], name="task"),
    ).to_parquet(root / "meta/tasks.parquet")
    (root / "meta/cosmos_policy.json").write_text(json.dumps(policy))
    return joint_state, joint_target


def _live_source_dataset(root: Path) -> tuple[np.ndarray, np.ndarray]:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    for video_key in LIVE_VIDEO_KEYS:
        video = root / "videos" / video_key / "chunk-000/file-000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video-placeholder")

    joint_state = np.asarray(
        [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06], [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]],
        dtype=np.float32,
    )
    joint_action = joint_state + np.float32(0.01)
    state = np.column_stack((joint_state, [0.0, 1.0])).astype(np.float32)
    action = np.column_stack((joint_action, [1.0, 1.0])).astype(np.float32)
    pq.write_table(
        pa.Table.from_pydict(
            {
                "observation.state": _fixed(state.tolist()),
                "action": _fixed(action.tolist()),
                "timestamp": pa.array([0.0, 0.05], type=pa.float32()),
                "frame_index": pa.array([0, 1], type=pa.int64()),
                "episode_index": pa.array([0, 0], type=pa.int64()),
                "index": pa.array([0, 1], type=pa.int64()),
                "task_index": pa.array([0, 0], type=pa.int64()),
            }
        ),
        root / "data/chunk-000/file-000.parquet",
    )
    episode_columns: dict[str, pa.Array] = {"episode_index": pa.array([0])}
    for video_key in LIVE_VIDEO_KEYS:
        episode_columns.update(
            {
                f"videos/{video_key}/chunk_index": pa.array([0]),
                f"videos/{video_key}/file_index": pa.array([0]),
                f"videos/{video_key}/from_timestamp": pa.array([0.0]),
                f"videos/{video_key}/to_timestamp": pa.array([0.1]),
            }
        )
    pq.write_table(
        pa.Table.from_pydict(episode_columns),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    vector = lambda width, names=None: {
        "dtype": "float32",
        "shape": [width],
        **({"names": names} if names is not None else {}),
    }
    video = {"dtype": "video", "shape": [16, 16, 3], "info": {}}
    bookkeeping = {
        "timestamp": vector(1),
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    names = [
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "gripper",
    ]
    info = {
        "fps": 20,
        "total_episodes": 1,
        "total_frames": 2,
        "total_tasks": 1,
        "features": {
            "observation.state": vector(7, names),
            "action": vector(7, names),
            **bookkeeping,
            **{video_key: video for video_key in LIVE_VIDEO_KEYS},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text("{}")
    pd.DataFrame(
        {"task_index": [0]}, index=pd.Index(["Pick up the block"], name="task")
    ).to_parquet(root / "meta/tasks.parquet")
    return state, action


def test_prepare_live_and_sim_sources_have_identical_training_schema(
    tmp_path: Path,
) -> None:
    sim_source = tmp_path / "sim-source"
    live_source = tmp_path / "live-source"
    _source_dataset(sim_source)
    live_state, live_action = _live_source_dataset(live_source)
    sim_output = tmp_path / "sim" / DROID_ROOT_NAME
    live_output = tmp_path / "live" / DROID_ROOT_NAME
    sim_output.parent.mkdir()
    live_output.parent.mkdir()

    prepare(sim_source, sim_output)
    result = prepare(live_source, live_output, source_profile="lite6-live")

    assert result["source_profile"] == "lite6-live"
    sim_info = json.loads((sim_output / "meta/info.json").read_text())
    live_info = json.loads((live_output / "meta/info.json").read_text())
    assert sim_info["features"] == live_info["features"]
    sim_data = pq.read_table(sim_output / "data/chunk-000/file-000.parquet")
    live_data = pq.read_table(live_output / "data/chunk-000/file-000.parquet")
    assert sim_data.schema == live_data.schema
    assert not any(name.startswith("source.") for name in sim_data.column_names)
    np.testing.assert_allclose(
        np.asarray(live_data["observation.state"].to_pylist()), live_state
    )
    np.testing.assert_allclose(np.asarray(live_data["action"].to_pylist()), live_action)
    sim_episode = pq.read_table(sim_output / "meta/episodes/chunk-000/file-000.parquet")
    live_episode = pq.read_table(
        live_output / "meta/episodes/chunk-000/file-000.parquet"
    )
    assert sim_episode.schema == live_episode.schema
    runtime = build_runtime_contract(live_output)
    assert runtime["embodiment"] == EMBODIMENT
    assert runtime["observation_schema"]["gripper_units"] == "binary"


def test_prepare_preserves_native_joint_targets_and_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    joint_state, joint_target = _source_dataset(source)
    output_parent = tmp_path / "prepared"
    output_parent.mkdir()
    output = output_parent / DROID_ROOT_NAME

    result = prepare(source, output)

    assert result["action_dim"] == 7
    table = pq.read_table(output / "data/chunk-000/file-000.parquet")
    np.testing.assert_allclose(
        np.asarray(table["action.joint_position"].to_pylist()), joint_target
    )
    np.testing.assert_allclose(
        np.asarray(table["observation.state.joint_positions"].to_pylist()), joint_state
    )
    np.testing.assert_allclose(
        np.asarray(table["action"].to_pylist()),
        np.column_stack((joint_target, [0.0, 1.0])),
    )
    np.testing.assert_allclose(
        np.asarray(table["observation.state"].to_pylist()),
        np.column_stack((joint_state, [0.0, 1.0])),
    )
    assert table["task_index"].to_pylist() == [0, 1]
    assert pq.read_table(output / "meta/tasks.parquet").to_pylist() == [
        {"task_index": 0, "task": "Pick up the block"},
        {"task_index": 1, "task": "Center"},
    ]
    info = json.loads((output / "meta/info.json").read_text())
    assert info["total_tasks"] == 2
    episode = pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet")
    video_features = {
        name
        for name, feature in info["features"].items()
        if feature["dtype"] == "video"
    }
    assert video_features == {
        "observation.images.wrist_image_left",
        "observation.images.exterior_image_1_left",
        "observation.images.exterior_image_2_left",
    }
    video_columns = {
        name for name in episode.column_names if name.startswith("videos/")
    }
    assert video_columns == {
        f"videos/{alias}/{suffix}"
        for alias in video_features
        for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    }
    assert {path.name for path in (output / "videos").iterdir()} == video_features
    for video_key in video_features:
        assert (output / "videos" / video_key / "chunk-000/file-000.mp4").is_file()
    for video_key in (*CANONICAL_VIDEO_KEYS, *COSMOSWRITER_VIDEO_KEYS):
        assert not (output / "videos" / video_key).exists()

    policy = json.loads((output / "meta/cosmos_policy.json").read_text())
    assert policy["video_keys"] == [
        "observation.images.wrist_image_left",
        "observation.images.exterior_image_1_left",
        "observation.images.exterior_image_2_left",
    ]
    assert policy["auxiliary_video_keys"] == {}
    assert policy["dropped_auxiliary_video_keys"] == sorted(COSMOSWRITER_VIDEO_KEYS)
    assert policy["tasks"] == ["Pick up the block", "Center"]

    assert policy["framework_action_space"] == "joint_pos"
    assert policy["framework_joint_count"] == 6
    assert policy["framework_use_state"] is True
    assert policy["native_action_schema"]["units"] == ["rad"] * 6 + ["binary"]

    runtime = build_runtime_contract(output)
    assert runtime["schema_version"] == 3
    assert runtime["action_chunk_size"] == 32
    assert runtime["action_schema"]["shape"] == [7]
    assert runtime["embodiment"] == EMBODIMENT
    assert runtime["observation_schema"]["gripper_units"] == "binary"


def test_prepare_maps_second_fixed_camera_to_second_exterior_view(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _source_dataset(source)
    fixed_2_key = "observation.images.fixed_2"
    fixed_2_video = source / "videos" / fixed_2_key / "chunk-000/file-000.mp4"
    fixed_2_video.parent.mkdir(parents=True)
    fixed_2_video.write_bytes(b"second-fixed-camera")

    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"][fixed_2_key] = {
        "dtype": "video",
        "shape": [16, 16, 3],
        "info": {},
    }
    info_path.write_text(json.dumps(info))

    episode_path = source / "meta/episodes/chunk-000/file-000.parquet"
    episode = pq.read_table(episode_path)
    for suffix, value in (
        ("chunk_index", 0),
        ("file_index", 0),
        ("from_timestamp", 0.0),
        ("to_timestamp", 0.1),
    ):
        episode = episode.append_column(
            f"videos/{fixed_2_key}/{suffix}", pa.array([value])
        )
    pq.write_table(episode, episode_path)

    output_parent = tmp_path / "prepared"
    output_parent.mkdir()
    output = output_parent / DROID_ROOT_NAME
    prepare(source, output)

    second_exterior = (
        output
        / "videos/observation.images.exterior_image_2_left/chunk-000/file-000.mp4"
    )
    assert second_exterior.read_bytes() == b"second-fixed-camera"


def test_ur20_runtime_contract_uses_declared_joint_order(tmp_path: Path) -> None:
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    embodiment = {
        "arm_model": "ur20",
        "gripper_model": "robotiq_2f_85",
        "gripper_mode": "position",
        "joint_names": joint_names,
    }
    source = tmp_path / "source"
    _source_dataset(source, embodiment)
    output_parent = tmp_path / "prepared"
    output_parent.mkdir()
    output = output_parent / DROID_ROOT_NAME
    prepare(source, output)

    runtime = build_runtime_contract(output)

    assert runtime["embodiment"] == embodiment
    assert runtime["action_schema"] == {
        "key": "action",
        "type": "absolute_joint_target",
        "shape": [7],
        "names": [*joint_names, "gripper"],
        "units": ["rad"] * 6 + ["normalized"],
        "fps": 20,
        "gripper_convention": "0=open, 1=closed",
    }
    assert runtime["observation_schema"] == {
        "joint_names": joint_names,
        "joint_units": "rad",
        "gripper_units": "normalized",
        "gripper_convention": "0=open, 1=closed",
        "camera_keys": [
            "observation.images.wrist_image_left",
            "observation.images.exterior_image_1_left",
            "observation.images.exterior_image_2_left",
        ],
        "camera_aliases": {
            "observation.images.wrist_image_left": "observation/wrist_image_left",
            "observation.images.exterior_image_1_left": "observation/exterior_image_1_left",
            "observation.images.exterior_image_2_left": "observation/exterior_image_2_left",
        },
    }
