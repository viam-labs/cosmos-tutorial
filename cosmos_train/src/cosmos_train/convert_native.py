"""Convert schema-v3 native episodes to a LeRobot v3 Cosmos Policy dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = 3
RUN_SCHEMA_VERSIONS = (4,)
VIDEO_KEYS = {
    "fixed": "observation.images.fixed",
    "wrist": "observation.images.wrist",
    "fixed_2": "observation.images.fixed_2",
}
COSMOSWRITER_ROLES = ("fixed", "wrist", "fixed_2")
COSMOSWRITER_MODALITIES = (
    "rgb",
    "depth",
    "edges",
    "segmentation",
    "shaded_seg",
)
AUXILIARY_VIDEO_KEYS = {
    role: {
        modality: f"auxiliary.cosmoswriter.{role}.{modality}"
        for modality in COSMOSWRITER_MODALITIES
    }
    for role in COSMOSWRITER_ROLES
}
ACTION_DIMS = {"joint_absolute": 7, "ee_relative": 10}


@dataclass(frozen=True)
class NativeEpisode:
    path: Path
    manifest: dict[str, Any]
    samples: dict[str, np.ndarray]


@dataclass(frozen=True)
class DerivedEpisode:
    parent: NativeEpisode
    episode_id: str
    task: str
    start_frame: int
    end_frame_exclusive: int

    @property
    def sample_count(self) -> int:
        return self.end_frame_exclusive - self.start_frame

    def sliced_native(self) -> NativeEpisode:
        manifest = {**self.parent.manifest, "sample_count": self.sample_count}
        samples = {
            name: values[self.start_frame : self.end_frame_exclusive]
            for name, values in self.parent.samples.items()
        }
        return NativeEpisode(self.parent.path, manifest, samples)


@dataclass(frozen=True)
class StageVideoLayout:
    info: dict[str, Any]
    frame_pts: tuple[int, ...]
    boundary_packet_indices: dict[int, int]
    packet_count: int


def _split_episode_selection(
    episodes: Sequence[NativeEpisode],
    split_episodes: bool,
    split_episode_count: int | None,
    split_episode_seed: int,
) -> tuple[frozenset[str], dict[str, Any]]:
    if isinstance(split_episode_seed, bool) or not isinstance(split_episode_seed, int):
        raise TypeError("split episode seed must be an integer")
    if split_episode_count is not None and (
        isinstance(split_episode_count, bool)
        or not isinstance(split_episode_count, int)
        or split_episode_count <= 0
    ):
        raise ValueError("split episode count must be a positive integer")
    if split_episode_count is not None and not split_episodes:
        raise ValueError("split episode count requires split_episodes")

    episode_ids = tuple(str(episode.manifest["episode_id"]) for episode in episodes)
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("selected native episodes must use unique episode IDs")
    if not split_episodes:
        selected_ids: tuple[str, ...] = ()
        mode = "disabled"
    elif split_episode_count is None:
        selected_ids = episode_ids
        mode = "all"
    else:
        if split_episode_count > len(episodes):
            raise ValueError(
                f"split episode count {split_episode_count} exceeds "
                f"{len(episodes)} eligible episodes"
            )
        selected_indices = sorted(
            random.Random(split_episode_seed).sample(
                range(len(episodes)), split_episode_count
            )
        )
        selected_ids = tuple(episode_ids[index] for index in selected_indices)
        mode = "sample"

    selection: dict[str, Any] = {
        "mode": mode,
        "eligible_count": len(episodes),
        "sample_count": len(selected_ids),
        "episode_ids": list(selected_ids),
    }
    if mode == "sample":
        selection["seed"] = split_episode_seed
    return frozenset(selected_ids), selection


def _derived_episodes(
    episodes: Sequence[NativeEpisode],
    overall_task_override: str | None,
    split_episode_ids: frozenset[str],
) -> list[DerivedEpisode]:
    derived: list[DerivedEpisode] = []
    for episode in episodes:
        episode_id = str(episode.manifest["episode_id"])
        overall_task = overall_task_override or str(episode.manifest["task"])
        derived.append(
            DerivedEpisode(
                episode, episode_id, overall_task, 0, episode.manifest["sample_count"]
            )
        )
        if episode_id in split_episode_ids:
            for stage_index, stage in enumerate(episode.manifest["stages"]):
                derived.append(
                    DerivedEpisode(
                        episode,
                        f"{episode_id}:stage-{stage_index}",
                        str(stage["task"]),
                        int(stage["start_frame"]),
                        int(stage["end_frame_exclusive"]),
                    )
                )
    return derived


def _robot_descriptor(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    robot = manifest.get("robot")
    if not isinstance(robot, dict):
        raise TypeError(f"{path}: robot descriptor is required")
    arm_model = robot.get("arm_model")
    gripper_model = robot.get("gripper_model")
    gripper_mode = robot.get("gripper_mode")
    joint_names = robot.get("joint_names")
    if not isinstance(arm_model, str) or not arm_model:
        raise ValueError(f"{path}: robot.arm_model must be a non-empty string")
    if not isinstance(gripper_model, str) or not gripper_model:
        raise ValueError(f"{path}: robot.gripper_model must be a non-empty string")
    if gripper_mode not in ("binary", "position"):
        raise ValueError(f"{path}: robot.gripper_mode must be binary or position")
    if (
        not isinstance(joint_names, list)
        or len(joint_names) != 6
        or any(not isinstance(name, str) or not name for name in joint_names)
        or len(set(joint_names)) != len(joint_names)
    ):
        raise ValueError(f"{path}: robot.joint_names must contain six unique names")
    return {
        "arm_model": arm_model,
        "gripper_model": gripper_model,
        "gripper_mode": gripper_mode,
        "joint_names": list(joint_names),
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: expected schema_version {SCHEMA_VERSION}, "
            f"got {manifest.get('schema_version')!r}"
        )
    sample_rate_hz = manifest.get("sample_rate_hz")
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, (int, float))
        or not math.isfinite(float(sample_rate_hz))
        or float(sample_rate_hz) <= 0.0
    ):
        raise ValueError(f"{path}: sample_rate_hz must be finite and positive")
    manifest["sample_rate_hz"] = float(sample_rate_hz)
    task = manifest.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"{path}: task must be a non-empty string")
    sample_count = manifest.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise ValueError(f"{path}: sample_count must be a positive integer")
    stages = manifest.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"{path}: stages must be a non-empty list")
    expected_start = 0
    stage_tasks: set[str] = set()
    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise TypeError(f"{path}: stage {stage_index} must be an object")
        start = stage.get("start_frame")
        end = stage.get("end_frame_exclusive")
        stage_task = stage.get("task")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start != expected_start
            or end <= start
        ):
            raise ValueError(f"{path}: stage {stage_index} has an invalid frame range")
        if not isinstance(stage_task, str) or not stage_task.strip():
            raise ValueError(f"{path}: stage {stage_index} task must be non-empty")
        if stage_task in stage_tasks or stage_task == task:
            raise ValueError(f"{path}: overall and stage tasks must be unique")
        stage_tasks.add(stage_task)
        expected_start = end
    if expected_start != sample_count:
        raise ValueError(f"{path}: final stage must end at sample_count")
    _robot_descriptor(path, manifest)
    return manifest


def _as_matrix(table: pa.Table, name: str, width: int) -> np.ndarray:
    values = np.asarray(table[name].to_pylist(), dtype=np.float32)
    if values.shape != (table.num_rows, width) or not np.isfinite(values).all():
        raise ValueError(
            f"{name} must have shape ({table.num_rows}, {width}) and finite values"
        )
    return values


def _as_column(
    table: pa.Table, name: str, dtype: type[np.floating[Any]] = np.float32
) -> np.ndarray:
    values = np.asarray(table[name].to_pylist(), dtype=dtype)
    if values.shape != (table.num_rows,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must have {table.num_rows} finite scalar values")
    return values


def _load_episode(path: Path) -> NativeEpisode:
    manifest = _read_manifest(path / "manifest.json")
    table = pq.read_table(path / manifest["samples_file"])
    if table.num_rows != manifest["sample_count"] or table.num_rows == 0:
        raise ValueError(f"{path}: sample count does not match manifest")
    frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(frame_index, np.arange(table.num_rows)):
        raise ValueError(f"{path}: frame_index must be contiguous from zero")
    sim_time = _as_column(table, "sim_time_s", np.float64)
    relative_time = sim_time - sim_time[0]
    sample_rate_hz = float(manifest["sample_rate_hz"])
    expected_time = np.arange(table.num_rows, dtype=np.float64) / sample_rate_hz
    if not np.allclose(relative_time, expected_time, atol=1e-4, rtol=0.0):
        raise ValueError(f"{path}: samples are not aligned at {sample_rate_hz:g} Hz")

    samples = {
        "joint_position": _as_matrix(table, "joint_position_rad", 6),
        "joint_velocity": _as_matrix(table, "joint_velocity_rad_s", 6),
        "joint_target": _as_matrix(table, "joint_target_rad", 6),
        "ee_position": _as_matrix(table, "ee_position_m", 3),
        "ee_rotation": _as_matrix(table, "ee_rotation_wxyz", 4),
        "gripper_command": _as_column(table, "gripper_command"),
        "gripper_state": _as_column(table, "gripper_state"),
    }
    if manifest["robot"]["gripper_mode"] == "binary":
        for name in ("gripper_command", "gripper_state"):
            if not np.isin(samples[name], (0.0, 1.0)).all():
                raise ValueError(f"{path}: binary {name} must contain only 0 or 1")
    return NativeEpisode(path, manifest, samples)


def _consistent_embodiment(episodes: Sequence[NativeEpisode]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot resolve embodiment from no episodes")
    embodiment = _robot_descriptor(episodes[0].path, episodes[0].manifest)
    for episode in episodes[1:]:
        candidate = _robot_descriptor(episode.path, episode.manifest)
        if candidate != embodiment:
            raise ValueError(
                f"{episode.path}: mixed embodiment; expected {embodiment}, got {candidate}"
            )
    return embodiment


def _rotation_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _relative_ee_actions(episode: NativeEpisode) -> np.ndarray:
    positions = episode.samples["ee_position"]
    rotations = episode.samples["ee_rotation"]
    count = len(positions)
    actions = np.empty((count, 10), dtype=np.float32)
    actions[:, :3] = 0.0
    actions[:, 3:9] = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    for index in range(count - 1):
        actions[index, :3] = positions[index + 1] - positions[index]
        relative = _rotation_matrix(rotations[index]).T @ _rotation_matrix(
            rotations[index + 1]
        )
        actions[index, 3:9] = relative[:, :2].T.reshape(6)
    actions[:, 9] = episode.samples["gripper_command"]
    return actions


def _joint_actions(episode: NativeEpisode) -> np.ndarray:
    return np.column_stack(
        (episode.samples["joint_target"], episode.samples["gripper_command"])
    ).astype(np.float32)


def _video_info(
    path: Path,
    expected_frames: int,
    *,
    expected_fps: float,
    expected_codec: str | None = None,
    expected_pixel_format: str | None = None,
) -> dict[str, Any]:
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError(f"{path}: expected one video stream, found {len(streams)}")
        stream = streams[0]
        average_rate = stream.average_rate
        if average_rate is None:
            raise ValueError(f"{path}: video frame rate is unavailable")
        actual_fps = float(average_rate)
        if not math.isclose(actual_fps, expected_fps, abs_tol=1e-6, rel_tol=0.0):
            raise ValueError(
                f"{path}: video frame rate is {actual_fps}, expected {expected_fps}"
            )

        frame_count = 0
        geometry: tuple[int, int] | None = None
        pixel_format: str | None = None
        for frame in container.decode(stream):
            current_geometry = (int(frame.width), int(frame.height))
            current_pixel_format = frame.format.name
            if geometry is None:
                geometry = current_geometry
                pixel_format = current_pixel_format
            elif current_geometry != geometry:
                raise ValueError(
                    f"{path}: decoded frame geometry changed from "
                    f"{geometry[0]}x{geometry[1]} to {frame.width}x{frame.height}"
                )
            elif current_pixel_format != pixel_format:
                raise ValueError(
                    f"{path}: decoded pixel format changed from {pixel_format!r} "
                    f"to {current_pixel_format!r}"
                )
            frame_count += 1

        if frame_count != expected_frames:
            raise ValueError(
                f"{path}: video has {frame_count} frames, expected {expected_frames}"
            )
        if geometry is None or pixel_format is None:
            raise ValueError(f"{path}: video has no decodable frames")
        codec = stream.codec_context.name
        if expected_codec is not None and codec != expected_codec:
            raise ValueError(
                f"{path}: video codec is {codec!r}, expected {expected_codec!r}"
            )
        if expected_pixel_format is not None and pixel_format != expected_pixel_format:
            raise ValueError(
                f"{path}: video pixel format is {pixel_format!r}, "
                f"expected {expected_pixel_format!r}"
            )
        return {
            "video.height": geometry[1],
            "video.width": geometry[0],
            "video.codec": codec,
            "video.pix_fmt": pixel_format,
            "video.is_depth_map": False,
            "video.fps": actual_fps,
            "video.channels": 3,
            "has_audio": bool(container.streams.audio),
        }


def _clip_video(
    source: Path,
    destination: Path,
    source_frames: int,
    start_frame: int,
    end_frame_exclusive: int,
    sample_rate_hz: float,
) -> dict[str, Any]:
    if not 0 <= start_frame < end_frame_exclusive <= source_frames:
        raise ValueError(f"{source}: invalid video frame range")
    expected_frames = end_frame_exclusive - start_frame
    destination.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source), mode="r") as input_container:
        input_streams = list(input_container.streams.video)
        if len(input_streams) != 1:
            raise ValueError(
                f"{source}: expected one video stream, found {len(input_streams)}"
            )
        input_stream = input_streams[0]
        expected_rate = Fraction(str(sample_rate_hz))
        actual_rate = input_stream.average_rate
        if actual_rate is None or Fraction(actual_rate) != expected_rate:
            raise ValueError(
                f"{source}: video frame rate is {actual_rate}, expected {expected_rate}"
            )
        with av.open(str(destination), mode="w") as output_container:
            output_stream: Any | None = None
            geometry: tuple[int, int] | None = None
            source_count = 0
            output_count = 0
            for source_index, frame in enumerate(input_container.decode(input_stream)):
                source_count += 1
                current_geometry = (int(frame.width), int(frame.height))
                if geometry is None:
                    geometry = current_geometry
                    rate = Fraction(str(sample_rate_hz))
                    output_stream = output_container.add_stream("libx264", rate=rate)
                    output_stream.width = geometry[0]
                    output_stream.height = geometry[1]
                    output_stream.pix_fmt = "yuv420p"
                elif current_geometry != geometry:
                    raise ValueError(
                        f"{source}: decoded frame geometry changed from "
                        f"{geometry[0]}x{geometry[1]} to {frame.width}x{frame.height}"
                    )
                if not start_frame <= source_index < end_frame_exclusive:
                    continue
                normalized = frame.reformat(
                    width=geometry[0], height=geometry[1], format="yuv420p"
                )
                normalized.pts = output_count
                normalized.time_base = Fraction(1, 1) / Fraction(str(sample_rate_hz))
                assert output_stream is not None
                for packet in output_stream.encode(normalized):
                    output_container.mux(packet)
                output_count += 1

            if source_count != source_frames:
                raise ValueError(
                    f"{source}: video has {source_count} frames, expected {source_frames}"
                )
            if output_count != expected_frames or output_stream is None:
                raise ValueError(
                    f"{source}: clipped {output_count} frames, expected {expected_frames}"
                )
            for packet in output_stream.encode():
                output_container.mux(packet)

    return _video_info(
        destination,
        expected_frames,
        expected_fps=sample_rate_hz,
        expected_codec="h264",
        expected_pixel_format="yuv420p",
    )


def _packet_nal_types(packet: av.Packet, nal_length_size: int) -> frozenset[int]:
    payload = bytes(packet)
    offset = 0
    nal_types: set[int] = set()
    while offset < len(payload):
        length_end = offset + nal_length_size
        if length_end > len(payload):
            raise ValueError("truncated H.264 NAL length")
        nal_size = int.from_bytes(payload[offset:length_end], "big")
        offset = length_end
        nal_end = offset + nal_size
        if nal_size <= 0 or nal_end > len(payload):
            raise ValueError("invalid H.264 NAL length")
        nal_types.add(payload[offset] & 0x1F)
        offset = nal_end
    return frozenset(nal_types)


def _stage_video_layout(
    source: Path,
    source_frames: int,
    stage_start_frames: tuple[int, ...],
    sample_rate_hz: float,
) -> StageVideoLayout:
    info = _video_info(source, source_frames, expected_fps=sample_rate_hz)
    if info["video.codec"] != "h264" or info["video.pix_fmt"] != "yuv420p":
        raise ValueError(f"{source}: stage splitting requires H.264/yuv420p video")
    if info["has_audio"]:
        raise ValueError(f"{source}: stage splitting requires a video-only MP4")

    with av.open(str(source), mode="r") as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        frame_pts = tuple(frame.pts for frame in frames)
        if any(pts is None for pts in frame_pts) or len(set(frame_pts)) != len(
            frame_pts
        ):
            raise ValueError(
                f"{source}: video frames require unique presentation times"
            )
        typed_frame_pts = tuple(int(pts) for pts in frame_pts if pts is not None)
        frame_times = tuple(
            float(frame.pts * frame.time_base)
            for frame in frames
            if frame.pts is not None
        )
        expected_times = np.arange(source_frames, dtype=np.float64) / sample_rate_hz
        if not np.allclose(frame_times, expected_times, atol=1e-6, rtol=0.0):
            raise ValueError(f"{source}: video timestamps are not frame-aligned")
        extradata = stream.codec_context.extradata
        if not extradata or len(extradata) < 5 or extradata[0] != 1:
            raise ValueError(f"{source}: expected AVC decoder configuration")
        nal_length_size = (extradata[4] & 0x03) + 1

    boundary_pts = {
        frame_index: typed_frame_pts[frame_index] for frame_index in stage_start_frames
    }
    boundary_packet_indices: dict[int, int] = {}
    packet_count = 0
    with av.open(str(source), mode="r") as container:
        stream = container.streams.video[0]
        pts_to_frame = {pts: frame_index for frame_index, pts in boundary_pts.items()}
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            frame_index = pts_to_frame.get(packet.pts)
            if frame_index is not None:
                if not packet.is_keyframe or 5 not in _packet_nal_types(
                    packet, nal_length_size
                ):
                    raise ValueError(
                        f"{source}: stage frame {frame_index} is not a closed-GOP IDR"
                    )
                if frame_index in boundary_packet_indices:
                    raise ValueError(
                        f"{source}: multiple packets map to stage frame {frame_index}"
                    )
                boundary_packet_indices[frame_index] = packet_count
            packet_count += 1

    missing = sorted(set(stage_start_frames) - set(boundary_packet_indices))
    if missing:
        raise ValueError(f"{source}: missing stage IDRs at frames {missing}")
    return StageVideoLayout(
        info, typed_frame_pts, boundary_packet_indices, packet_count
    )


def _remux_stage_video(
    source: Path,
    destination: Path,
    layout: StageVideoLayout,
    start_frame: int,
    end_frame_exclusive: int,
    sample_rate_hz: float,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    start_packet = layout.boundary_packet_indices[start_frame]
    end_packet = layout.boundary_packet_indices.get(
        end_frame_exclusive, layout.packet_count
    )
    start_pts = layout.frame_pts[start_frame]
    with (
        av.open(str(source), mode="r") as input_container,
        av.open(str(destination), mode="w") as output_container,
    ):
        input_stream = input_container.streams.video[0]
        output_stream = output_container.add_stream_from_template(input_stream)
        packet_index = 0
        for packet in input_container.demux(input_stream):
            if packet.pts is None:
                continue
            if start_packet <= packet_index < end_packet:
                if packet.dts is None:
                    raise ValueError(f"{source}: video packet has no decode timestamp")
                packet.pts -= start_pts
                packet.dts -= start_pts
                packet.stream = output_stream
                output_container.mux(packet)
            packet_index += 1

    return _video_info(
        destination,
        end_frame_exclusive - start_frame,
        expected_fps=sample_rate_hz,
        expected_codec="h264",
        expected_pixel_format=layout.info["video.pix_fmt"],
    )


def _feature(dtype: str, width: int) -> dict[str, Any]:
    return {"dtype": dtype, "shape": [width]}


def _features(
    video_info: dict[str, dict[str, Any]], action_space: str
) -> dict[str, Any]:
    features: dict[str, Any] = {
        video_key: {
            "dtype": "video",
            "shape": [info["video.height"], info["video.width"], 3],
            "info": info,
        }
        for video_key, info in video_info.items()
    }
    features.update(
        {
            "observation.state": _feature("float32", 7),
            "action": _feature("float32", ACTION_DIMS[action_space]),
            "action.joint_absolute": _feature("float32", 7),
            "action.ee_relative": _feature("float32", 10),
            "source.joint_position_rad": _feature("float32", 6),
            "source.joint_velocity_rad_s": _feature("float32", 6),
            "source.joint_target_rad": _feature("float32", 6),
            "source.ee_position_m": _feature("float32", 3),
            "source.ee_rotation_wxyz": _feature("float32", 4),
            "source.gripper_command": _feature("float32", 1),
            "source.gripper_state": _feature("float32", 1),
            "timestamp": _feature("float32", 1),
            "frame_index": _feature("int64", 1),
            "episode_index": _feature("int64", 1),
            "index": _feature("int64", 1),
            "task_index": _feature("int64", 1),
        }
    )
    return features


def _fixed_list(values: np.ndarray) -> pa.Array:
    width = values.shape[1]
    return pa.array(values.tolist(), type=pa.list_(pa.float32(), width))


def _vector_array(values: np.ndarray) -> pa.Array:
    if values.shape[1] == 1:
        return pa.array(values[:, 0], type=pa.float32())
    return _fixed_list(values)


def _frame_table(
    episode: NativeEpisode,
    episode_index: int,
    global_start: int,
    task_index: int,
    action_space: str,
) -> tuple[pa.Table, dict[str, np.ndarray]]:
    count = episode.manifest["sample_count"]
    joint_action = _joint_actions(episode)
    ee_action = _relative_ee_actions(episode)
    observation_state = np.column_stack(
        (episode.samples["joint_position"], episode.samples["gripper_state"])
    ).astype(np.float32)
    vectors = {
        "observation.state": observation_state,
        "action": joint_action if action_space == "joint_absolute" else ee_action,
        "action.joint_absolute": joint_action,
        "action.ee_relative": ee_action,
        "source.joint_position_rad": episode.samples["joint_position"],
        "source.joint_velocity_rad_s": episode.samples["joint_velocity"],
        "source.joint_target_rad": episode.samples["joint_target"],
        "source.ee_position_m": episode.samples["ee_position"],
        "source.ee_rotation_wxyz": episode.samples["ee_rotation"],
        "source.gripper_command": episode.samples["gripper_command"][:, None],
        "source.gripper_state": episode.samples["gripper_state"][:, None],
    }
    columns: dict[str, pa.Array] = {
        name: _vector_array(values) for name, values in vectors.items()
    }
    columns.update(
        {
            "timestamp": pa.array(
                np.arange(count, dtype=np.float32)
                / float(episode.manifest["sample_rate_hz"])
            ),
            "frame_index": pa.array(np.arange(count, dtype=np.int64)),
            "episode_index": pa.array(np.full(count, episode_index, dtype=np.int64)),
            "index": pa.array(
                np.arange(global_start, global_start + count, dtype=np.int64)
            ),
            "task_index": pa.array(np.full(count, task_index, dtype=np.int64)),
        }
    )
    return pa.Table.from_pydict(columns), vectors


def _stats(arrays: Iterable[np.ndarray]) -> dict[str, Any]:
    values = np.concatenate(list(arrays), axis=0).astype(np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _native_episode_paths(source: Path) -> tuple[list[Path], str, int | None]:
    run_path = source / "run.json"
    task = "Pick up the block"
    episodes_dir = source
    run_schema_version: int | None = None
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run_schema_version = run.get("schema_version")
        if run_schema_version not in RUN_SCHEMA_VERSIONS:
            raise ValueError(
                f"{run_path}: expected run schema_version to be one of "
                f"{sorted(RUN_SCHEMA_VERSIONS)}"
            )
        config = run.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"{run_path}: config must be an object")
        if run_schema_version == 4 and not isinstance(
            config.get("cosmos_output"), bool
        ):
            raise ValueError(f"{run_path}: config.cosmos_output must be a boolean")
        run_task = config.get("task")
        if run_task is not None:
            if not isinstance(run_task, str) or not run_task.strip():
                raise ValueError(f"{run_path}: config.task must be a non-empty string")
            task = run_task
        episodes_dir = source / "episodes"
        if not episodes_dir.is_dir():
            raise ValueError(f"{run_path}: episodes directory is missing")
    paths = sorted(
        path
        for path in episodes_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    return paths, task, run_schema_version


def _cosmoswriter_video_sources(
    source: Path, episodes: Sequence[NativeEpisode], camera_roles: Sequence[str]
) -> dict[str, dict[str, Path]]:
    run_path = source / "run.json"
    if not run_path.is_file():
        raise ValueError(
            "include_cosmoswriter requires a run directory containing run.json"
        )
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run_schema_version = run.get("schema_version")
    if run_schema_version != 4:
        raise ValueError(f"{run_path}: expected run schema_version 4")
    config = run.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{run_path}: config must be an object")
    cosmos_output = config.get("cosmos_output")
    if not isinstance(cosmos_output, bool):
        raise ValueError(f"{run_path}: config.cosmos_output must be a boolean")
    if not cosmos_output:
        raise ValueError(
            f"{run_path}: include_cosmoswriter requires config.cosmos_output=true"
        )
    attempts = run.get("attempts")
    if not isinstance(attempts, list):
        raise TypeError(f"{run_path}: attempts must be a list")

    attempts_by_id: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise TypeError(f"{run_path}: each attempt must be an object")
        episode_id = attempt.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"{run_path}: every attempt requires an episode_id")
        if episode_id in attempts_by_id:
            raise ValueError(
                f"{run_path}: duplicate attempt for episode_id {episode_id}"
            )
        attempts_by_id[episode_id] = attempt

    result: dict[str, dict[str, Path]] = {}
    for episode in episodes:
        episode_id = episode.manifest["episode_id"]
        attempt = attempts_by_id.get(episode_id)
        if attempt is None:
            raise ValueError(f"{run_path}: no attempt for episode_id {episode_id}")
        if attempt.get("status") != "finalized":
            raise ValueError(
                f"{run_path}: attempt {episode_id} is not finalized "
                f"(status={attempt.get('status')!r})"
            )
        for field in ("sample_count", "seed", "outcome"):
            if field in attempt and attempt[field] != episode.manifest[field]:
                raise ValueError(
                    f"{run_path}: attempt {episode_id} {field} does not match "
                    "the native manifest"
                )

        cosmoswriter = attempt.get("cosmoswriter")
        if not isinstance(cosmoswriter, dict):
            raise TypeError(
                f"{run_path}: finalized attempt {episode_id} has no CosmosWriter metadata"
            )
        cosmos_path = source / "cosmoswriter" / episode_id
        roles = cosmoswriter.get("roles")
        if not isinstance(roles, dict):
            raise TypeError(
                f"{run_path}: attempt {episode_id} CosmosWriter roles are missing"
            )

        episode_sources: dict[str, Path] = {}
        for role in camera_roles:
            modalities = roles.get(role)
            if not isinstance(modalities, dict):
                raise TypeError(
                    f"{run_path}: attempt {episode_id} is missing CosmosWriter role {role}"
                )
            for modality in COSMOSWRITER_MODALITIES:
                if modality not in modalities:
                    raise ValueError(
                        f"{run_path}: attempt {episode_id} is missing CosmosWriter "
                        f"stream {role}/{modality}"
                    )
                video_key = AUXILIARY_VIDEO_KEYS[role][modality]
                video_path = cosmos_path / role / "clip_0000" / f"{modality}.mp4"
                if not video_path.is_file() or video_path.stat().st_size == 0:
                    raise ValueError(
                        f"{run_path}: attempt {episode_id} has missing or empty "
                        f"CosmosWriter video {video_path}"
                    )
                episode_sources[video_key] = video_path
        result[episode_id] = episode_sources
    return result


def convert(
    source: Path,
    output: Path,
    *,
    action_space: str = "joint_absolute",
    outcomes: frozenset[str] = frozenset({"success"}),
    task: str | None = None,
    include_cosmoswriter: bool = False,
    split_episodes: bool = False,
    split_episode_count: int | None = None,
    split_episode_seed: int = 0,
) -> dict[str, Any]:
    if action_space not in ACTION_DIMS:
        raise ValueError(f"unsupported action space {action_space!r}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    episode_paths, _, run_schema_version = _native_episode_paths(source)
    episodes = [
        episode
        for path in episode_paths
        if (episode := _load_episode(path)).manifest["outcome"] in outcomes
    ]
    if not episodes:
        raise ValueError(f"no finalized episodes matched outcomes {sorted(outcomes)}")
    sample_rates = {float(episode.manifest["sample_rate_hz"]) for episode in episodes}
    if len(sample_rates) != 1:
        raise ValueError(
            f"selected native episodes have mixed sample rates: {sorted(sample_rates)}"
        )
    sample_rate_hz = next(iter(sample_rates))
    if task is not None and not task.strip():
        raise ValueError("task override must not be empty")
    split_episode_ids, split_episode_selection = _split_episode_selection(
        episodes, split_episodes, split_episode_count, split_episode_seed
    )
    derived_episodes = _derived_episodes(episodes, task, split_episode_ids)
    task_names = list(dict.fromkeys(item.task for item in derived_episodes))
    task_indices = {name: index for index, name in enumerate(task_names)}
    stage_task_sequences = {
        tuple(str(stage["task"]) for stage in episode.manifest["stages"])
        for episode in episodes
    }
    if len(stage_task_sequences) != 1:
        raise ValueError("selected native episodes must use consistent stage tasks")
    expected_task_count = (
        1 + len(next(iter(stage_task_sequences))) if split_episodes else 1
    )
    if len(task_names) != expected_task_count:
        raise ValueError(
            f"selected episodes must resolve to {expected_task_count} consistent tasks"
        )
    overall_tasks = {task or str(episode.manifest["task"]) for episode in episodes}
    if len(overall_tasks) != 1:
        raise ValueError("selected native episodes must use one overall task")
    overall_task = next(iter(overall_tasks))
    embodiment = _consistent_embodiment(episodes)
    role_sets = [
        {camera["role"] for camera in episode.manifest["cameras"]}
        for episode in episodes
    ]
    if any(roles != role_sets[0] for roles in role_sets[1:]):
        raise ValueError("selected native episodes must use the same camera roles")
    if not {"fixed", "wrist"} <= role_sets[0] or role_sets[0] - set(VIDEO_KEYS):
        raise ValueError(f"unsupported native camera roles: {sorted(role_sets[0])}")
    camera_roles = tuple(role for role in COSMOSWRITER_ROLES if role in role_sets[0])
    video_keys = {role: VIDEO_KEYS[role] for role in camera_roles}
    canonical_video_keys = frozenset(video_keys.values())

    temporary = output.with_name(f".{output.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"partial conversion already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        first_manifest = episodes[0].manifest
        cosmoswriter_sources = (
            _cosmoswriter_video_sources(source, episodes, camera_roles)
            if include_cosmoswriter
            else {}
        )
        video_info: dict[str, dict[str, Any]] = {}
        stage_video_layouts: dict[Path, StageVideoLayout] = {}
        vector_batches: dict[str, list[np.ndarray]] = {}
        metadata_rows: list[dict[str, Any]] = []
        global_index = 0

        for episode_index, derived in enumerate(derived_episodes):
            parent = derived.parent
            count = derived.sample_count
            parent_count = int(parent.manifest["sample_count"])
            role_files = {
                camera["role"]: camera["file"] for camera in parent.manifest["cameras"]
            }
            all_sources: dict[str, Path] = {}
            for role, video_key in video_keys.items():
                if role not in role_files:
                    raise ValueError(
                        f"{parent.path}: manifest is missing camera role {role}"
                    )
                all_sources[video_key] = parent.path / role_files[role]
            if include_cosmoswriter:
                all_sources.update(cosmoswriter_sources[parent.manifest["episode_id"]])

            table, vectors = _frame_table(
                derived.sliced_native(),
                episode_index,
                global_index,
                task_indices[derived.task],
                action_space,
            )
            chunk_index, file_index = divmod(episode_index, 1000)
            data_path = (
                temporary
                / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
            )
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, data_path, compression="snappy")

            metadata: dict[str, Any] = {
                "episode_index": episode_index,
                "episode_id": derived.episode_id,
                "parent_episode_id": parent.manifest["episode_id"],
                "native_start_frame": derived.start_frame,
                "native_end_frame_exclusive": derived.end_frame_exclusive,
                "seed": parent.manifest["seed"],
                "outcome": parent.manifest["outcome"],
                "tasks": [derived.task],
                "length": count,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + count,
            }
            for video_key, source_video in all_sources.items():
                video_path = (
                    temporary
                    / f"videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
                )
                layout: StageVideoLayout | None = None
                if parent.manifest["episode_id"] in split_episode_ids:
                    layout = stage_video_layouts.get(source_video)
                    if layout is None:
                        stage_start_frames = tuple(
                            int(stage["start_frame"])
                            for stage in parent.manifest["stages"]
                        )
                        layout = _stage_video_layout(
                            source_video,
                            parent_count,
                            stage_start_frames,
                            sample_rate_hz,
                        )
                        stage_video_layouts[source_video] = layout
                whole_episode = (
                    derived.start_frame == 0
                    and derived.end_frame_exclusive == parent_count
                )
                if whole_episode and (
                    split_episodes or video_key in canonical_video_keys
                ):
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_video, video_path)
                    current_info = (
                        layout.info
                        if layout is not None
                        else _video_info(video_path, count, expected_fps=sample_rate_hz)
                    )
                elif layout is not None:
                    current_info = _remux_stage_video(
                        source_video,
                        video_path,
                        layout,
                        derived.start_frame,
                        derived.end_frame_exclusive,
                        sample_rate_hz,
                    )
                else:
                    current_info = _clip_video(
                        source_video,
                        video_path,
                        parent_count,
                        derived.start_frame,
                        derived.end_frame_exclusive,
                        sample_rate_hz,
                    )
                expected_info = video_info.setdefault(video_key, current_info)
                if current_info != expected_info:
                    raise ValueError(
                        f"{source_video}: video format differs from first episode "
                        f"for {video_key}"
                    )
                metadata.update(
                    {
                        f"videos/{video_key}/chunk_index": chunk_index,
                        f"videos/{video_key}/file_index": file_index,
                        f"videos/{video_key}/from_timestamp": 0.0,
                        f"videos/{video_key}/to_timestamp": count / sample_rate_hz,
                    }
                )
            metadata_rows.append(metadata)
            for name, values in vectors.items():
                vector_batches.setdefault(name, []).append(values)
            global_index += count

        metadata_root = temporary / "meta"
        metadata_root.mkdir()
        for chunk_index in range(math.ceil(len(metadata_rows) / 1000)):
            rows = metadata_rows[chunk_index * 1000 : (chunk_index + 1) * 1000]
            path = metadata_root / f"episodes/chunk-{chunk_index:03d}/file-000.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy")
        pd.DataFrame(
            {"task_index": range(len(task_names))},
            index=pd.Index(task_names, name="task"),
        ).to_parquet(metadata_root / "tasks.parquet")

        features = _features(video_info, action_space)
        info = {
            "codebase_version": "v3.0",
            "robot_type": first_manifest["robot"]["arm_model"],
            "total_episodes": len(derived_episodes),
            "total_frames": global_index,
            "total_tasks": len(task_names),
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": sample_rate_hz,
            "splits": {"train": f"0:{len(derived_episodes)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
        }
        (metadata_root / "info.json").write_text(
            json.dumps(info, indent=2, sort_keys=True) + "\n"
        )
        statistics = {name: _stats(values) for name, values in vector_batches.items()}
        (metadata_root / "stats.json").write_text(
            json.dumps(statistics, indent=2, sort_keys=True) + "\n"
        )
        dataset_variant = (
            "direct_dataset_with_cosmoswriter"
            if include_cosmoswriter
            else "direct_dataset"
        )
        adapter = {
            "format": "lerobot-v3",
            "dataset_variant": dataset_variant,
            "fps": sample_rate_hz,
            "task": overall_task,
            "tasks": task_names,
            "action_space": action_space,
            "action_key": "action",
            "available_action_keys": {
                "joint_absolute_7d": "action.joint_absolute",
                "ee_relative_10d": "action.ee_relative",
            },
            "state_key": "observation.state",
            "video_keys": list(video_keys.values()),
            "auxiliary_video_keys": (
                {role: AUXILIARY_VIDEO_KEYS[role] for role in camera_roles}
                if include_cosmoswriter
                else {}
            ),
            "rotation_6d": "first two columns of R_current^T R_next, column-major",
            "embodiment": embodiment,
            "native_action_schema": {
                "key": "action.joint_absolute",
                "type": "absolute_joint_target",
                "shape": [7],
                "names": [*embodiment["joint_names"], "gripper"],
                "units": [
                    *["rad"] * 6,
                    "binary"
                    if embodiment["gripper_mode"] == "binary"
                    else "normalized",
                ],
                "fps": sample_rate_hz,
                "gripper_convention": "0=open, 1=closed",
            },
            "source_schema_version": episodes[0].manifest["schema_version"],
            "source_run_schema_version": run_schema_version,
            "included_outcomes": sorted(outcomes),
            "split_episodes": split_episodes,
            "split_episode_selection": split_episode_selection,
        }
        (metadata_root / "cosmos_policy.json").write_text(
            json.dumps(adapter, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, output)
    except BaseException as exc:
        shutil.rmtree(temporary)
        temporary.mkdir()
        (temporary / "conversion_error.json").write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n"
        )
        raise

    return {
        "output": str(output),
        "episodes": len(derived_episodes),
        "frames": global_index,
        "action_space": action_space,
        "outcomes": sorted(outcomes),
        "include_cosmoswriter": include_cosmoswriter,
        "split_episodes": split_episodes,
        "split_episode_selection": split_episode_selection,
        "dataset_variant": dataset_variant,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="native run directory or episodes directory"
    )
    parser.add_argument("output", type=Path, help="new LeRobot v3 dataset directory")
    parser.add_argument(
        "--action-space", choices=sorted(ACTION_DIMS), default="joint_absolute"
    )
    parser.add_argument(
        "--outcomes",
        default="success",
        help="comma-separated native outcomes to include (default: success)",
    )
    parser.add_argument("--task", help="override the run's natural-language task")
    parser.add_argument(
        "--include-cosmoswriter",
        action="store_true",
        help="include and normalize all CosmosWriter auxiliary videos",
    )
    parser.add_argument(
        "--split-episodes",
        action="store_true",
        help="emit the overall episode plus one episode per captured stage",
    )
    parser.add_argument(
        "--split-episode-count",
        type=int,
        help="randomly select this many eligible episodes for stage splitting",
    )
    parser.add_argument(
        "--split-episode-seed",
        type=int,
        default=0,
        help="deterministic sampling seed (default: 0)",
    )
    args = parser.parse_args()
    args.outcomes = frozenset(
        value.strip() for value in args.outcomes.split(",") if value.strip()
    )
    allowed = {"success", "failure", "incomplete"}
    if not args.outcomes or not args.outcomes <= allowed:
        parser.error(f"--outcomes must contain only {sorted(allowed)}")
    return args


def main() -> int:
    args = _parse_args()
    result = convert(
        args.source,
        args.output,
        action_space=args.action_space,
        outcomes=args.outcomes,
        task=args.task,
        include_cosmoswriter=args.include_cosmoswriter,
        split_episodes=args.split_episodes,
        split_episode_count=args.split_episode_count,
        split_episode_seed=args.split_episode_seed,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
