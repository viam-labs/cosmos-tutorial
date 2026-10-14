import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_train.convert_native import (
    NativeEpisode,
    _consistent_embodiment,
    _native_episode_paths,
    convert,
)


def _manifest(
    *,
    arm: str = "lite6",
    gripper: str = "lite6_gripper",
    gripper_mode: str = "binary",
    names=None,
):
    return {
        "robot": {
            "arm_model": arm,
            "gripper_model": gripper,
            "gripper_mode": gripper_mode,
            "joint_names": names or [f"joint{index}" for index in range(1, 7)],
        }
    }


ROLES = ("fixed", "wrist")
MODALITIES = ("rgb", "depth", "edges", "segmentation", "shaded_seg")
CANONICAL_VIDEO_KEYS = {
    "fixed": "observation.images.fixed",
    "wrist": "observation.images.wrist",
}
AUXILIARY_VIDEO_KEYS = {
    role: {
        modality: f"auxiliary.cosmoswriter.{role}.{modality}" for modality in MODALITIES
    }
    for role in ROLES
}


def _write_video(
    path: Path,
    levels: list[int],
    *,
    fps: float,
    width: int = 16,
    height: int = 12,
    keyframe_indices: frozenset[int] = frozenset({0}),
    encoder_options: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(
            "libx264",
            rate=Fraction(str(fps)),
            options=encoder_options or {},
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame_index, level in enumerate(levels):
            rgb = np.full((height, width, 3), level % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            if frame_index in keyframe_indices:
                frame.pict_type = av.video.frame.PictureType.I
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _video_properties(
    path: Path,
) -> tuple[dict[str, object], list[float], list[float]]:
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        frames = list(container.decode(video=0))
        properties: dict[str, object] = {
            "codec": stream.codec_context.name,
            "pixel_format": stream.codec_context.pix_fmt,
            "width": stream.width,
            "height": stream.height,
            "fps": float(stream.average_rate),
        }
        means = [float(frame.to_ndarray(format="rgb24").mean()) for frame in frames]
        timestamps = [float(frame.pts * stream.time_base) for frame in frames]
    return properties, means, timestamps


def _video_payloads(path: Path) -> list[bytes]:
    with av.open(str(path), mode="r") as container:
        return [
            bytes(packet)
            for packet in container.demux(video=0)
            if packet.pts is not None
        ]


def _native_samples(
    count: int, episode_index: int, sample_rate_hz: float
) -> tuple[pa.Table, dict[str, np.ndarray]]:
    frame = np.arange(count, dtype=np.float32)
    joint_position = (
        episode_index + frame[:, None] + np.arange(6, dtype=np.float32)[None, :] / 10
    )
    joint_target = joint_position + np.float32(0.25)
    joint_velocity = np.full((count, 6), episode_index + 0.5, dtype=np.float32)
    ee_position = np.column_stack((frame / 10, frame / 5, np.full(count, 0.3)))
    ee_rotation = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (count, 1))
    gripper_command = (np.arange(count) % 2).astype(np.float32)
    gripper_state = gripper_command.copy()
    table = pa.Table.from_pydict(
        {
            "frame_index": np.arange(count, dtype=np.int64),
            "sim_time_s": episode_index
            + np.arange(count, dtype=np.float64) / sample_rate_hz,
            "joint_position_rad": joint_position.tolist(),
            "joint_velocity_rad_s": joint_velocity.tolist(),
            "joint_target_rad": joint_target.tolist(),
            "ee_position_m": ee_position.tolist(),
            "ee_rotation_wxyz": ee_rotation.tolist(),
            "gripper_command": gripper_command,
            "gripper_state": gripper_state,
        }
    )
    joint_action = np.column_stack((joint_target, gripper_command)).astype(np.float32)
    ee_action = np.tile(
        np.asarray([0.1, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        (count, 1),
    ).astype(np.float32)
    ee_action[-1, :3] = 0.0
    ee_action[:, -1] = gripper_command
    return table, {"joint": joint_action, "ee": ee_action}


def _cosmoswriter_export(
    root: Path,
    episode_index: int,
    count: int,
    sample_rate_hz: float = 20.0,
    roles_to_write=ROLES,
) -> dict[str, object]:
    roles: dict[str, dict[str, dict[str, object]]] = {}
    for role_index, role in enumerate(roles_to_write):
        roles[role] = {}
        for modality_index, modality in enumerate(MODALITIES):
            base_level = 30 + episode_index * 2 + role_index * 10
            base_level += modality_index * 20
            levels = [base_level + i * 35 for i in range(count)]
            # Solid grayscale makes frame order visible after lossy encoding.
            relative_path = Path(role) / "clip_0000" / f"{modality}.mp4"
            _write_video(
                root / relative_path,
                levels,
                fps=sample_rate_hz,
                keyframe_indices=frozenset(range(count)),
            )
            roles[role][modality] = {
                "relative_path": relative_path.as_posix(),
                "frame_count": count,
                "width": 16,
                "height": 12,
                "codec": "h264",
                "pixel_format": "yuv420p",
                "raw_average_fps": float(sample_rate_hz),
            }
    return {"path": str(root), "roles": roles}


def _write_native_run(
    root: Path,
    counts: tuple[int, ...],
    run_schema_version: int = 4,
    cosmos_output: bool = True,
    roles_to_write=ROLES,
    sample_rate_hz: float | tuple[float, ...] = 20.0,
    stage_tasks: tuple[str, ...] = ("Center", "Rotate", "Descend", "Grip and lift"),
) -> dict[str, dict[str, np.ndarray]]:
    episodes_root = root / "episodes"
    if any(count < 4 for count in counts):
        raise ValueError("test episodes need at least one frame per stage")
    episodes_root.mkdir(parents=True)
    cosmos_root = root / "cosmoswriter"
    attempts = []
    actions: dict[str, dict[str, np.ndarray]] = {}

    for episode_index, count in enumerate(counts):
        episode_rate = (
            sample_rate_hz[episode_index]
            if isinstance(sample_rate_hz, tuple)
            else sample_rate_hz
        )
        episode_id = f"episode-{episode_index:03d}"
        episode = episodes_root / episode_id
        episode.mkdir()
        table, actions[episode_id] = _native_samples(count, episode_index, episode_rate)
        pq.write_table(table, episode / "samples.parquet")
        cameras = []
        for role_index, role in enumerate(roles_to_write):
            levels = [15 + role_index * 10 + frame * 30 for frame in range(count)]
            _write_video(
                episode / f"{role}.mp4",
                levels,
                fps=episode_rate,
                keyframe_indices=frozenset(range(count)),
            )
            cameras.append(
                {
                    "role": role,
                    "name": f"{role}_camera",
                    "file": f"{role}.mp4",
                    "width": 16,
                    "height": 12,
                }
            )
        base_length, remainder = divmod(count, len(stage_tasks))
        stages = []
        start_frame = 0
        for stage_index, stage_task in enumerate(stage_tasks):
            length = base_length + (1 if stage_index < remainder else 0)
            end_frame = start_frame + length
            stages.append(
                {
                    "start_frame": start_frame,
                    "end_frame_exclusive": end_frame,
                    "task": stage_task,
                }
            )
            start_frame = end_frame
        manifest = {
            "schema_version": 3,
            "episode_id": episode_id,
            "seed": 100 + episode_index,
            "started_at": "2026-01-01T00:00:00Z",
            "sample_rate_hz": episode_rate,
            "sample_count": count,
            "samples_file": "samples.parquet",
            "outcome": "success",
            "task": "Pick up the block",
            "stages": stages,
            "robot": _manifest()["robot"],
            "cameras": cameras,
            "target": {"position_m": [0.4, 0.0, 0.1]},
        }
        (episode / "manifest.json").write_text(json.dumps(manifest))
        cosmoswriter = _cosmoswriter_export(
            cosmos_root / episode_id,
            episode_index,
            count,
            episode_rate,
            roles_to_write,
        )
        attempts.append(
            {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "seed": manifest["seed"],
                "status": "finalized",
                "outcome": "success",
                "sample_count": count,
                "path": str(episode),
                "cosmoswriter": cosmoswriter,
            }
        )

    config: dict[str, object] = {"task": "Pick up the block"}
    if run_schema_version == 4:
        config["cosmos_output"] = cosmos_output
    (root / "run.json").write_text(
        json.dumps(
            {
                "schema_version": run_schema_version,
                "config": config,
                "attempts": attempts,
            }
        )
    )
    return actions


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def test_run_input_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    (tmp_path / "episodes").mkdir()
    (tmp_path / "run.json").write_text(
        json.dumps({"schema_version": 1, "config": {"task": "pick"}})
    )

    with pytest.raises(ValueError, match="schema_version"):
        _native_episode_paths(tmp_path)


def test_run_input_propagates_schema_four_task(tmp_path: Path) -> None:
    episode = tmp_path / "episodes/000000"
    episode.mkdir(parents=True)
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "config": {"task": "Pick up the block", "cosmos_output": False},
            }
        )
    )

    paths, task, schema_version = _native_episode_paths(tmp_path)

    assert paths == [episode]
    assert task == "Pick up the block"
    assert schema_version == 4


@pytest.mark.parametrize(
    "changed",
    [
        _manifest(arm="ur20"),
        _manifest(gripper="robotiq_2f_85"),
        _manifest(names=["joint2", "joint1", "joint3", "joint4", "joint5", "joint6"]),
    ],
)
def test_mixed_embodiment_is_rejected(changed: dict) -> None:
    episodes = [
        NativeEpisode(Path("first"), _manifest(), {}),
        NativeEpisode(Path("changed"), changed, {}),
    ]

    with pytest.raises(ValueError, match="mixed embodiment"):
        _consistent_embodiment(episodes)


def test_default_conversion_keeps_whole_episode(tmp_path: Path) -> None:
    source = tmp_path / "native"
    _write_native_run(source, (4,))
    output = tmp_path / "whole"

    result = convert(source, output)

    info = _read_json(output / "meta/info.json")
    assert result["split_episodes"] is False
    assert result["episodes"] == info["total_episodes"] == 1
    assert result["frames"] == info["total_frames"] == 4
    assert info["total_tasks"] == 1
    assert pq.read_table(output / "meta/tasks.parquet").to_pylist() == [
        {"task_index": 0, "task": "Pick up the block"}
    ]
    assert not (output / "data/chunk-000/file-001.parquet").exists()


def test_split_conversion_samples_source_episodes_deterministically(
    tmp_path: Path,
) -> None:
    source = tmp_path / "native"
    _write_native_run(source, (4, 4, 4))
    output = tmp_path / "sampled"

    result = convert(
        source,
        output,
        split_episodes=True,
        split_episode_count=1,
        split_episode_seed=0,
    )

    selection = {
        "mode": "sample",
        "eligible_count": 3,
        "sample_count": 1,
        "episode_ids": ["episode-001"],
        "seed": 0,
    }
    assert result["split_episode_selection"] == selection
    assert result["episodes"] == 7
    assert result["frames"] == 16
    policy = _read_json(output / "meta/cosmos_policy.json")
    assert policy["split_episode_selection"] == selection
    episode_rows = pq.read_table(
        output / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    assert [row["episode_id"] for row in episode_rows] == [
        "episode-000",
        "episode-001",
        "episode-001:stage-0",
        "episode-001:stage-1",
        "episode-001:stage-2",
        "episode-001:stage-3",
        "episode-002",
    ]


def test_split_conversion_rejects_invalid_sample_count(tmp_path: Path) -> None:
    source = tmp_path / "native"
    _write_native_run(source, (4, 4))

    with pytest.raises(ValueError, match="requires split_episodes"):
        convert(source, tmp_path / "disabled", split_episode_count=1)
    with pytest.raises(ValueError, match="exceeds 2 eligible episodes"):
        convert(
            source,
            tmp_path / "too-many",
            split_episodes=True,
            split_episode_count=3,
        )


def test_split_conversion_requires_stage_aligned_closed_gop_idrs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "native"
    _write_native_run(source, (12,))
    video = source / "episodes/episode-000/fixed.mp4"
    levels = [15 + frame * 10 for frame in range(12)]

    _write_video(
        video,
        levels,
        fps=20,
        keyframe_indices=frozenset({0, 2, 5, 8}),
    )
    with pytest.raises(ValueError, match=r"stage (frame|IDRs).*(3|6|9)"):
        convert(source, tmp_path / "misaligned", split_episodes=True)

    _write_video(
        video,
        levels,
        fps=20,
        keyframe_indices=frozenset(),
        encoder_options={"x264-params": "open-gop=1:keyint=3:min-keyint=3:scenecut=0"},
    )
    with pytest.raises(ValueError, match=r"stage frame (3|6|9).*closed-GOP IDR"):
        convert(source, tmp_path / "open-gop", split_episodes=True)


def test_conversion_preserves_optional_second_fixed_camera(tmp_path: Path) -> None:
    source = tmp_path / "native"
    _write_native_run(source, (4,), roles_to_write=("fixed", "wrist", "fixed_2"))
    output = tmp_path / "converted"

    convert(source, output, include_cosmoswriter=True)

    policy = _read_json(output / "meta/cosmos_policy.json")
    assert policy["video_keys"] == [
        "observation.images.fixed",
        "observation.images.wrist",
        "observation.images.fixed_2",
    ]
    assert set(policy["auxiliary_video_keys"]) == {"fixed", "wrist", "fixed_2"}
    assert (
        output / "videos/observation.images.fixed_2/chunk-000/file-000.mp4"
    ).is_file()


def test_direct_conversion_omits_cosmoswriter_without_opt_in(tmp_path: Path) -> None:
    source = tmp_path / "native"
    actions = _write_native_run(source, (4,), sample_rate_hz=15.0)
    output = tmp_path / "direct"

    result = convert(source, output, action_space="ee_relative", split_episodes=True)

    assert result["dataset_variant"] == "direct_dataset"
    assert result["include_cosmoswriter"] is False
    assert result["split_episodes"] is True
    info = _read_json(output / "meta/info.json")
    assert info["fps"] == 15.0
    assert set(CANONICAL_VIDEO_KEYS.values()) <= set(info["features"])
    assert not any(
        key.startswith("auxiliary.cosmoswriter.") for key in info["features"]
    )
    assert info["features"]["action"]["shape"] == [10]
    assert info["features"]["action.joint_absolute"]["shape"] == [7]
    assert info["features"]["action.ee_relative"]["shape"] == [10]
    assert info["total_episodes"] == 5
    assert info["total_frames"] == 8
    assert info["total_tasks"] == 5

    table = pq.read_table(output / "data/chunk-000/file-000.parquet")
    expected = actions["episode-000"]
    np.testing.assert_allclose(
        np.asarray(table["action.joint_absolute"].to_pylist()), expected["joint"]
    )
    np.testing.assert_allclose(
        np.asarray(table["action.ee_relative"].to_pylist()), expected["ee"]
    )
    np.testing.assert_allclose(np.asarray(table["action"].to_pylist()), expected["ee"])

    for video_key in CANONICAL_VIDEO_KEYS.values():
        path = output / f"videos/{video_key}/chunk-000/file-000.mp4"
        properties, means, timestamps = _video_properties(path)
        assert properties == {
            "codec": "h264",
            "pixel_format": "yuv420p",
            "width": 16,
            "height": 12,
            "fps": pytest.approx(15.0),
        }
        assert len(means) == 4
        np.testing.assert_allclose(timestamps, np.arange(4) / 15, atol=1e-6, rtol=0.0)
        source_payloads = _video_payloads(
            source / f"episodes/episode-000/{video_key.rsplit('.', 1)[-1]}.mp4"
        )
        ranges = ((0, 4), (0, 1), (1, 2), (2, 3), (3, 4))
        for file_index, (start, end) in enumerate(ranges):
            split_path = (
                output / f"videos/{video_key}/chunk-000/file-{file_index:03d}.mp4"
            )
            assert _video_payloads(split_path) == source_payloads[start:end]
    assert not (output / "videos/auxiliary.cosmoswriter.fixed.rgb").exists()
    episode = pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet")
    assert not any(
        name.startswith("videos/auxiliary.cosmoswriter.")
        for name in episode.column_names
    )
    episode_rows = episode.to_pylist()
    assert [row["episode_id"] for row in episode_rows] == [
        "episode-000",
        "episode-000:stage-0",
        "episode-000:stage-1",
        "episode-000:stage-2",
        "episode-000:stage-3",
    ]
    assert [row["length"] for row in episode_rows] == [4, 1, 1, 1, 1]
    assert {row["parent_episode_id"] for row in episode_rows} == {"episode-000"}
    tasks = pq.read_table(output / "meta/tasks.parquet").to_pylist()
    assert tasks == [
        {"task_index": 0, "task": "Pick up the block"},
        {"task_index": 1, "task": "Center"},
        {"task_index": 2, "task": "Rotate"},
        {"task_index": 3, "task": "Descend"},
        {"task_index": 4, "task": "Grip and lift"},
    ]
    stage_table = pq.read_table(output / "data/chunk-000/file-001.parquet")
    assert stage_table["frame_index"].to_pylist() == [0]
    assert stage_table["timestamp"].to_pylist() == pytest.approx([0.0])
    assert stage_table["task_index"].to_pylist() == [1]
    np.testing.assert_allclose(
        np.asarray(stage_table["action.joint_absolute"].to_pylist()),
        expected["joint"][:1],
    )
    np.testing.assert_allclose(
        np.asarray(stage_table["action.ee_relative"].to_pylist())[0, :3],
        np.zeros(3),
    )

    policy = _read_json(output / "meta/cosmos_policy.json")
    assert policy["dataset_variant"] == "direct_dataset"
    assert policy["fps"] == 15.0
    assert policy["native_action_schema"]["fps"] == 15.0
    assert policy["video_keys"] == list(CANONICAL_VIDEO_KEYS.values())
    assert policy["auxiliary_video_keys"] == {}
    assert policy["source_run_schema_version"] == 4
    assert policy["tasks"] == [row["task"] for row in tasks]


def test_cosmoswriter_opt_in_normalizes_every_auxiliary_video(tmp_path: Path) -> None:
    source = tmp_path / "native"
    actions = _write_native_run(source, (4,))
    raw_video = source / "cosmoswriter/episode-000/fixed/clip_0000/rgb.mp4"
    raw_properties, _, _ = _video_properties(raw_video)
    assert raw_properties["fps"] == pytest.approx(20.0)
    output = tmp_path / "with-cosmoswriter"

    result = convert(source, output, include_cosmoswriter=True, split_episodes=True)

    assert result["dataset_variant"] == "direct_dataset_with_cosmoswriter"
    assert result["include_cosmoswriter"] is True
    assert result["split_episodes"] is True
    policy = _read_json(output / "meta/cosmos_policy.json")
    assert policy["dataset_variant"] == "direct_dataset_with_cosmoswriter"
    assert policy["video_keys"] == list(CANONICAL_VIDEO_KEYS.values())
    assert policy["auxiliary_video_keys"] == AUXILIARY_VIDEO_KEYS

    info = _read_json(output / "meta/info.json")
    assert info["fps"] == 20
    assert info["total_episodes"] == 5
    assert info["total_frames"] == 8
    auxiliary_keys = [
        AUXILIARY_VIDEO_KEYS[role][modality]
        for role in ROLES
        for modality in MODALITIES
    ]
    assert {
        key for key in info["features"] if key.startswith("auxiliary.cosmoswriter.")
    } == set(auxiliary_keys)
    for video_key in auxiliary_keys:
        feature = info["features"][video_key]
        assert feature["dtype"] == "video"
        assert feature["shape"] == [12, 16, 3]
        assert feature["info"] == {
            "has_audio": False,
            "video.channels": 3,
            "video.codec": "h264",
            "video.fps": 20,
            "video.height": 12,
            "video.is_depth_map": False,
            "video.pix_fmt": "yuv420p",
            "video.width": 16,
        }

    episode_rows = pq.read_table(
        output / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    all_video_keys = [*CANONICAL_VIDEO_KEYS.values(), *auxiliary_keys]
    ranges = ((0, 4), (0, 1), (1, 2), (2, 3), (3, 4))
    for episode_index, (row, (start, end)) in enumerate(zip(episode_rows, ranges)):
        count = end - start
        for video_key in all_video_keys:
            prefix = f"videos/{video_key}"
            assert row[f"{prefix}/chunk_index"] == 0
            assert row[f"{prefix}/file_index"] == episode_index
            assert row[f"{prefix}/from_timestamp"] == 0.0
            assert row[f"{prefix}/to_timestamp"] == pytest.approx(count / 20)

        table = pq.read_table(
            output / f"data/chunk-000/file-{episode_index:03d}.parquet"
        )
        np.testing.assert_allclose(
            np.asarray(table["action.joint_absolute"].to_pylist()),
            actions["episode-000"]["joint"][start:end],
        )
        for role in ROLES:
            for modality in MODALITIES:
                video_key = AUXILIARY_VIDEO_KEYS[role][modality]
                path = (
                    output
                    / f"videos/{video_key}/chunk-000/file-{episode_index:03d}.mp4"
                )
                properties, means, timestamps = _video_properties(path)
                assert properties == {
                    "codec": "h264",
                    "pixel_format": "yuv420p",
                    "width": 16,
                    "height": 12,
                    "fps": pytest.approx(20.0),
                }
                source_video = (
                    source / f"cosmoswriter/episode-000/{role}/clip_0000/{modality}.mp4"
                )
                source_properties, source_means, _ = _video_properties(source_video)
                assert source_properties["fps"] == pytest.approx(20.0)
                np.testing.assert_allclose(
                    means, source_means[start:end], atol=4.0, rtol=0.0
                )
                np.testing.assert_allclose(
                    timestamps, np.arange(count) / 20, atol=1e-6, rtol=0.0
                )


def test_schema_four_cosmoswriter_ignores_attempt_path(tmp_path: Path) -> None:
    source = tmp_path / "native"
    _write_native_run(source, (4,))
    tampered = tmp_path / "tampered-cosmoswriter"
    _cosmoswriter_export(tampered, 0, 4)
    tampered_video = tampered / "fixed/clip_0000/rgb.mp4"
    _write_video(tampered_video, [220] * 4, fps=60)

    run_path = source / "run.json"
    run = _read_json(run_path)
    run["attempts"][0]["cosmoswriter"]["path"] = str(tampered)
    run_path.write_text(json.dumps(run))

    output = tmp_path / "converted"
    convert(source, output, include_cosmoswriter=True)

    converted = (
        output / "videos/auxiliary.cosmoswriter.fixed.rgb/chunk-000/file-000.mp4"
    )
    source_video = source / "cosmoswriter/episode-000/fixed/clip_0000/rgb.mp4"
    _, converted_means, _ = _video_properties(converted)
    _, source_means, _ = _video_properties(source_video)
    _, tampered_means, _ = _video_properties(tampered_video)
    np.testing.assert_allclose(converted_means, source_means, atol=4.0, rtol=0.0)
    assert not np.allclose(converted_means, tampered_means, atol=4.0, rtol=0.0)


def test_schema_four_cosmoswriter_opt_in_requires_enabled_capture(
    tmp_path: Path,
) -> None:
    source = tmp_path / "native"
    _write_native_run(source, (4,), cosmos_output=False)

    with pytest.raises(ValueError, match="cosmos_output"):
        convert(source, tmp_path / "converted", include_cosmoswriter=True)


def test_cosmoswriter_opt_in_rejects_mixed_incomplete_and_mismatched_data(
    tmp_path: Path,
) -> None:
    source = tmp_path / "native"
    _write_native_run(source, (4, 4))
    target = source / "cosmoswriter/episode-001/wrist/clip_0000/shaded_seg.mp4"
    original = target.read_bytes()

    target.unlink()
    with pytest.raises(ValueError, match="(?i)(missing|incomplete)"):
        convert(source, tmp_path / "missing-stream", include_cosmoswriter=True)
    target.write_bytes(original)

    _write_video(target, [32], fps=20)
    with pytest.raises(ValueError, match="(?i)(frame|sample_count)"):
        convert(source, tmp_path / "wrong-frame-count", include_cosmoswriter=True)
    target.write_bytes(original)

    _write_video(target, [32, 77, 120, 180], fps=20, width=18)
    with pytest.raises(ValueError, match="(?i)(geometry|format|dimension|width)"):
        convert(source, tmp_path / "wrong-geometry", include_cosmoswriter=True)
    target.write_bytes(original)

    run_path = source / "run.json"
    run = _read_json(run_path)
    del run["attempts"][1]["cosmoswriter"]
    run_path.write_text(json.dumps(run))
    with pytest.raises(TypeError, match="(?i)(cosmoswriter|mixed|selected episode)"):
        convert(source, tmp_path / "mixed-attempts", include_cosmoswriter=True)


def test_conversion_rejects_mixed_native_sample_rates(tmp_path: Path) -> None:
    source = tmp_path / "native"
    _write_native_run(source, (4, 4), sample_rate_hz=(15.0, 20.0))

    with pytest.raises(ValueError, match="mixed sample rates"):
        convert(source, tmp_path / "converted")


def test_conversion_accepts_merged_rotate_and_descend_stage(tmp_path: Path) -> None:
    source = tmp_path / "native"
    _write_native_run(
        source,
        (4,),
        sample_rate_hz=15.0,
        stage_tasks=("Center", "Rotate and descend", "Grip and lift"),
    )
    output = tmp_path / "converted"

    convert(source, output, split_episodes=True)

    info = _read_json(output / "meta/info.json")
    policy = _read_json(output / "meta/cosmos_policy.json")
    assert info["total_episodes"] == 4
    assert info["total_tasks"] == 4
    assert policy["tasks"] == [
        "Pick up the block",
        "Center",
        "Rotate and descend",
        "Grip and lift",
    ]
