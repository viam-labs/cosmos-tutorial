from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cosmos_train.export_bundle import package_export, resolve_export_runtime

COMMIT = "efba2076c80648663a89a7fdbc92eb262fae39ac"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _source_export(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    workspace = tmp_path / "training-workspace"
    dataset = workspace / "droid_lerobot_20260115_no_noops"
    for name, value in {
        "cosmos_policy.json": {
            "embodiment": {"arm_model": "lite6"},
            "video_keys": [
                "observation.images.wrist_image_left",
                "observation.images.exterior_image_1_left",
                "observation.images.exterior_image_2_left",
            ],
        },
        "info.json": {"fps": 20},
        "stats.json": {"action": {"mean": [0.0] * 7}},
        "source_dataset.json": {
            "schema_version": 1,
            "source": "viam-dataset",
            "dataset_id": "6ac7a551",
            "content_sha256": "b" * 64,
        },
    }.items():
        _write_json(dataset / "meta" / name, value)
    vae = workspace / "wan22_vae" / "Wan2.2_VAE.pth"
    vae.parent.mkdir(parents=True)
    vae.write_bytes(b"fixture-vae")
    training_config = workspace / "config.yaml"
    training_config.write_text(
        "dataloader_train:\n"
        "  dataloader:\n"
        "    datasets:\n"
        "      droid:\n"
        "        dataset:\n"
        f"          root: {dataset}\n"
        "          fps: 20.0\n"
        "          chunk_length: 32\n"
        "          action_space: joint_pos\n"
        "          action_normalization: null\n"
        "          viewpoint: concat_view\n"
        "          resolution: '480'\n"
        "          joint_count: 6\n"
        "          use_state: true\n"
        "          format_prompt_as_json: true\n"
        f"model:\n  config:\n    tokenizer:\n      vae_path: {vae}\n",
        encoding="utf-8",
    )

    export = workspace / "export"
    export.mkdir()
    _write_json(
        export / "checkpoint.json",
        {
            "config_file": str(training_config),
            "config_file_type": "yaml",
            "policy": {
                "action_chunk_size": 32,
                "conditioning_fps": 20.0,
                "domain_name": "droid_lerobot",
            },
        },
    )
    _write_json(
        export / "config.json",
        {
            "architectures": ["Cosmos3OmniModel"],
            "model": {
                "config": {"action_gen": True, "tokenizer": {"vae_path": str(vae)}}
            },
        },
    )
    _write_json(export / "export_manifest.json", {"framework_commit": COMMIT})
    _write_json(
        export / "policy_contract.json",
        {
            "schema_version": 3,
            "embodiment": {
                "arm_model": "lite6",
                "gripper_model": "lite6_gripper",
                "gripper_mode": "binary",
                "joint_names": [f"joint{i}" for i in range(1, 7)],
            },
            "action_schema": {"shape": [7]},
            "observation_schema": {
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
            },
            "framework": {
                "loader": "DROIDLeRobotDataset:joint_pos",
                "action_space": "joint_pos",
                "joint_count": 6,
                "use_state": True,
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
            },
            "action_chunk_size": 32,
            "conditioning_fps": 20.0,
            "domain_name": "droid_lerobot",
        },
    )
    return workspace, export, training_config, dataset, vae


def test_relocated_export_resolves_without_training_workspace(tmp_path: Path) -> None:
    workspace, export, training_config, dataset, vae = _source_export(tmp_path)
    package_export(
        export,
        training_config,
        dataset,
        vae,
        framework_commit=COMMIT,
        dataset_provenance=dataset / "meta/source_dataset.json",
    )
    relocated = tmp_path / "downloaded" / "policy"
    relocated.parent.mkdir()
    shutil.copytree(export, relocated)
    shutil.rmtree(workspace)

    runtime = resolve_export_runtime(relocated)

    assert runtime.training_config == relocated / "runtime/training_config.yaml"
    assert runtime.dataset_stats == relocated / "runtime/dataset/meta/stats.json"
    assert runtime.vae == relocated / "runtime/vae/Wan2.2_VAE.pth"
    assert str(workspace) not in runtime.training_config.read_text(encoding="utf-8")
    assert runtime.contract["transform"] == {
        "action_normalization": None,
        "viewpoint": "concat_view",
        "resolution": "480",
        "format_prompt_as_json": True,
        "append_viewpoint_info": True,
        "append_duration_fps_timestamps": True,
        "append_resolution_info": True,
        "append_idle_frames": False,
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
    checkpoint = json.loads((relocated / "checkpoint.json").read_text())
    config = json.loads((relocated / "config.json").read_text())
    assert checkpoint["config_file"] == "runtime/training_config.yaml"
    assert checkpoint["checkpoint_path"] == "."
    assert str(workspace) not in json.dumps(checkpoint)
    assert (
        config["model"]["config"]["tokenizer"]["vae_path"]
        == "runtime/vae/Wan2.2_VAE.pth"
    )
    manifest = json.loads((relocated / "runtime_manifest.json").read_text())
    assert manifest["dataset"]["normalization"] == {
        "mode": None,
        "statistics": "runtime/dataset/meta/stats.json",
        "embedded_in_checkpoint": False,
        "statistics_used": False,
    }
    assert manifest["dataset"]["source"]["dataset_id"] == "6ac7a551"
    assert manifest["dataset"]["source"]["content_sha256"] == "b" * 64


def test_package_export_rejects_incomplete_provenance(tmp_path: Path) -> None:
    _, export, training_config, dataset, vae = _source_export(tmp_path)
    provenance = dataset / "meta/source_dataset.json"
    _write_json(provenance, {"schema_version": 1, "dataset_id": "6ac7a551"})
    with pytest.raises(ValueError, match="dataset_id and content_sha256"):
        package_export(
            export, training_config, dataset, vae, framework_commit=COMMIT, dataset_provenance=provenance
        )



def test_runtime_rejects_missing_or_conflicting_artifacts(tmp_path: Path) -> None:
    _, export, training_config, dataset, vae = _source_export(tmp_path)
    package_export(
        export,
        training_config,
        dataset,
        vae,
        framework_commit=COMMIT,
        dataset_provenance=dataset / "meta/source_dataset.json",
    )

    missing = tmp_path / "missing"
    shutil.copytree(export, missing)
    (missing / "runtime/dataset/meta/stats.json").unlink()
    with pytest.raises(ValueError, match="required dataset metadata is missing"):
        resolve_export_runtime(missing)

    conflicting = tmp_path / "conflicting"
    shutil.copytree(export, conflicting)
    stats = conflicting / "runtime/dataset/meta/stats.json"
    stats.write_text('{"action": {"mean": [123]}}', encoding="utf-8")
    with pytest.raises(ValueError, match="dataset metadata checksum conflict"):
        resolve_export_runtime(conflicting)

    wrong_vae = tmp_path / "wrong-vae.pth"
    wrong_vae.write_bytes(b"different")
    with pytest.raises(ValueError, match="VAE override conflicts"):
        resolve_export_runtime(export, vae_override=wrong_vae)
