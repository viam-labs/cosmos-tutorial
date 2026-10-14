"""Package and validate the runtime artifacts required by a Cosmos policy export."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 2
RUNTIME_MANIFEST = "runtime_manifest.json"
_REQUIRED_DATASET_METADATA = (
    "cosmos_policy.json",
    "info.json",
    "source_dataset.json",
    "stats.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid required JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid required JSON {path}: expected an object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"runtime manifest field {field!r} must be a relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"runtime manifest field {field!r} is not a portable relative path: {value!r}")
    resolved = root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"runtime manifest field {field!r} escapes the export: {value!r}") from exc
    return resolved


def _copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ValueError(f"required runtime artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

def _yaml_scalars(text: str, key: str) -> set[str]:
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*([^#]*?)\s*(?:#.*)?$", re.MULTILINE)
    return {match.group(1).strip().strip("\"'").lower() for match in pattern.finditer(text)}


def _validate_training_contract(text: str, contract: dict[str, Any]) -> None:
    transform = contract.get("transform")
    framework = contract.get("framework")
    if not isinstance(transform, dict) or not isinstance(framework, dict):
        raise ValueError("policy contract lacks transform or framework metadata")
    conditioning_fps = contract.get("conditioning_fps")
    if not isinstance(conditioning_fps, (int, float)):
        raise ValueError("policy contract conditioning_fps must be numeric")
    expected = {
        "fps": str(float(conditioning_fps)).lower(),
        "chunk_length": str(contract.get("action_chunk_size")).lower(),
        "action_space": str(framework.get("action_space")).lower(),
        "joint_count": str(framework.get("joint_count")).lower(),
        "use_state": str(bool(framework.get("use_state"))).lower(),
        "action_normalization": "null" if transform.get("action_normalization") is None else str(transform["action_normalization"]).lower(),
        "viewpoint": str(transform.get("viewpoint")).lower(),
        "resolution": str(transform.get("resolution")).lower(),
        "format_prompt_as_json": str(bool(transform.get("format_prompt_as_json"))).lower(),
    }
    for key, value in expected.items():
        values = _yaml_scalars(text, key)
        numeric_match = key == "fps" and value.removesuffix(".0") in values
        if value not in values and not numeric_match:
            raise ValueError(
                f"training config conflicts with policy contract for {key}: expected {value!r}, found {sorted(values)!r}"
            )


def _replace_exact_path(text: str, source: Path, replacement: str, label: str) -> str:
    candidates = {str(source), str(source.resolve())}
    matches = [candidate for candidate in candidates if candidate in text]
    if not matches:
        raise ValueError(f"training config does not contain the expected {label} path {source}")
    for candidate in matches:
        text = text.replace(candidate, replacement)
    return text


def _validate_provenance(provenance: Mapping[str, Any]) -> None:
    """Require Viam dataset provenance: dataset_id + content_sha256."""
    if not (
        isinstance(provenance.get("dataset_id"), str)
        and isinstance(provenance.get("content_sha256"), str)
    ):
        raise ValueError("dataset provenance must contain dataset_id and content_sha256")


@dataclass(frozen=True)
class ResolvedExportRuntime:
    training_config: Path
    dataset_root: Path
    dataset_policy: Path
    dataset_info: Path
    dataset_stats: Path
    policy_contract: Path
    vae: Path
    contract: dict[str, Any]


def package_export(
    export_dir: Path,
    training_config: Path,
    dataset_root: Path,
    vae: Path,
    *,
    framework_commit: str,
    dataset_provenance: Path,
) -> None:
    """Copy runtime inputs into *export_dir* and rewrite every reference portably."""
    export_dir = export_dir.resolve()
    if not export_dir.is_dir():
        raise ValueError(f"export directory does not exist: {export_dir}")
    checkpoint_path = export_dir / "checkpoint.json"
    config_path = export_dir / "config.json"
    contract_path = export_dir / "policy_contract.json"
    checkpoint = _read_json(checkpoint_path)
    config = _read_json(config_path)
    contract = _read_json(contract_path)

    manifest_path = export_dir / "export_manifest.json"
    export_manifest = _read_json(manifest_path)
    if export_manifest.get("framework_commit") != framework_commit:
        raise ValueError(
            f"export framework commit conflicts with packager: {export_manifest.get('framework_commit')!r} != {framework_commit!r}"
        )
    provenance = _read_json(dataset_provenance)
    _validate_provenance(provenance)

    runtime = export_dir / "runtime"
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir()
    bundled_config = runtime / "training_config.yaml"
    dataset_destination = runtime / "dataset"
    vae_destination = runtime / "vae" / vae.name

    config_text = training_config.read_text(encoding="utf-8")
    _validate_training_contract(config_text, contract)
    config_text = _replace_exact_path(config_text, dataset_root, "runtime/dataset", "dataset")
    config_text = _replace_exact_path(config_text, vae, f"runtime/vae/{vae.name}", "VAE")
    bundled_config.write_text(config_text, encoding="utf-8")

    metadata_hashes: dict[str, str] = {}
    for name in _REQUIRED_DATASET_METADATA:
        source = dataset_provenance if name == "source_dataset.json" else dataset_root / "meta" / name
        destination = dataset_destination / "meta" / name
        _copy_required(source, destination)
        metadata_hashes[name] = _sha256(destination)
    _copy_required(vae, vae_destination)

    tokenizer = config.get("model", {}).get("config", {}).get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError("export config has no model.config.tokenizer object")
    tokenizer["vae_path"] = f"runtime/vae/{vae.name}"
    _write_json(config_path, config)

    checkpoint["config_file"] = "runtime/training_config.yaml"
    checkpoint["config_file_type"] = "yaml"
    checkpoint["runtime_manifest"] = RUNTIME_MANIFEST
    checkpoint["checkpoint_path"] = "."
    checkpoint["checkpoint_type"] = "safetensors"
    if "credential_path" in checkpoint:
        checkpoint["credential_path"] = None
    _write_json(checkpoint_path, checkpoint)

    runtime_manifest = {
        "schema_version": SCHEMA_VERSION,
        "framework_commit": framework_commit,
        "training_config": {
            "path": "runtime/training_config.yaml",
            "sha256": _sha256(bundled_config),
        },
        "dataset": {
            "root": "runtime/dataset",
            "metadata_sha256": metadata_hashes,
            "source": provenance,
            "normalization": {
                "mode": contract.get("transform", {}).get("action_normalization"),
                "statistics": "runtime/dataset/meta/stats.json",
                "embedded_in_checkpoint": False,
                "statistics_used": contract.get("transform", {}).get("action_normalization") is not None,
            },
        },
        "policy_contract": {
            "path": "policy_contract.json",
            "sha256": _sha256(contract_path),
        },
        "vae": {"path": f"runtime/vae/{vae.name}", "sha256": _sha256(vae_destination)},
    }
    _write_json(export_dir / RUNTIME_MANIFEST, runtime_manifest)


def resolve_export_runtime(export_dir: Path, *, vae_override: Path | None = None) -> ResolvedExportRuntime:
    """Resolve a relocated export and reject absent, stale, or conflicting artifacts."""
    export_dir = export_dir.resolve()
    checkpoint = _read_json(export_dir / "checkpoint.json")
    config = _read_json(export_dir / "config.json")
    manifest_name = checkpoint.get("runtime_manifest")
    manifest_path = _portable_path(export_dir, manifest_name, "checkpoint.runtime_manifest")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported runtime manifest schema: {manifest.get('schema_version')!r}")
    export_manifest = _read_json(export_dir / "export_manifest.json")
    if manifest.get("framework_commit") != export_manifest.get("framework_commit"):
        raise ValueError("runtime manifest framework commit conflicts with export manifest")

    def checked(entry: object, field: str) -> Path:
        if not isinstance(entry, dict):
            raise ValueError(f"runtime manifest field {field!r} must be an object")
        path = _portable_path(export_dir, entry.get("path"), f"{field}.path")
        expected = entry.get("sha256")
        if not path.is_file():
            raise ValueError(f"required runtime artifact is missing: {path}")
        actual = _sha256(path)
        if not isinstance(expected, str) or actual != expected:
            raise ValueError(f"runtime artifact checksum conflict for {field}: {path}")
        return path

    training_config = checked(manifest.get("training_config"), "training_config")
    contract_path = checked(manifest.get("policy_contract"), "policy_contract")
    bundled_vae = checked(manifest.get("vae"), "vae")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("runtime manifest field 'dataset' must be an object")
    dataset_root = _portable_path(export_dir, dataset.get("root"), "dataset.root")
    metadata_hashes = dataset.get("metadata_sha256")
    if not isinstance(metadata_hashes, dict) or set(metadata_hashes) != set(_REQUIRED_DATASET_METADATA):
        raise ValueError("runtime manifest dataset metadata set is incomplete or conflicting")
    metadata_paths: dict[str, Path] = {}
    for name in _REQUIRED_DATASET_METADATA:
        path = dataset_root / "meta" / name
        if not path.is_file():
            raise ValueError(f"required dataset metadata is missing: {path}")
        if _sha256(path) != metadata_hashes[name]:
            raise ValueError(f"dataset metadata checksum conflict: {path}")
        metadata_paths[name] = path

    if checkpoint.get("config_file") != manifest["training_config"]["path"]:
        raise ValueError("checkpoint config_file conflicts with runtime manifest")
    tokenizer = config.get("model", {}).get("config", {}).get("tokenizer")
    if not isinstance(tokenizer, dict) or tokenizer.get("vae_path") != manifest["vae"]["path"]:
        raise ValueError("model tokenizer VAE path conflicts with runtime manifest")
    contract = _read_json(contract_path)
    _validate_training_contract(training_config.read_text(encoding="utf-8"), contract)
    policy = checkpoint.get("policy")
    expected_policy = {
        "action_chunk_size": contract.get("action_chunk_size"),
        "conditioning_fps": contract.get("conditioning_fps"),
        "domain_name": contract.get("domain_name"),
    }
    dataset_policy = _read_json(metadata_paths["cosmos_policy.json"])
    video_keys = dataset_policy.get("video_keys")
    observation = contract.get("observation_schema")
    transform = contract.get("transform")
    expected_aliases = (
        {key: key.replace("observation.images.", "observation/", 1) for key in video_keys}
        if isinstance(video_keys, list) and all(isinstance(key, str) for key in video_keys)
        else None
    )
    if (
        expected_aliases is None
        or not isinstance(observation, dict)
        or not isinstance(transform, dict)
        or observation.get("camera_keys") != video_keys
        or observation.get("camera_aliases") != expected_aliases
        or transform.get("camera_keys") != video_keys
        or transform.get("camera_aliases") != expected_aliases
    ):
        raise ValueError("packaged camera metadata conflicts with policy contract")
    if policy != expected_policy:
        raise ValueError("checkpoint policy metadata conflicts with policy contract")
    normalization = dataset.get("normalization")
    if not isinstance(normalization, dict) or normalization.get("embedded_in_checkpoint") is not False:
        raise ValueError("runtime manifest must explicitly declare external normalization state")
    if normalization.get("mode") != contract.get("transform", {}).get("action_normalization"):
        raise ValueError("normalization metadata conflicts with policy contract")
    if normalization.get("statistics_used") != (normalization.get("mode") is not None):
        raise ValueError("normalization statistics usage conflicts with configured mode")
    statistics = _portable_path(export_dir, normalization.get("statistics"), "dataset.normalization.statistics")
    if statistics != metadata_paths["stats.json"]:
        raise ValueError("normalization statistics path conflicts with packaged dataset stats")

    selected_vae = bundled_vae
    if vae_override is not None:
        selected_vae = vae_override.expanduser().resolve()
        if not selected_vae.is_file():
            raise ValueError(f"VAE override is missing: {selected_vae}")
        if _sha256(selected_vae) != manifest["vae"]["sha256"]:
            raise ValueError("VAE override conflicts with the VAE used by the exported policy")
    return ResolvedExportRuntime(
        training_config=training_config,
        dataset_root=dataset_root,
        dataset_policy=metadata_paths["cosmos_policy.json"],
        dataset_info=metadata_paths["info.json"],
        dataset_stats=metadata_paths["stats.json"],
        policy_contract=contract_path,
        vae=selected_vae,
        contract=contract,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--framework-commit", required=True)
    parser.add_argument("--dataset-provenance", type=Path, required=True)
    args = parser.parse_args()
    package_export(
        args.export_dir,
        args.training_config,
        args.dataset_root,
        args.vae,
        framework_commit=args.framework_commit,
        dataset_provenance=args.dataset_provenance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
