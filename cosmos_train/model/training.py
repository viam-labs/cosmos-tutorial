"""Viam custom training script: Isaac episode zips -> Cosmos3-Edge action policy export.

Invoked by the Viam training container as
``python -m model.training --dataset_file=... --model_output_directory=... [custom args]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from cosmos_train import framework_setup
from cosmos_train.convert_native import _read_manifest, convert
from cosmos_train.export_bundle import package_export
from cosmos_train.policy_contract import build_runtime_contract
from cosmos_train.prepare_droid import DROID_ROOT_NAME, prepare

STAGES = ("prepare", "train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_file", dest="dataset_file", type=Path, required=True)
    parser.add_argument(
        "--model_output_directory", dest="model_dir", type=Path, required=True
    )
    parser.add_argument(
        "--task", default=None, help="task string; defaults to convert_native's"
    )
    parser.add_argument(
        "--action_space",
        choices=("joint_absolute", "ee_relative"),
        default="joint_absolute",
    )
    parser.add_argument(
        "--outcomes",
        default="success",
        help="space- or comma-separated subset of success,failure,incomplete",
    )
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument(
        "--save_iter",
        type=int,
        default=None,
        help="save checkpoint every this many steps; defaults to only the final one",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="dataloader workers per GPU",
    )
    parser.add_argument(
        "--samples_per_gpu",
        type=int,
        default=8,
        help="samples per GPU per step; lower it (and raise grad_accum) if GPU memory runs out",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=2,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        "--train_lr",
        type=float,
        default=2.5e-5,
        help="base learning rate",
    )
    parser.add_argument(
        "--train_lr_end_factor",
        type=float,
        default=0.2,
        help="final learning rate as a fraction of the peak (0-1)",
    )
    parser.add_argument(
        "--dataset_id", default="", help="Viam dataset ID, recorded in provenance"
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="train",
        help="prepare stops before the GPU work",
    )
    args = parser.parse_args(argv)
    args.outcomes = frozenset(args.outcomes.replace(",", " ").split())
    if not args.outcomes:
        parser.error("--outcomes must name at least one outcome")
    if args.max_iter <= 0:
        parser.error("--max_iter must be positive")
    if args.warmup_steps is None:
        args.warmup_steps = framework_setup.default_warmup(args.max_iter)
    if args.warmup_steps <= 0:
        parser.error("--warmup_steps must be positive")
    if args.save_iter is not None and args.save_iter <= 0:
        parser.error("--save_iter must be positive")
    if args.num_workers <= 0:
        parser.error("--num_workers must be positive")
    if args.samples_per_gpu <= 0:
        parser.error("--samples_per_gpu must be positive")
    if args.grad_accum is not None and args.grad_accum <= 0:
        parser.error("--grad_accum must be positive")
    if args.train_lr is not None and args.train_lr <= 0:
        parser.error("--train_lr must be positive")
    if args.train_lr_end_factor is not None and not 0.0 <= args.train_lr_end_factor <= 1.0:
        parser.error("--train_lr_end_factor must be between 0 and 1")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> list[str]:
    """Extract one episode zip; returns the top-level entries."""
    destination.mkdir(parents=True, exist_ok=True)
    tops: set[str] = set()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            parts = Path(name).parts
            if (
                not parts
                or parts[0] in ("", ".", "..")
                or ".." in parts
                or name.startswith("/")
                or name.startswith("\\")
            ):
                raise ValueError(f"{archive}: unsafe member {name!r}")
            # Zip encodes symlinks via the Unix mode bits in external_attr (create_system == 3).
            if info.create_system == 3 and stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"{archive}: links are not allowed ({name})")
            tops.add(parts[0])
        zf.extractall(destination)
    return sorted(tops)


def _resolve_archive(image_path: Path, dataset_root: Path) -> Path:
    """Viam cloud JSONL may use absolute paths; `viam dataset export` writes
    `<export-dir>/data/<file>`, i.e. relative to the export's parent, so also try it with the
    export-dir prefix stripped."""
    candidates = [image_path, dataset_root / image_path]
    if len(image_path.parts) > 1:
        candidates.append(dataset_root.joinpath(*image_path.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"episode archive not found: {image_path}")


def _suffixed_episode_id(
    episode_id: str, binary_data_id: str, archive_sha256: str, taken: set[str]
) -> str:
    """Suffix a colliding episode_id from the upload's identity, stable across reruns."""
    source = binary_data_id or archive_sha256
    digest = hashlib.sha256(source.encode()).hexdigest()
    # Lengthen the suffix only if a shorter one is already taken.
    for length in range(8, len(digest) + 1, 4):
        candidate = f"{episode_id}-{digest[:length]}"
        if candidate not in taken:
            return candidate
    raise ValueError(f"cannot derive a unique episode_id for {episode_id!r}")


def ingest_episodes(dataset_file: Path, episodes_root: Path) -> dict[str, object]:
    """Unpack every episode zip listed in the Viam JSONL into *episodes_root*/<episode_id>/.

    Separate capture runs can reuse episode IDs. An archive identical to one
    already listed is the same episode uploaded twice and is skipped. Every
    remaining episode whose ID collides is renamed (folder and manifest) with a
    suffix derived from its own binary_data_id, so IDs do not depend on the
    order of the JSONL; episodes without collisions keep their IDs.
    """
    episodes_root.mkdir(parents=True)
    dataset_root = dataset_file.parent
    staged: list[dict] = []
    by_sha256: dict[str, dict] = {}  # archive sha256 -> staged item
    # IDs are assigned only after every archive is read, so episodes stay
    # staged until the second pass moves them into place.
    with tempfile.TemporaryDirectory(dir=episodes_root.parent) as scratch:
        staging = Path(scratch)
        # Pass 1: unpack and validate each archive, skipping identical copies.
        with dataset_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                image_path = entry.get("image_path")
                if not isinstance(image_path, str) or not image_path:
                    raise ValueError(f"{dataset_file}:{line_number}: missing image_path")
                archive = _resolve_archive(Path(image_path), dataset_root)
                # Viam's cloud JSONL strips the upload's file_extension, so sniff rather than trust the name.
                if not zipfile.is_zipfile(archive):
                    print(f"skipping non-episode file {archive.name}", flush=True)
                    continue
                binary_data_id = str(entry.get("binary_data_id", ""))
                label = f"{archive.name} ({binary_data_id})" if binary_data_id else archive.name
                archive_sha256 = _sha256(archive)
                kept = by_sha256.get(archive_sha256)
                if kept is not None:
                    # Keep the copy with the smallest binary_data_id so the
                    # surviving upload, and any suffix derived from it, does
                    # not depend on JSONL order.
                    if binary_data_id < kept["binary_data_id"]:
                        skipped = kept["label"]
                        kept.update(
                            label=label,
                            file_name=archive.name,
                            binary_data_id=binary_data_id,
                        )
                    else:
                        skipped = label
                    print(
                        f"skipping {skipped}: identical to {kept['label']}",
                        flush=True,
                    )
                    continue
                destination = staging / str(len(staged))
                tops = _safe_extract(archive, destination)
                if len(tops) != 1:
                    raise ValueError(
                        f"{archive.name}: expected one episode directory, found {tops}"
                    )
                extracted = destination / tops[0]
                manifest_path = extracted / "manifest.json"
                if not manifest_path.is_file():
                    raise ValueError(
                        f"{archive.name}: episode directory lacks manifest.json"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                episode_id = manifest.get("episode_id") or tops[0]
                if not isinstance(episode_id, str) or not episode_id:
                    raise ValueError(f"{archive.name}: manifest.json has no episode_id")
                if Path(episode_id).name != episode_id:
                    raise ValueError(
                        f"{archive.name}: episode_id {episode_id!r} is not a plain name"
                    )
                item = {
                    "extracted": extracted,
                    "manifest": manifest,
                    "episode_id": episode_id,
                    "label": label,
                    "file_name": archive.name,
                    "binary_data_id": binary_data_id,
                    "sha256": archive_sha256,
                }
                staged.append(item)
                by_sha256[archive_sha256] = item

        # Pass 2: suffix every member of a colliding group, then move into place.
        counts: dict[str, int] = {}
        for item in staged:
            counts[item["episode_id"]] = counts.get(item["episode_id"], 0) + 1
        taken = {item["episode_id"] for item in staged}
        records: list[dict[str, str]] = []
        for item in staged:
            source_episode_id = episode_id = item["episode_id"]
            if counts[source_episode_id] > 1:
                episode_id = _suffixed_episode_id(
                    source_episode_id, item["binary_data_id"], item["sha256"], taken
                )
                taken.add(episode_id)
                item["manifest"]["episode_id"] = episode_id
                (item["extracted"] / "manifest.json").write_text(
                    json.dumps(item["manifest"], indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"renamed duplicate episode_id {source_episode_id!r} in "
                    f"{item['label']} to {episode_id!r}",
                    flush=True,
                )
            shutil.move(str(item["extracted"]), str(episodes_root / episode_id))
            record = {
                "episode_id": episode_id,
                "file_name": item["file_name"],
                "binary_data_id": item["binary_data_id"],
                "sha256": item["sha256"],
            }
            if episode_id != source_episode_id:
                record["source_episode_id"] = source_episode_id
            records.append(record)
    if not records:
        raise ValueError(f"{dataset_file} lists no episode zips")
    records.sort(key=lambda item: item["episode_id"])
    content = hashlib.sha256(
        json.dumps([(r["episode_id"], r["sha256"]) for r in records]).encode()
    ).hexdigest()
    return {"episodes": records, "content_sha256": content}


def _write_contract(directory: Path, contract: dict) -> None:
    (directory / "policy_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True)
    )


def _preflight_native_episodes(episodes_root: Path) -> None:
    """Validate every unpacked episode manifest before conversion or training starts."""
    failures: list[str] = []
    for episode_dir in sorted(episodes_root.iterdir()):
        try:
            _read_manifest(episode_dir / "manifest.json")
        except (ValueError, TypeError, OSError) as exc:
            failures.append(f"  - {episode_dir.name}: {exc}")
    if failures:
        raise ValueError(
            "dataset is not in the expected shape; refusing to convert or train. "
            "Re-capture the affected episodes with a schema-compatible writer "
            "or update the trainer. Issues:\n" + "\n".join(failures)
        )


def prepare_dataset(args: argparse.Namespace, work: Path) -> tuple[Path, Path, dict]:
    """Build the DROID-layout dataset; returns (droid_root, provenance_path, runtime_contract)."""
    native = work / "native/episodes"
    ingest = ingest_episodes(args.dataset_file, native)
    print(f"ingested {len(ingest['episodes'])} episodes", flush=True)

    _preflight_native_episodes(native)

    direct = work / "direct"
    summary = convert(
        native,
        direct,
        action_space=args.action_space,
        outcomes=args.outcomes,
        task=args.task,
    )
    print(json.dumps({"convert": summary}), flush=True)

    droid_root = work / "prepared" / DROID_ROOT_NAME
    summary = prepare(direct, droid_root)
    print(json.dumps({"prepare": summary}), flush=True)

    contract = build_runtime_contract(droid_root)
    provenance = {
        "schema_version": 1,
        "source": "viam-dataset",
        "dataset_file": str(args.dataset_file),
        "dataset_id": args.dataset_id,
        "task": args.task,
        "action_space": args.action_space,
        "outcomes": sorted(args.outcomes),
        **ingest,
    }
    provenance_path = droid_root / "meta/source_dataset.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    return droid_root, provenance_path, contract


def train_and_export(
    args: argparse.Namespace,
    work: Path,
    droid_root: Path,
    provenance_path: Path,
    contract: dict,
) -> None:
    # Everything below lands on the boot disk; keep caches under work so they are easy to drop.
    os.environ["UV_CACHE_DIR"] = str(work / ".uv-cache")
    os.environ["HF_HOME"] = str(work / ".hf-cache")
    # docker/viam-training.Dockerfile bakes the checkout + venv and sets COSMOS_FRAMEWORK_ROOT.
    prebuilt = os.environ.get("COSMOS_FRAMEWORK_ROOT")
    if prebuilt:
        framework_root = Path(prebuilt)
        framework_setup.check_framework_root(framework_root)
    else:
        framework_root = work / "cosmos-framework"
        framework_setup.clone_framework(framework_root)
        framework_setup.sync_framework(framework_root)
        shutil.rmtree(work / ".uv-cache", ignore_errors=True)
    # The dataset's own frame rate, as validated into the runtime contract.
    fps = contract["conditioning_fps"]
    framework_setup.patch_framework_for_edge(framework_root, fps=fps)
    os.environ["LD_LIBRARY_PATH"] = framework_setup.driver_library_path(framework_root)

    hf_checkpoint, vae = framework_setup.download_checkpoints()
    base_checkpoint = framework_setup.convert_to_dcp(
        framework_root, hf_checkpoint, work / "checkpoints/Cosmos3-Edge"
    )

    gpus = framework_setup.gpu_count(framework_root)
    if gpus < 1:
        raise RuntimeError("no CUDA devices visible")
    output_root = work / "output"
    env = framework_setup.training_environment(
        base_checkpoint=base_checkpoint,
        vae=vae,
        droid_root=droid_root,
        output_root=output_root,
    )
    command = framework_setup.train_command(
        framework_root,
        gpus=gpus,
        max_iter=args.max_iter,
        warmup_steps=args.warmup_steps,
        fps=fps,
        save_iter=args.save_iter,
        num_workers=args.num_workers,
        samples_per_gpu=args.samples_per_gpu,
        grad_accum=args.grad_accum,
        train_lr=args.train_lr,
        train_lr_end_factor=args.train_lr_end_factor,
    )
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=framework_root, env=env, check=True)

    run_dir = output_root / framework_setup.RUN_RELATIVE_DIR
    export_dir = args.model_dir.resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    command = framework_setup.export_command(
        framework_root, run_dir=run_dir, export_dir=export_dir
    )
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=framework_root, env=env, check=True)

    # After export_model, which owns export_dir.
    _write_contract(export_dir, contract)
    package_export(
        export_dir,
        run_dir / "config.yaml",
        droid_root,
        vae,
        framework_commit=framework_setup.FRAMEWORK_REV,
        dataset_provenance=provenance_path,
    )
    print(
        json.dumps(
            {
                "export": str(export_dir),
                "files": sorted(p.name for p in export_dir.iterdir()),
            }
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(
            prefix="cosmos-train-",
            dir="/workspace" if Path("/workspace").is_dir() else None,
        )
    )
    print(json.dumps({"work_dir": str(work), "stage": args.stage}), flush=True)

    droid_root, provenance_path, contract = prepare_dataset(args, work)
    if args.stage == "prepare":
        _write_contract(args.model_dir, contract)
        shutil.copyfile(provenance_path, args.model_dir / "source_dataset.json")
        return 0
    train_and_export(args, work, droid_root, provenance_path, contract)
    return 0


if __name__ == "__main__":
    sys.exit(main())
