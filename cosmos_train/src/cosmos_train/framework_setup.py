"""Prepare a pinned NVIDIA cosmos-framework checkout for Cosmos3-Edge action-policy training.

Used by model/training.py (Viam custom training). The patches and SFT config under
cosmos_train/assets/ are shipped with the package so the pip-installed sdist is self-contained.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

FRAMEWORK_REPO = "https://github.com/NVIDIA/cosmos-framework.git"
FRAMEWORK_REV = "efba2076c80648663a89a7fdbc92eb262fae39ac"
BASE_CHECKPOINT_REPO = "nvidia/Cosmos3-Edge"
WAN_VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B"
WAN_VAE_FILE = "Wan2.2_VAE.pth"
EDGE_POLICY_METADATA = {"action_chunk_size": 32, "domain_name": "droid_lerobot"}
ASSETS = Path(__file__).resolve().parent / "assets"
JOINT_COUNT_PATCH = ASSETS / "cosmos_joint_count.patch"
LOSS_PARTS_PATCH = ASSETS / "cosmos_loss_parts.patch"
SFT_TOML = ASSETS / "action_policy_edge.toml"
RUN_RELATIVE_DIR = Path("cosmos3_action/viam_train/lite6_native_joint_action_policy_edge")


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(list(command), cwd=cwd, env=dict(env) if env else None, check=True)


def _venv_python(root: Path) -> Path:
    return root / ".venv/bin/python"


def clone_framework(root: Path, *, revision: str = FRAMEWORK_REV) -> None:
    """Clone cosmos-framework at the pinned revision and apply the baked-in patches."""
    _run(["git", "clone", FRAMEWORK_REPO, str(root)])
    _run(["git", "-C", str(root), "checkout", revision])
    _run(["git", "-C", str(root), "apply", str(JOINT_COUNT_PATCH)])
    _run(["git", "-C", str(root), "apply", str(LOSS_PARTS_PATCH)])


def sync_framework(root: Path) -> None:
    _run(["uv", "sync", "--all-extras", "--group=cu130-train"], cwd=root)


def patch_framework_for_edge(root: Path, *, fps: float) -> None:
    """Derive the Edge post-training config from the nano one and pin the Edge export metadata.

    ``fps`` is the dataset's frame rate; the exported model records it as its
    conditioning rate, so it must match what training used.
    """
    package = root / "cosmos_framework"
    source = (
        package
        / "configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py"
    )
    target = source.with_name("action_policy_droid_edge.py")
    text = source.read_text()
    text = text.replace("action_policy_droid_nano", "action_policy_droid_edge")
    text = text.replace(
        "from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG",
        "from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG",
    )
    text = text.replace("NANO_MODEL_CONFIG", "EDGE_MODEL_CONFIG")
    text = text.replace(
        '{"override /checkpoint": "s3"}', '{"override /checkpoint": "local"}'
    )
    target.write_text(text)

    config = package / "configs/base/config.py"
    needle = (
        "    import cosmos_framework.configs.base.experiment.action.posttrain_config."
        "action_policy_droid_nano  # noqa: F401\n"
    )
    replacement = needle + needle.replace(
        "action_policy_droid_nano", "action_policy_droid_edge"
    )
    text = config.read_text()
    if replacement in text:
        pass
    elif needle in text:
        config.write_text(text.replace(needle, replacement, 1))
    else:
        raise RuntimeError("pinned framework config import changed")

    exporter = package / "scripts/export_model.py"
    needle = """    edge_policy_metadata = (
        _build_edge_policy_metadata(checkpoint_args.load_config())
        if is_edge and model_dict["config"].get("action_gen")
        else None
    )
"""
    metadata = {**EDGE_POLICY_METADATA, "conditioning_fps": float(fps)}
    replacement = f"""    edge_policy_metadata = (
        {json.dumps(metadata)}
        if is_edge and model_dict["config"].get("action_gen")
        else None
    )
"""
    text = exporter.read_text()
    if replacement in text:
        pass
    elif needle in text:
        exporter.write_text(text.replace(needle, replacement, 1))
    else:
        raise RuntimeError("pinned Edge policy exporter changed")

    # Train on every episode. The DROID loader holds out 3% as a validation split that this
    # recipe never runs; we keep our own hold-out out of the dataset.
    sft_dataset = package / "data/generator/action/datasets/action_sft_dataset.py"
    needle = "    dataset: Dataset = DROIDLeRobotDataset(root=root, **shard_kwargs)\n"
    replacement = (
        '    dataset: Dataset = DROIDLeRobotDataset(root=root, split="full", **shard_kwargs)\n'
    )
    text = sft_dataset.read_text()
    if replacement in text:
        pass
    elif needle in text:
        sft_dataset.write_text(text.replace(needle, replacement, 1))
    else:
        raise RuntimeError("pinned DROID dataset factory changed")


def download_checkpoints() -> tuple[Path, Path]:
    """Fetch the ungated base checkpoint and VAE anonymously into the HF cache (HF_HOME).

    The framework re-fetches both through its own hf_cli at convert/train time, so sharing the
    cache keeps one copy on disk. Returns (hf_checkpoint_snapshot_dir, vae_path).
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    vae = Path(hf_hub_download(WAN_VAE_REPO, WAN_VAE_FILE, token=False))
    checkpoint = Path(
        snapshot_download(
            BASE_CHECKPOINT_REPO,
            ignore_patterns=["assets/*", "images/*", "*.md"],
            token=False,
        )
    )
    return checkpoint, vae


def convert_to_dcp(framework_root: Path, hf_checkpoint: Path, output: Path) -> Path:
    _run(
        [
            str(_venv_python(framework_root)),
            "-m",
            "cosmos_framework.scripts.convert_model_to_dcp",
            "--checkpoint-path",
            str(hf_checkpoint),
            "-o",
            str(output),
        ],
        cwd=framework_root,
    )
    for name in ("model/__0_0.distcp", "model/.metadata"):
        if not (output / name).is_file() or (output / name).stat().st_size == 0:
            raise RuntimeError(f"DCP conversion produced no {name}")
    return output


VERTEX_DRIVER_DIRS = ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib")


def driver_library_path(framework_root: Path) -> str:
    """LD_LIBRARY_PATH for framework subprocesses: the Vertex driver mount plus the venv's pinned
    nvidia-* wheel libraries, and nothing inherited.

    Vertex mounts the host driver under /usr/local/nvidia without running ldconfig. Inherited
    entries are dropped on purpose: on NGC images they carry the CUDA toolkit and the system
    torch, which would shadow the venv's own cu130 libraries. torch preloads most nvidia-* wheel
    libraries itself, but torchcodec dlopens NPP by soname and only finds it through the loader
    path.
    """
    venv_nvidia = sorted(
        str(path)
        for path in framework_root.glob(".venv/lib/python3*/site-packages/nvidia/*/lib")
    )
    return os.pathsep.join([*VERTEX_DRIVER_DIRS, *venv_nvidia])


def check_framework_root(root: Path, *, revision: str = FRAMEWORK_REV) -> None:
    """Fail before any download if a prebuilt checkout is unusable."""
    if not _venv_python(root).is_file():
        raise RuntimeError(
            f"{root} has no .venv; run uv sync or unset COSMOS_FRAMEWORK_ROOT"
        )
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != revision:
        raise RuntimeError(f"{root} is at {head}, expected cosmos-framework {revision}")


def gpu_count(framework_root: Path) -> int:
    result = subprocess.run(
        [
            str(_venv_python(framework_root)),
            "-c",
            "import torch; print(torch.cuda.device_count())",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    count = int(result.stdout.strip())
    if count < 1:
        for command in (["nvidia-smi", "-L"], ["ls", "-la", *VERTEX_DRIVER_DIRS]):
            try:
                probe = subprocess.run(command, capture_output=True, text=True)
            except FileNotFoundError:
                print(f"+ {' '.join(command)}\n(not found)", flush=True)
                continue
            print(f"+ {' '.join(command)}\n{probe.stdout}{probe.stderr}", flush=True)
        print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}", flush=True)
    return count


def train_command(
    framework_root: Path,
    *,
    gpus: int,
    max_iter: int,
    warmup_steps: int,
    fps: float,
    save_iter: int | None = None,
    num_workers: int = 4,
    samples_per_gpu: int = 8,
    grad_accum: int | None = 2,
    train_lr: float | None = 2.5e-5,
    train_lr_end_factor: float | None = 0.2,
) -> list[str]:
    if max_iter <= 0 or warmup_steps <= 0:
        raise ValueError("max_iter and warmup_steps must be positive")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if save_iter is not None and save_iter <= 0:
        raise ValueError("save_iter must be positive")
    if grad_accum is not None and grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    if train_lr is not None and train_lr <= 0:
        raise ValueError("train_lr must be positive")
    if train_lr_end_factor is not None and not 0.0 <= train_lr_end_factor <= 1.0:
        raise ValueError("train_lr_end_factor must be between 0 and 1")
    checkpoint_save_iter = save_iter if save_iter is not None else max_iter
    args = [
        str(framework_root / ".venv/bin/torchrun"),
        f"--nproc_per_node={gpus}",
        "-m",
        "cosmos_framework.scripts.train",
        f"--sft-toml={SFT_TOML}",
        "--",
        f"trainer.max_iter={max_iter}",
        f"checkpoint.save_iter={checkpoint_save_iter}",
        f"scheduler.cycle_lengths=[{max_iter}]",
        f"scheduler.warm_up_steps=[{warmup_steps}]",
        f"dataloader_train.dataloader.num_workers={num_workers}",
        f"dataloader_train.max_samples_per_batch={samples_per_gpu}",
        # batch_size=1: each worker hands over one sample at a time, so the packer fills each
        # batch from several workers (episodes) instead of consecutive windows of one episode.
        "dataloader_train.dataloader.batch_size=1",
        "dataloader_train.dataloader.persistent_workers=False",
        f"dataloader_train.dataloader.datasets.droid.dataset.fps={float(fps)}",
        "dataloader_train.dataloader.datasets.droid.dataset.action_space=joint_pos",
        "dataloader_train.dataloader.datasets.droid.dataset.joint_count=6",
        "dataloader_train.dataloader.datasets.droid.dataset.use_state=True",
        "dataloader_train.dataloader.datasets.droid.dataset.use_image_augmentation=False",
    ]
    if grad_accum is not None:
        args.append(f"trainer.grad_accum_iter={grad_accum}")
    if train_lr is not None:
        args.append(f"optimizer.lr={train_lr}")
    if train_lr_end_factor is not None:
        args.append(f"scheduler.f_min=[{train_lr_end_factor}]")
    return args


def training_environment(
    *, base_checkpoint: Path, vae: Path, droid_root: Path, output_root: Path
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "BASE_CHECKPOINT_PATH": str(base_checkpoint),
            "WAN_VAE_PATH": str(vae),
            "DROID_ROOT": str(droid_root),
            "IMAGINAIRE_OUTPUT_ROOT": str(output_root),
            "PYTHONPATH": os.pathsep.join(
                p for p in (".", env.get("PYTHONPATH", "")) if p
            ),
            # Every dataloader worker tokenizes its own prompts, and the Rust tokenizer builds
            # a thread pool per process sized to the CPU count. With enough GPUs x workers that
            # exhausts the container's thread limit ("ThreadPoolBuildError ... Resource
            # temporarily unavailable"). One prompt at a time needs no pool.
            "TOKENIZERS_PARALLELISM": "false",
            "RAYON_NUM_THREADS": "1",
        }
    )
    return env


def default_warmup(max_iter: int) -> int:
    return (max_iter + 19) // 20


def export_command(
    framework_root: Path, *, run_dir: Path, export_dir: Path
) -> list[str]:
    checkpoint_name = (
        (run_dir / "checkpoints/latest_checkpoint.txt").read_text().strip()
    )
    return [
        str(_venv_python(framework_root)),
        "-m",
        "cosmos_framework.scripts.export_model",
        "--checkpoint-path",
        str(run_dir / "checkpoints" / checkpoint_name),
        "--config-file",
        str(run_dir / "config.yaml"),
        "-o",
        str(export_dir),
    ]
