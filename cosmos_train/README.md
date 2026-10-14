# Cosmos Train

`cosmos-train` is the Viam custom-training path for Cosmos3-Edge action policies.
A Viam custom training job runs `model/training.py`, which unpacks Isaac episode
zips, converts them into a DROID-layout LeRobot dataset, bootstraps the pinned
NVIDIA cosmos-framework (`cosmos_train/framework_setup.py`), trains Cosmos3-Edge
on every visible GPU, and writes a portable export into the job's model output
directory.

## Viam custom training

Episodes live in a Viam dataset as one `.zip` per native episode directory
(`<episode_id>/{manifest.json,samples.parquet,fixed.mp4,wrist.mp4}`). The training
job runs `model/training.py`, which unpacks them, runs
`convert_native` → `prepare_droid` → `policy_contract`, bootstraps the pinned
cosmos-framework, trains Cosmos3-Edge on every visible GPU, and writes the
portable export into the job's model output directory. The only external
download is the public `nvidia/Cosmos3-Edge` base checkpoint and the Wan2.2 VAE,
both anonymous.

```sh
uv build --sdist                                    # dist/cosmos_train-<version>.tar.gz
viam training-script upload --path dist/cosmos_train-<version>.tar.gz \
    --org-id <org-id> --script-name cosmos-edge-lite6 --framework pytorch --version <version>
viam train submit custom from-registry --dataset-id <dataset-id> --org-id <org-id> \
    --model-name cosmos-edge-lite6 --script-name <namespace>:cosmos-edge-lite6 \
    --version <version> --container-version cosmos-train:cuda13-baked \
    --args max_iter=2500,dataset_id=<dataset-id>
```

Custom args (all optional, passed via `--args key=value,...`; Viam translates
each pair to `--key=value`).

Dataset / stage selection:

| arg            | default               | notes                                                                                              |
| -------------- | --------------------- | -------------------------------------------------------------------------------------------------- |
| `task`         | from `convert_native` | task string override                                                                               |
| `action_space` | `joint_absolute`      | or `ee_relative`                                                                                   |
| `outcomes`     | `success`             | subset of `success failure incomplete`; space-separated and quoted since `--args` splits on commas |
| `dataset_id`   |                       | recorded in provenance                                                                             |
| `stage`        | `train`               | `prepare` stops after dataset preparation (CPU-only)                                               |

Training schedule:

| arg            | default               | notes                                                                                                                                                                                      |
| -------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `max_iter`     | `20`                  | total training iterations; the default is a smoke test only, set it for real training (see Tuning)                                                                                         |
| `warmup_steps` | `ceil(max_iter / 20)` | linear LR warmup length                                                                                                                                                                    |
| `save_iter`    | _final only_          | also save every N-th checkpoint to disk; only the final one is exported. The run trains on every episode (no validation loss), so keep your own held-out episodes to check for overfitting |

Optimization / dataloader:

| arg                   | default  | notes                                                                                                                                                      |
| --------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_lr`            | `2.5e-5` | base learning rate (`optimizer.lr`); tuned for a batch of 64–128                                                                                           |
| `train_lr_end_factor` | `0.2`    | final LR as a fraction of peak (`scheduler.f_min`); linear decay after warmup                                                                              |
| `grad_accum`          | `2`      | gradient accumulation steps; set it so the batch is 64–128 (see Tuning)                                                                                    |
| `num_workers`         | `4`      | dataloader workers per GPU; each batch mixes samples from at most this many episodes, so up to `samples_per_gpu` helps. More workers need more CPU and RAM |
| `samples_per_gpu`     | `8`      | samples per GPU per step; lower it and raise `grad_accum` if GPU memory runs out                                                                           |

The frame rate comes from the dataset; there is no arg for it.

Tuning:

- **Batch:** `batch = samples_per_gpu × gpus × grad_accum`. Aim for 64–128 (both trained well with the default LR): 8 GPUs → `grad_accum` 1–2, 4 GPUs → 2–4, 2 GPUs → 4–8.
- **Length:** pick `max_iter` for about 3–6 passes over the data: `max_iter ≈ passes × windows / batch`, where `windows ≈ total frames − 32 × episodes`. For reference, 1500 steps at batch 64 on ~100 episodes and 4000 steps at batch 128 on ~700 episodes both worked; a few hundred steps does not give a usable model.
- **Out of GPU memory:** halve `samples_per_gpu` and double `grad_accum` to keep the same batch. Activation checkpointing is already on. For reference, 8 samples per GPU with checkpointing off peaked at ~65 GB per GPU on 8 GPUs and ~80 GB on 2 GPUs; fewer GPUs means more memory per GPU.

The container is `docker/viam-training.Dockerfile`: `nvidia/cuda:13.0.3-base-ubuntu24.04`
plus git/ffmpeg/uv with the pinned cosmos-framework checkout and its venv baked in
(`COSMOS_FRAMEWORK_ROOT`), registered as `cosmos-train:cuda13-baked` (and
`cosmos-train:cuda13` without the venv, which bootstraps it at job start). The stock
`pytorch:26.06` container does not work: it ships no FFmpeg libraries, so torchcodec
cannot load in the dataloader.

Local check without a GPU. The `prepare` stage unpacks the dataset zips,
runs `convert_native` → `prepare_droid` → `policy_contract` to check the shape of the data.

```sh
viam dataset export --destination ds-export --dataset-id <dataset-id>
uv sync
uv run python -m model.training \
    --dataset_file ds-export/dataset.jsonl \
    --model_output_directory out \
    --stage prepare
```
