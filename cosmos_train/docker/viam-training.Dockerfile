# Training container for Viam custom training running `python -m model.training`.
# Bakes the pinned cosmos-framework checkout and its uv venv so jobs don't rebuild
# them; the base checkpoint is downloaded at job start.
#
# Build:
#   docker buildx build --platform linux/amd64 -f docker/viam-training.Dockerfile \
#     -t <registry>/cosmos-train:<tag> --push .
FROM nvidia/cuda:13.0.3-base-ubuntu24.04

ARG FRAMEWORK_REV=efba2076c80648663a89a7fdbc92eb262fae39ac
ENV COSMOS_FRAMEWORK_ROOT=/opt/cosmos-framework \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    PIP_BREAK_SYSTEM_PACKAGES=1

# build-essential + linux-libc-dev: a few lerobot deps build from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-dev build-essential linux-libc-dev git ffmpeg libx11-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.12.23

# Patches are applied after `uv sync` so the 12 GB sync layer stays cacheable.
RUN git clone https://github.com/NVIDIA/cosmos-framework.git "$COSMOS_FRAMEWORK_ROOT" \
    && git -C "$COSMOS_FRAMEWORK_ROOT" checkout "$FRAMEWORK_REV"
RUN --mount=type=cache,target=/root/.cache/uv \
    cd "$COSMOS_FRAMEWORK_ROOT" && uv sync --all-extras --group=cu130-train
COPY src/cosmos_train/assets/cosmos_joint_count.patch /tmp/cosmos_joint_count.patch
COPY src/cosmos_train/assets/cosmos_loss_parts.patch /tmp/cosmos_loss_parts.patch
RUN git -C "$COSMOS_FRAMEWORK_ROOT" apply /tmp/cosmos_joint_count.patch \
    && git -C "$COSMOS_FRAMEWORK_ROOT" apply /tmp/cosmos_loss_parts.patch \
    && rm /tmp/cosmos_joint_count.patch /tmp/cosmos_loss_parts.patch

WORKDIR /workspace
