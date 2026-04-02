#!/usr/bin/env bash
set -euo pipefail

echo "[imc2025] Installing offline wheels..."
pip install --no-index /kaggle/input/imc2024-packages-lightglue-rerun-kornia/* --no-deps

echo "[imc2025] Preparing torch checkpoint cache..."
mkdir -p /root/.cache/torch/hub/checkpoints

echo "[imc2025] Copying ALIKED + LightGlue checkpoints..."
cp /kaggle/input/aliked/pytorch/aliked-n16/1/aliked-n16.pth /root/.cache/torch/hub/checkpoints/
cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/
cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth    /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv-pth

echo "[imc2025] Done."
