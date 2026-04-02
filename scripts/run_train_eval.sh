#!/usr/bin/env bash
set -euo pipefail

python -m imc2025 run   --config configs/kaggle_offline.yaml   --submission-path outputs/train_submission.csv

python -m imc2025 score   --config configs/kaggle_offline.yaml   --submission-path outputs/train_submission.csv
