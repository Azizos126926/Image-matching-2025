# Image Matching Challenge 2025 — Modular Project
> This is a one of the most prestigious computer vision competition happening annually.
> A clean, production-style refactor of the original Kaggle notebook into an organized ML project.
##  🥈 Silver Medal solution Kaggle
> I managed to deliver a solution that would rank amon top 50 (silver Kaggle medal) but my official solution that uploaded was ranked top 141.
## Highlights

- **Notebook → project**: the original exploratory notebook is preserved under `notebooks/`, while the main pipeline now lives in a typed Python package.
- **Clear module boundaries**: dataset I/O, shortlist generation, local features, matching, reconstruction, scoring, and submission writing are separated.
- **Kaggle-ready**: the project keeps the same offline-first assumptions as the original competition workflow.
- **Engineer-friendly structure**: config-driven runs, CLI entrypoints, tests, reusable scripts, and documentation.

## Original solution summary

This project preserves the original solution logic:

1. **DINOv2 shortlisting** for scalable image-pair proposal.
2. **ALIKED** local feature extraction.
3. **Rotation-aware ALIKED extraction** to recover matches in orientation-sensitive scenes.
4. **LightGlue** descriptor matching.
5. **pycolmap** reconstruction for geometric verification and pose recovery.
6. **CSV submission export** for Kaggle.

The original README described the solution as an IMC 2025 pipeline with image grouping, feature extraction and matching, geometric verification, pose estimation, outlier handling, and submission generation. This refactor keeps that workflow intact while improving readability, extensibility, and maintainability.

---

## Project layout

```text
imc2025_project/
├── configs/
│   ├── default.yaml
│   └── kaggle_offline.yaml
├── docs/
│   └── architecture.md
├── notebooks/
│   └── original_imc_2025_submission.ipynb
├── scripts/
│   ├── kaggle_bootstrap.sh
│   ├── run_test_submission.sh
│   └── run_train_eval.sh
├── src/imc2025/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── datasets.py
│   ├── pipeline.py
│   ├── scoring.py
│   ├── submission.py
│   ├── types.py
│   ├── features/
│   │   ├── global_descriptor.py
│   │   └── local.py
│   ├── matching/
│   │   ├── lightglue.py
│   │   └── pairs.py
│   ├── reconstruction/
│   │   └── colmap.py
│   └── utils/
│       ├── image.py
│       └── logging.py
├── tests/
│   ├── test_config.py
│   └── test_submission.py
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Quick start

### 1) Install

```bash
python -m pip install -e .[dev]
```

### 2) Prepare Kaggle offline runtime

```bash
bash scripts/kaggle_bootstrap.sh
```

### 3) Run on the hidden test split

```bash
python -m imc2025 run   --config configs/kaggle_offline.yaml   --submission-path outputs/submission.csv
```

### 4) Score locally on train labels

```bash
python -m imc2025 score   --config configs/kaggle_offline.yaml   --submission-path outputs/submission.csv
```

---

## Design choices

### 1. Typed configuration instead of notebook globals
The original notebook used top-level variables such as `is_train`, `data_dir`, and hard-coded thresholds. Those are now moved into YAML config files plus typed dataclasses in `config.py`.

### 2. Small, focused modules
Instead of one long notebook cell, each stage is isolated:
- `datasets.py` handles sample loading.
- `matching/pairs.py` handles exhaustive and descriptor-based pair selection.
- `features/local.py` handles ALIKED extraction.
- `matching/lightglue.py` handles descriptor matching.
- `reconstruction/colmap.py` handles COLMAP import and incremental mapping.
- `submission.py` handles Kaggle CSV formatting.

### 3. Clear Kaggle/runtime boundary
Competition-specific bootstrapping lives in `scripts/`, while reusable Python logic stays in `src/imc2025/`.

### 4. Notebook is archived, not deleted
The original notebook remains in `notebooks/` for traceability.

---

## Notes on compatibility

This refactor intentionally keeps the original competition assumptions:
- offline wheel installation,
- local model checkpoints,
- Kaggle dataset folder conventions,
- pycolmap-based reconstruction,
- optional competition utility modules for scoring/import helpers.

If your runtime differs from Kaggle, update the YAML config paths and bootstrap script first.

---

## Scientific bibliography

1. Zhao, X., Wu, X., Chen, W., Chen, P. C. Y., Xu, Q., & Li, Z. **ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation.** arXiv:2304.03608.
2. Lindenberger, P., Sarlin, P.-E., & Pollefeys, M. **LightGlue: Local Feature Matching at Light Speed.** ICCV 2023 / arXiv:2306.13643.
3. Oquab, M., Darcet, T., Moutakanni, T., et al. **DINOv2: Learning Robust Visual Features without Supervision.** arXiv:2304.07193.
4. Image Matching Challenge 2025, Kaggle competition page.

---

## What changed from the notebook

- Removed long monolithic execution flow.
- Added CLI commands for `run` and `score`.
- Added testable utilities for config loading and CSV formatting.
- Added explicit logging and cleaner error messages.
- Separated competition runtime setup from algorithmic code.
- Preserved the original notebook for reproducibility.

## Suggested next improvements

- Add Hydra or Pydantic Settings if you want more advanced experiment management.
- Add caching for DINO global descriptors per dataset.
- Add integration tests around a tiny synthetic dataset.
- Add experiment tracking (Weights & Biases or MLflow) if you want leaderboard-oriented sweeps.
