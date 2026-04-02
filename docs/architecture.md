# Architecture

## End-to-end flow

```text
competition csv
    │
    ▼
dataset loader
    │
    ▼
image shortlist generation
(exhaustive or DINOv2)
    │
    ▼
ALIKED keypoint extraction
(optional rotation-aware augmentation)
    │
    ▼
LightGlue matching
    │
    ▼
COLMAP database import
    │
    ▼
pycolmap exhaustive verification + incremental mapping
    │
    ▼
predicted cluster / pose assignment
    │
    ▼
submission.csv
```

## Module responsibilities

### `config.py`
Defines the typed experiment configuration and YAML parsing.

### `datasets.py`
Loads `sample_submission.csv` or `train_labels.csv` and converts rows into prediction objects.

### `features/global_descriptor.py`
Computes DINOv2 global descriptors used for image-pair shortlisting.

### `features/local.py`
Extracts ALIKED local keypoints and descriptors, with optional rotation-aware augmentation.

### `matching/pairs.py`
Implements exhaustive pairing and DINOv2-based shortlist selection.

### `matching/lightglue.py`
Reads HDF5 feature files and writes matched index pairs into `matches.h5`.

### `reconstruction/colmap.py`
Contains the reconstruction adapter that imports features and matches into COLMAP-compatible storage and launches pycolmap.

### `pipeline.py`
Coordinates end-to-end processing across all datasets.

### `submission.py`
Serializes predictions to the exact Kaggle submission format.

### `scoring.py`
Optionally computes the competition metric on the train split when the metric package is present.
