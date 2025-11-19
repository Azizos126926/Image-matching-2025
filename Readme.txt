Image Matching Challenge 2025

Final Rank: 140 / 950
Private Score (Unsubmitted): 38 → Would have ranked Top-50
This repository contains the code for my best official submission.

🧭 Overview

This repository contains my solution to the Image Matching Challenge 2025, which requires:

Grouping images into 3D scene clusters

Reconstructing each scene by predicting camera poses

Handling visually similar scenes and outlier images

Producing a single submission.csv with clusters + poses

The challenge extends the 2024 edition by adding scene partitioning, making robust clustering as important as accurate pose estimation.

🏁 Competition Summary

Start: April 1, 2025

Final Submission: June 2, 2025

Participants: 950+

Format: Kaggle Code Competition (9-hour notebook limit, no internet)

🧩 Task Description

You are given mixed, unlabeled images for each dataset. The tasks:

Assign each image to a scene cluster

Optionally mark images as outliers

Predict camera rotation (3×3) and translation (3D)

Use nan for poses when images cannot be registered

Example row:

dataset1,cluster1,image1.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3


Cluster names can be arbitrary (e.g., cluster1, scene_002, etc.).

🧮 Evaluation

Each dataset is scored using:

mAA (mean Average Accuracy)
Measures recall based on registered camera centers.

Clustering Score
Measures cluster purity (precision).

Final Score
Harmonic mean of the two (equivalent to an F1-style metric).

Scenes are assigned to clusters via greedy matching to maximize mAA.

📁 Repository Structure

(Adjust these names if needed)

src/               # Matching, clustering, registration
notebooks/         # Kaggle submission notebooks
models/            # Pretrained weights (if used)
utils/             # Helpers for geometry, IO, metrics
submission/        # Example output files
README.md

⚙️ Method Overview

Feature extraction and matching

Clustering via geometric + descriptor-space consistency

Outlier detection using multi-stage filtering

Pose estimation using Scene-wise SfM

Safe fallback: cluster assignment with nan pose when registration fails

The repository contains the version of this pipeline that achieved my official rank of 140/950.
My later private version reached a 38 score (Top-50 range) but was not submitted in time.

📤 Kaggle Submission Requirements

Notebook runtime ≤ 9 hours (CPU or GPU)

No internet access

Public external data & pretrained models allowed

Output must be named submission.csv

📚 Scientific Bibliography

Structure-from-Motion & Geometry

Snavely, Seitz, Szeliski — Photo Tourism, SIGGRAPH 2006

Hartley & Zisserman — Multiple View Geometry, 2004

Schönberger & Frahm — SfM Revisited, CVPR 2016

Features & Matching

Lowe — SIFT, IJCV 2004

Revaud et al. — R2D2, NeurIPS 2019

Sarlin et al. — LightGlue, ICCV 2023

Rocco et al. — SuperGlue, CVPR 2020

Pose Estimation

Kneip et al. — P3P Parameterization, CVPR 2011

Gao et al. — P3P Classification, PAMI 2003

Clustering & Outliers

Ester et al. — DBSCAN, 1996

Huber — Robust Statistics, 1981
