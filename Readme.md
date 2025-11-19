## Image Matching Challenge 2025 — Solution
🏆 Final Rank: 140 / 950
⚡ Private (Unsubmitted) Score: 38 → Would have placed Top-50

This repository contains the exact code used for my best official submission in the Image Matching Challenge 2025.

## 📋 Overview
A complete pipeline for the Image Matching Challenge 2025, featuring:

🖼️ Image grouping (scene clustering)

🔍 Feature extraction and feature matching

✅ Geometric verification

📐 Camera pose estimation

🎯 Outlier handling

📊 Submission CSV generation

The final model integrates ALIKED local features with LightGlue matching, optimized for Kaggle's constraints (no internet, 9-hour runtime, no external downloads).

## 🏗️ Pipeline Architecture
1. Dependencies & Model Weights
The notebook installs all feature-matching dependencies using offline wheels:
ALIKED feature extractor
LightGlue matcher
rerun, kornia, and other IMC-compatible packages
Pretrained weights (.pth and .pt) are copied to:
text
/root/.cache/torch/hub/checkpoints/
3. Dataset Loading
Parses competition metadata to load:

Train sets

Validation sets (from training split)

Test set (images only; no poses)

Unified interface structure:

python
samples[dataset] = [Prediction(...)]
3. Feature Extraction
Uses ALIKED for robust local feature detection:

Excellent for low-texture regions

Strong wide-baseline matching

Efficient for Kaggle runtime limits

4. Feature Matching
LightGlue provides adaptive matching:

Avoids brute-force descriptor comparisons

Robust to repetitive structures and viewpoint changes

High-quality putative matches

5. Geometric Verification
Matches filtered using:

RANSAC

Essential matrix checks

Custom thresholding based on match count, spatial distribution, and confidence

6. Scene Clustering
Images grouped via:

Connectivity in verified match graph

Component analysis (connected components = scenes)

Additional filtering for weakly connected nodes

Fallback: isolated images → outliers

7. Pose Estimation (Structure-from-Motion)
For each cluster:

Minimal spanning set of verified pairs selected

Pairwise relative poses estimated

Reconstruction graph built

Poses propagated across cluster

Failed registrations receive:

python
rotation_matrix = nan
translation_vector = nan
8. Submission File Creation
Generates valid submission.csv with columns:

dataset, scene, image, rotation_matrix, translation_vector

Rotation matrices flattened in row-major order: r11;r12;...;r33

📁 Repository Structure
text
src/
    matching/      # Feature extraction and matching
    clustering/    # Scene grouping algorithms
    geometry/      # Geometric verification
    reconstruction/# Pose estimation
    io/           # Data loading and submission generation
notebooks/
    imc2025_solution.ipynb  # Main competition notebook
models/           # Model architectures
utils/            # Utility functions
submission/       # Generated output files
README.md         # This file
🚀 Quick Start
Running on Kaggle
Add offline package datasets:

ALIKED weights

LightGlue weights

IMC2024/2025 dependency wheels

Install dependencies:

bash
pip install --no-index /kaggle/input/imc2024-packages-lightglue-rerun-kornia/* --no-deps
mkdir -p /root/.cache/torch/hub/checkpoints
cp /kaggle/input/aliked/.../*.pth /root/.cache/torch/hub/checkpoints/
cp /kaggle/input/lightglue/.../* /root/.cache/torch/hub/checkpoints/
Run the notebook (≤ 9 hours runtime)

Download the generated submission.csv

📊 Results
Metric	Score
Official Submission Score	Corresponds to rank 140/950
Private Best Score	38 (not submitted; Top-50 potential)
📚 Scientific Bibliography
ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation
LightGlue: Local Feature Matching at Light Speed
IMC 2025: Image Matching Challenge Technical Report
