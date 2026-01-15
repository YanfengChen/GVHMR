# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GVHMR (Gravity-View Human Motion Recovery) is a deep learning system for recovering world-grounded 3D human motion from monocular video. It predicts SMPL-X body model parameters in a gravity-aligned coordinate system, enabling accurate global motion estimation even with camera movement.

## Common Commands

### Demo/Inference
```bash
# Single video inference (use -s for static camera to skip visual odometry)
python tools/demo/demo.py --video=path/to/video.mp4 -s

# Batch processing folder of videos
python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s

# Specify focal length (in mm, for known camera)
python tools/demo/demo.py --video=video.mp4 --f_mm=24
```

### Training
```bash
# Train GVHMR model (default: 2x4090 for 420 epochs)
python tools/train.py exp=gvhmr/mixed/mixed
```

### Testing/Evaluation
```bash
# Test on 3DPW, RICH, and EMDB datasets
python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt

# Individual dataset testing (change global/task)
python tools/train.py global/task=gvhmr/test_3dpw exp=gvhmr/mixed/mixed ckpt_path=...
python tools/train.py global/task=gvhmr/test_emdb exp=gvhmr/mixed/mixed ckpt_path=...
python tools/train.py global/task=gvhmr/test_rich exp=gvhmr/mixed/mixed ckpt_path=...
```

### Installation
```bash
conda create -y -n gvhmr python=3.10
conda activate gvhmr
pip install -r requirements.txt  # or requirements_blackwell.txt for RTX 50xx
pip install -e .
```

## Architecture

### Core Components (`hmr4d/`)

- **`model/gvhmr/`**: Main model implementation
  - `gvhmr_pl.py`: PyTorch Lightning module for training/validation
  - `gvhmr_pl_demo.py`: Simplified demo inference module
  - `pipeline/gvhmr_pipeline.py`: Forward pass pipeline - encodes conditions, runs transformer, decodes SMPL params
  - `utils/endecoder.py`: Normalizes/denormalizes motion data for network I/O
  - `utils/postprocess.py`: Post-processing for static joint detection and IK refinement

- **`network/`**: Neural network architectures
  - `gvhmr/relative_transformer.py`: Main transformer network with RoPE positional encoding. Takes 2D keypoints, CLIFF camera params, camera angular velocity, and ViT features as input
  - `hmr2/`: HMR2 feature extractor (ViT-based)
  - `base_arch/`: Transformer building blocks with rotary embeddings

- **`utils/preproc/`**: Preprocessing pipeline components
  - `tracker.py`: YOLO-based person tracking
  - `vitpose.py`: ViTPose 2D keypoint detection
  - `vitfeat_extractor.py`: HMR2 ViT feature extraction
  - `relpose/simple_vo.py`: Lightweight visual odometry (default, replaces DPVO)
  - `slam.py`: DPVO integration (optional, requires separate build)

- **`dataset/`**: Dataset loaders for BEDLAM, H36M, 3DPW, EMDB, RICH, AMASS

- **`configs/`**: Hydra configuration files
  - `demo.yaml`: Demo inference config
  - `train.yaml`: Training config
  - `exp/gvhmr/mixed/mixed.yaml`: Main experiment config

### Key Design Patterns

1. **Hydra Configuration**: All configs in `hmr4d/configs/`. Use overrides like `exp=gvhmr/mixed/mixed` or `global/task=gvhmr/test_3dpw`

2. **Gravity-View Coordinates**: The model predicts motion in a gravity-aligned coordinate system (Y-axis = gravity). Camera-relative predictions are transformed to world coordinates using camera angular velocity

3. **Preprocessing Pipeline**: Demo runs sequentially: tracking (YOLO) -> 2D pose (ViTPose) -> ViT features (HMR2) -> visual odometry (SimpleVO/DPVO) -> GVHMR inference -> rendering

4. **Static Camera Mode (-s flag)**: Skips visual odometry, assumes camera is fixed. Results in faster inference and avoids VO drift for tripod shots

## Dependencies

- PyTorch 2.3+ with CUDA
- PyTorch Lightning + Hydra for training
- pytorch3d for 3D operations
- SMPL-X body model (requires registration at smpl-x.is.tue.mpg.de)

## Code Style

Uses Black formatter with 120 character line length (see `pyproject.toml`).
