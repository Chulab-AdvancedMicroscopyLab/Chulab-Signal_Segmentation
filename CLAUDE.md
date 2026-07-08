# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

High-performance 2D/3D microscopy image segmentation pipeline. Targets sparse-signal volumes (150GB+) from fluorescence microscopy. Optimized for RAM efficiency via shared memory and async I/O.

## Commands

```bash
# Install (after setting up PyTorch with CUDA manually)
pip install -r requirements.txt

# Train
python train.py --config configs/config_cell.json

# Inference
python inference.py --config configs/config_cell.json

# Preprocess (normalize intensities before training/inference)
python preprocess.py --input /path/to/raw --output /path/to/norm --config configs/config_cell.json --mode inference

# Convert format (OME-Zarr, Zarr, Tiff, Nifti, Scroll-Tiff, Scroll-Nifti)
python converter.py --config configs/config_cell.json

# Evaluate predictions against ground truth
python analysis.py --base_dir ./datas/path/to/results --gt_name images_mask --pred_prefix images_mask_

# GUSL (non-DL alternative: Saab + RFT + LNT + XGBoost)
python train_GUSL.py --config configs/config_GUSL.json
python inference_GUSL.py --config configs/config_GUSL.json
```

## Architecture

### Data Flow

```
Volume directory (Tiff/Zarr/NIfTI slices)
  → FileReader (parallel load, stats, normalization)
  → PatchCropper (Numba-JIT, mask filtering)
  → Shared Memory tensors (torch.share_memory_())
  → DataLoader (workers read zero-copy from shared tensors)
  → Model → Stitcher → FileWriter
```

### Config System

Single JSON file controls everything. All scripts read the same file; each has its own top-level key (`"train"`, `"inference"`, `"converter"`). Registries at the top (`"model"`, `"loss"`, `"metrics"`, `"normalization"`, `"outputs"`) define available options and their defaults — the `"train"` section references them by name.

Key config fields in `"train"`:
- `data_path` (or `img_path`/`mask_path`): list of root directories containing `input_name`/`mask_name` subdirs
- `model_type`: key into `"model"` registry (`"unet"`, `"attention_unet"`, `"swin_unetr"`, `"vnet"`, `"gusl"`)
- `training_patch_size`: `[D, H, W]` — if `D == 1`, model uses 2D mode (patch dim squeezed automatically)
- `loss`: dict of `{loss_name: weight}` — weighted sum from loss registry
- `metrics`: list of metric keys from metrics registry; computed every `metric_interval` epochs

### Shared Memory Dataset (critical design)

`TrainMicroscopyDataset` pre-crops all patches at construction, stacks them into a single `(N, 1, D, H, W)` shared tensor. DataLoader workers index into this tensor with zero-copy — avoids RAM duplication across workers essential for 150GB+ volumes. `InferenceMicroscopyDataset` does the same for inference chunks.

Patch extraction uses Numba-JIT functions in `utils/cropper.py`. `filter_indices_by_mask` drops background-only patches based on `training_neg_keep_ratio`.

### 2D vs 3D Mode

Determined automatically from `training_patch_size[0]`: if `D == 1`, the depth dimension is squeezed before passing to the model. Models in `models/` are MONAI wrappers and handle both `spatial_dims=2` and `spatial_dims=3`.

SwinUNETR requires input dimensions divisible by 32 — `datasets.py` auto-pads patches at the end to satisfy this.

### Inference Pipeline

`inference.py` uses a Disk Manager thread pattern: one thread alternates between reading the next Z-window and writing the previous result while the main process runs GPU inference. This serializes I/O to maximize sequential disk throughput and prevent OOM from concurrent large reads.

Z-windows are tiled with overlap (`inference_overlay`) and stitched by `utils/stitcher.py` using Numba-JIT averaging in the overlap regions.

### GUSL Model

Alternative to DL: coarse-to-fine classical pipeline (Saab transform → Random Forest Transform → Linear Network Transform → XGBoost). Lives in `models/GUSL.py` + `models/gusl_utils/`. Uses the same `IO/` and `utils/` infrastructure as DL models. Trained via `train_GUSL.py`, inferred via `inference_GUSL.py`.

### Output Formats

`IO/IO_types.py` maps UI labels to internal writer keys. Writer in `IO/writer.py` dispatches on these keys. Scroll-Tiff/Scroll-Nifti export per-slice files along a chosen axis.

### Loss Functions (`utils/loss.py`)

All custom: `DiceLoss`, `FocalLoss`, `TverskyLoss`, `LogCoshDiceLoss`, `BCELoss`, plus `HybridLoss` that takes a `{name: weight}` dict and sums them. Built by `build_loss_from_config`.

## Data Directory Layout

Training expects volumes discovered by recursive search for directories named `input_name` (default `"images"`) under `data_path`. Corresponding mask dirs named `mask_name` (default `"images_mask"`) must exist at the same relative path. Weights saved under `save_path/<model_name>/weights/`, visualizations under `.../visualization/`, logs under `.../artifacts/`.
