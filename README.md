# Microscopy Segmentation Trainer/Inferencer

High-performance 2D/3D microscopy image segmentation using MONAI, PyTorch, and Shared Memory for efficient processing of massive datasets (150GB+). Deep-learning models (UNet, AttentionUNet, SwinUNETR, VNet) and a non-backprop model (GUSL) share the same training and inference scripts — switching models is one config line.

## Features

- **Interchangeable models:** UNet, AttentionUNet, SwinUNETR, VNet and GUSL (Saab + RFT + LNT + XGBoost/MLP) all run through `train.py` / `inference.py`; pick one with `train.model_type`.
- **Shared Memory:** Utilizes `torch.multiprocessing` to prevent RAM duplication across workers, critical for large volumes.
- **Numba Acceleration:** JIT-compiled patch cropping, mask filtering, and volume stitching.
- **Asynchronous Pipeline:** Inference uses a synchronized Disk Manager thread to maximize sequential I/O speed.
- **Gaussian-weighted stitching:** Overlapping patch predictions are blended with a centre-peaked weight map, so patch borders count less.
- **Leak-free validation:** Validation is a held-out spatial block (never neighbouring patches), and patch sampling is seeded, so runs are reproducible and comparable.
- **Augmentation:** Flips on every axis + 90° in-plane rotations, plus intensity/bias-field augmentation.
- **Hybrid Loss:** Weighted sums of Dice, Tversky, Focal, BCE, Log-Cosh Dice and topology-aware clDice.
- **Intensity Normalization:** Z-score, Min-Max (+gamma) and global histogram equalization.
- **16-bit Logic:** Optimized for uint16 microscopy data.

## Structure

- `train.py`: Training for every model (epoch loop for DL models; one-shot `fit()` for GUSL).
- `inference.py`: Batch inference with the async Disk Manager and Gaussian-weighted stitching.
- `profile_model.py`: FLOPs / timing / memory profiler on synthetic tensors, or on any saved checkpoint (`--checkpoint`).
- `preprocess.py`: Configuration-driven intensity normalization.
- `converter.py`: Format conversion (OME-Zarr, Zarr, Tiff, Nifti, Scroll-Tiff, Scroll-Nifti).
- `analysis.py`: Segmentation metrics (precision, recall, F1, MCC, clDice, Hausdorff, object F1, …) against ground truth.
- `IO/`: Readers, writers and shared-memory dataset classes.
- `models/`: Model factory; `GUSL.py` + `gusl_utils/` (RFT, LNT) for GUSL.
- `utils/`: Stitcher, patch cropper, losses, metrics, normalization, visualization.
- `configs/`: Ready-made configs (see below).

## Installation

1. Create a Python 3.10 environment (e.g., Miniconda). The lab workstation already has a complete one: `conda activate unet-pytorch_310`.
2. **Install PyTorch** for your platform (CUDA/CPU): [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
3. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. *(Optional, GUSL)* `pip install "cupy-cuda12x<14"` lets GUSL keep XGBoost data on the GPU (faster, much less host RAM). Pin `<14` to keep numpy 1.x; no driver change is needed.

## Quick Start

```bash
python train.py     --config configs/config_vessel.json   # train
python inference.py --config configs/config_vessel.json   # predict
python analysis.py  --base_dir /path/to/test_root --config configs/config_vessel.json   # evaluate
```

### Ready-made configs

| Config | Data | Model it is tuned for |
|---|---|---|
| `config_vessel.json` | Lectin vessels | UNet (overlapping crops, augmentation, lr 3e-4), threshold 0.9; `gusl` entry holds the tuned vessel GUSL |
| `config_GUSL_vessel.json` | Lectin vessels | lean half-resolution GUSL (levels 3–2, Z-neighbour features, 150 RFT features, ≤150 depth-6 trees per level), threshold 0.5 — less inference energy than the vessel UNet |
| `config_cfos.json` | c-Fos cells (Hung-Yu) | UNet with recall-weighted Tversky, threshold 0.1 |
| `config_GUSL_cfos.json` | c-Fos cells (Hung-Yu) | lean half-resolution GUSL (`levels 2, finest_level 2`), threshold 0.4 — ~40% less inference energy than the c-Fos UNet |
| `config_cell.json` | general cell template | — |

In the vessel, GUSL and c-Fos configs `train.model_name` is a fresh run name, so retraining does not overwrite the kept best models (`inference.model_path` points at those).

### Switching models

Set `train.model_type` to `unet`, `attention_unet`, `swin_unetr`, `vnet` or `gusl`. Model parameters come from the matching entry in the `model` registry. Inference loads whatever model `inference.model_path` points to — no other change is needed.

## Configuration Guide

All scripts read one JSON file; each script uses its own top-level section, and the registries (`model`, `loss`, `metrics`, `normalization`, `outputs`) define available options and defaults.

### 1. Global Resources (`resources`)
- `numba_threads`, `dask_threads`: JIT / Dask thread counts.
- `io_workers`: background workers for file reading/writing.
- `memory_limit`: soft memory limit in GB for large volume reads.

### 2. Normalization (`normalization`)
- `z-score`: `std_multiplier`. `histogram`: `bins`. `min-max`: no parameters. `min-max-gamma`: `gamma`.

### 3. Model registry (`model`)
- `unet` / `attention_unet`: `channels`, `strides`, `num_res_units`, `dropout`.
- `swin_unetr`: `feature_size`, `use_checkpoint`. `vnet`: `dropout_prob`.
- `gusl`: see [GUSL](#gusl) below.

### 4. Loss registry (`loss`)
Hyperparameters per loss; `train.loss` picks a weighted sum, e.g. `{"tversky": 1, "focal": 0.2}`.
- `dice`, `log_cosh_dice`: `smooth`. `focal`: `alpha`, `gamma`. `bce`.
- `tversky`: `alpha` (false-positive weight), `beta` (false-negative weight). `alpha < beta` favours recall — useful for small, faint cells.
- `cldice`: topology-aware loss on soft skeletons (`iterations`, `smooth`). Rewards connectivity; it does not constrain thickness, so combine it with a voxel loss.

### 5. Training (`train`)
- `data_path` (or `img_path` / `mask_path`): roots searched recursively for `input_name` / `mask_name` folders.
- `model_type`, `model_name`, `save_path`: which model, run name, output root (`save_path/model_name/{weights,artifacts,visualization}`).
- `preprocess`: `normalize_mode`, `pad_mode`, `sample_rate`, `low_cut`, `high_cut`.
- `training_patch_size`: `[D, H, W]`; `D == 1` switches to 2D mode.
- `training_overlay`: overlap between training crops — more, more varied crops (e.g. `[16, 32, 32]`).
- `training_neg_keep_ratio`: background-only patches kept **per positive patch**.
- `val_ratio`: size of the held-out validation block. It is the last slab of Z when that stays ≤ 2×`val_ratio` of the volume; thin stacks fall back to the longest axis. Patches crossing the cut are dropped.
- `seed`: seeds patch sampling, the split, torch and augmentation.
- `training_epochs`, `warmup_epochs`, `learning_rate`: linear warmup, then cosine decay to 1% of the base LR.
- `training_batch_size`, `training_num_workers`, `loss`, `metrics`, `metric_interval`.
- `pad_div32` *(optional)*: pad patches to multiples of 32. Defaults to what the model needs (only SwinUNETR).

### 6. Inference (`inference`)
- `input_path` / `input_name`, `output_path` / `output_name`, `model_path`, `device`, `batch_size`.
- `inference_patch_size`, `inference_overlay`: sliding-window geometry.
- `blend`: `gaussian` (default) or `constant` weighting of overlapping patches.
- `output`: `type` (`Scroll-Tiff`, `Zarr`, `OME-Zarr`, …), `dtype`, and `threshold` — the foreground probability cut-off (default 0.5). **Tune it**: the best value is model- and data-dependent (0.9 for the vessel UNet, 0.1 for the c-Fos UNet).

### 7. Format Converter (`converter`)
- `output.type`: target format; `scroll_axis` for per-slice exports (`0-2` forward, `3-5` reverse).

## GUSL

A non-backprop, coarse-to-fine voxel regressor. Each level (deepest → level 1, at XY scale 1/2^(L-1)) computes Saab, neighbourhood, raw and gradient features on the GPU, keeps the most informative ones (RFT), adds linear projections (LNT), and regresses the residual of the coarser level with XGBoost (or a small MLP). It saves as a normal `.pth` and runs through the standard inference pipeline.

Key `model.gusl` options (per-level lists run deepest → level 1; a single value applies to all levels):
- `levels`, `finest_level` (stop at a coarser level and upsample the output — e.g. `levels: 2, finest_level: 2` = one half-resolution level; best for compact objects like c-Fos cells); `kernel_size`, `kernel_depth`; `neigh_size`, `neigh_depth`, `neigh_stride`; `use_grad`, `grad_size`, `grad_depth`.
- `n_selected` (RFT features kept), `lnt_depth`, `lnt_num_tree`, `boundary_window`, `neg_keep_frac`.
- Sample caps (voxels per level): `saab_samples`, `encode_samples`, `decode_samples`, `val_samples`. `decode_samples` drives memory (~8M samples ≈ 14 GB, on the GPU when cupy is installed).
- `head`: `xgboost` (default) or `mlp` (`mlp_hidden: []` = linear). XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `early_stopping_rounds`, `max_bin`.

## Profiling

```bash
python profile_model.py --model unet --gpu 0                       # untrained DL model, synthetic input
python profile_model.py --checkpoint output/.../weights/X.pth --patch 32 64 64 --batch_size 16   # any saved model (required for GUSL)
```

## Evaluation

```bash
python analysis.py --base_dir /path/to/test_root --output_name evaluation_report --config configs/config_vessel.json
```
Ground truth = every folder ending `_mask` under `--base_dir`; predictions = sibling folders ending `.scroll-tif(f)`. Writes `<output_name>.xlsx` with per-model metrics. `--config` only supplies the `metrics` registry parameters.

For downstream vessel / cell statistics use `Chulab-Signal_Analyzer` (`vessel_analyzer.py`, `cell_analyzer.py`). For vessel masks set `"fill_holes_px": 10` in each `vessel_analyzer` task: small holes inside vessels otherwise create false trifurcations.
