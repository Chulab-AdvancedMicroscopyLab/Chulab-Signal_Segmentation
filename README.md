# Microscopy Segmentation Trainer/Inferencer

High-performance 2D/3D microscopy image segmentation using MONAI, PyTorch, and Shared Memory for efficient processing of massive datasets (150GB+).

## Features

- **SwinUNETR Support:** State-of-the-art Transformer-based encoder for 3D segmentation, fully integrated with padding safety.
- **Automatic Model Padding:** Ensures all input patches are at least the size of `patch_size` and their dimensions are divisible by 32 (required for SwinUNETR/Transformer models).
- **Numba Acceleration:** High-performance JIT-compiled algorithms for 3D patch cropping, mask filtering, and volume stitching.
- **Shared Memory:** Utilizes `torch.multiprocessing` to prevent RAM duplication across workers, critical for large volumes.
- **Asynchronous Pipeline:** Optimized inference using a synchronized Disk Manager thread to maximize sequential I/O speed.
- **Pre-Packed Patches:** Zero-computation inference workers by pre-cropping patches into shared contiguous tensors.
- **Hybrid Loss:** Focal + Tversky + Dice loss combinations to handle extreme class imbalance in sparse microscopy signals.
- **Intensity Normalization:** Support for Z-score, Min-Max, and Global Histogram Equalization via `preprocess.py` and reader integration.
- **16-bit Logic:** Optimized for 16-bit (uint16) microscopy data with automatic scaling for visualization (65535 for foreground).

## Structure

- `preprocess.py`: Configuration-driven intensity normalization (Z-score, Min-Max, Histogram).
- `train.py`: Main training script with functional epoch handlers.
- `inference.py`: Optimized batch inference script with async Disk Manager.
- `converter.py`: Utility for format conversion (OME-Zarr, Zarr, Tiff, Nifti).
- `analysis.py`: Metrics calculation (F1, Precision, Recall) against Ground Truth.
- `IO/`: Unified readers, writers, and shared-memory dataset classes.
- `models/`: Model architecture factory (UNet, AttentionUNet, SwinUNETR, VNet).
- `utils/`: Numba-optimized stitcher, patch cropper, and visualization tools.

## Installation

1. Create a Python 3.10+ environment (e.g., using Miniconda).
2. **Install PyTorch** following the official instructions for your platform (CUDA/CPU):
   [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
   
   Example (CUDA 11.8):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `numba` is used for high-speed JIT acceleration.*

4. Make scripts executable (Linux):
   ```bash
   chmod +x *.py
   ```

## Preprocessing

Before training or inference, you can normalize your volume intensities:
```bash
python preprocess.py --input /path/to/raw --output /path/to/norm --config configs/config.json --mode inference
```

## Configuration Guide

The behavior of all scripts is controlled via a central `configs/config.json` file. 

### 1. Global Resources (`resources`)
- `numba_threads`: Number of threads for JIT-accelerated operations (default: 8).
- `io_workers`: Number of background workers for file reading/writing (default: 4).
- `memory_limit`: Soft memory limit in GB to avoid OOM during large volume reads.

### 2. Normalization (`normalization`)
Global defaults for intensity normalization:
- `z-score`: `std_multiplier` (default: 1.0).
- `histogram`: `bins` (default: 1024).
- `min-max`: (no parameters required).

### 3. Format Converter (`converter`)
- `output_type`: Target format(s). Options: `OME-Zarr`, `Zarr`, `Tiff`, `Nifti`, `Scroll-Tiff`, `Scroll-Nifti`.
- `scroll_axis`: Axis for per-slice exports. `0-2` for forward, `3-5` for reverse.

### 4. Model Architecture (`model`)
- `swin_unetr`: feature_size=48, spatial_dims=3.
- `attention_unet`: channels=[32, 64, 128, 256, 512].
- `unet`: standard MONAI UNet configuration.

### 5. Training (`train`)
- `preprocess`: `method` ("z-score", "min-max", "histogram"), `low_cut`, `high_cut`.
- `training_patch_size`: [64, 64, 64]. *Note: Will be automatically padded if not divisible by 32.*

### 6. Inference (`inference`)
- `output`: `type` ("Scroll-Tiff", "Zarr", etc.), `dtype` ("uint16", "uint8").
- `inference_patch_size`: [64, 64, 64]. 
- `inference_overlay`: [16, 16, 16]. 
- `batch_size`: patch count processed by GPU at once.

## Evaluation

Calculate metrics against Ground Truth:
```bash
python analysis.py --base_dir ./datas/path/to/results --gt_name images_mask --pred_prefix images_mask_
```
Outputs a `metrics.xlsx` with detailed performance statistics.
