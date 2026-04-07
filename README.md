# Microscopy Segmentation Trainer/Inferencer

High-performance 2D/3D microscopy image segmentation using MONAI, PyTorch, and Shared Memory for efficient processing of massive datasets (150GB+).

## Features

- **Numba Acceleration:** High-performance JIT-compiled algorithms for 3D patch cropping, mask filtering, and volume stitching.
- **Shared Memory:** Utilizes `torch.multiprocessing` to prevent RAM duplication across workers, critical for large volumes on Windows.
- **Asynchronous Pipeline:** Optimized inference using a synchronized Disk Manager thread to maximize sequential I/O speed.
- **Pre-Packed Patches:** Zero-computation inference workers by pre-cropping patches into shared contiguous tensors.
- **Hybrid Loss:** Focal + Tversky loss to handle extreme class imbalance in sparse microscopy signals.
- **Global Normalization:** Automatic volume-level Z-score normalization using calculated metadata.

## Structure

- `train.py`: Main training script with functional epoch handlers.
- `inference.py`: Optimized batch inference script with async Disk Manager.
- `converter.py`: Utility for format conversion (OME-Zarr, Zarr, Tiff, Nifti).
- `analysis.py`: Metrics calculation (F1, Precision, Recall) against Ground Truth.
- `IO/`: Unified readers, writers, and shared-memory dataset classes.
- `models/`: Model architecture (U-Net) and factory.
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

## Docker Workflow

The repo includes a GPU-enabled Docker runner:
- Build and run: `python run_docker.py`
- This mounts `./datas` to `/workspace/datas` and provides an interactive bash shell.

## Data Layout

Training and Inference expect volumes organized in subfolders. Standard folders like `Flatten_561` or `images` are automatically discovered.

```
datas/
  dataset_name/
    volume_01/
      Flatten_561/      # Raw images
      Flatten_561_mask/ # Binary masks (detected)
    volume_02/
      Flatten_561/      # Raw images
      Flatten_561_mask/ # Binary masks (detected)
    volume_02/
      Flatten_647/      # Raw images
      Flatten_647_mask/ # Binary masks (ignored)
```

## Configuration Guide

The behavior of all scripts is controlled via a central `configs/config.json` file. The configuration is divided into sections for each task.

### 1. Global Resources (`resources`)
Settings that affect system-wide performance and concurrency.
- `numba_threads`: Number of threads for JIT-accelerated operations (default: 8).
- `dask_threads`: Number of threads for Dask-based I/O (default: 4).
- `io_workers`: Number of background workers for file reading/writing (default: 4).
- `memory_limit`: Soft memory limit in GB to avoid OOM during large volume reads.

### 2. Format Converter (`converter`)
Configuration for `converter.py`. Can be a single dictionary or a list of tasks.
- `input_path`: Path to the source volume or directory.
- `output_path`: Directory for converted output.
- `output_type`: Target format(s). Options: `ome-zarr`, `zarr`, `single-tiff`, `scroll-tiff`, `single-nii`, `scroll-nii`.
- `scroll_axis`: Axis for per-slice exports. `0-2` for forward (Z, Y, X), `3-5` for reverse (-Z, -Y, -X).
- `transpose`: Optional axis permutation (e.g., `[1, 0, 2]`).
- `resize_shape`: Optional (Z, Y, X) shape for resizing during conversion.
- `chunk_size`: Internal chunking for Zarr/OME-Zarr (default: 128).
- `levels`: Number of pyramid levels for OME-Zarr (default: 5).

### 3. Model Architecture (`model`)
Used by both training and inference.
- `type`: Model class name (e.g., `UNet`).
- `in_channels`: Input channels (usually 1).
- `out_channels`: Output classes (usually 1 for binary).
- `features`: List of feature counts per level (e.g., `[32, 64, 128, 256]`).

### 4. Training (`train`)
Configuration for `train.py`.
- `img_path`: Root directory containing raw training volumes.
- `mask_path`: Root directory containing ground truth masks.
- `input_name`: Name of the subfolder containing raw slices (default: `Flatten_561`).
- `mask_name`: Name of the subfolder containing mask slices (default: `Flatten_561_mask`).
- `training_patch_size`: (Z, Y, X) size of patches extracted for training.
- `training_epochs`: Number of epochs to train (default: 30).
- `training_batch_size`: Batch size per step (default: 8).
- `learning_rate`: Initial learning rate (default: 1e-4).
- `val_ratio`: Fraction of volumes held out for validation (default: 0.3).
- `training_neg_keep_ratio`: Probability of keeping a patch that contains no foreground signal (0.0 to 1.0).

### 5. Inference (`inference`)
Configuration for `inference.py`.
- `input_path`: Root directory to scan for volumes.
- `input_name`: Subfolder name to trigger inference on (e.g., `Flatten_561`).
- `output_path`: Directory where results will be saved, mimicking the input structure.
- `output_type`: Format for saved inference masks (e.g., `scroll-tiff`).
- `model_path`: Path to the `.pth` model checkpoint.
- `inference_patch_size`: (Z, Y, X) window size for sliding inference.
- `inference_overlay`: (Z, Y, X) overlap between windows to prevent edge artifacts.
- `batch_size`: Number of patches processed by GPU in parallel.

## Full Example `config.json`

```json
{
  "resources": {
    "numba_threads": 8,
    "io_workers": 4
  },
  "model": {
    "type": "UNet",
    "in_channels": 1,
    "out_channels": 1,
    "features": [32, 64, 128, 256]
  },
  "train": {
    "img_path": "./datas/training",
    "mask_path": "./datas/training",
    "training_patch_size": [1, 256, 256],
    "training_epochs": 50,
    "training_neg_keep_ratio": 0.05
  },
  "inference": {
    "input_path": "./datas/inference",
    "input_name": "Flatten_561",
    "output_path": "./output",
    "output_type": "scroll-tiff",
    "model_path": "./weights/best_model.pth",
    "inference_patch_size": [1, 512, 512],
    "inference_overlay": [0, 64, 64]
  }
}
```

## Training

Configure `configs/config.json` and run:
```bash
python train.py --config configs/config.json
```
Outputs are organized into:
- `visualization/`: Dataset previews and periodic validation results.
- `weights/`: Best and epoch-named `.pth` checkpoints.
- `artifacts/`: Training logs, learning curves, and a copy of the model architecture used.

## Inference

High-speed windowed inference for massive volumes:
```bash
python inference.py --config configs/config.json
```
The script automatically discovers all `input_name` directories and maintains the folder structure in the `output_path`.

## Evaluation

Calculate metrics against Ground Truth:
```bash
python analysis.py --base_dir ./datas/path/to/results --gt_name Flatten_561_mask --pred_prefix Flatten_561_mask_
```
Outputs a `metrics.xlsx` (or `.csv`) with detailed performance statistics.
