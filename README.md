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
