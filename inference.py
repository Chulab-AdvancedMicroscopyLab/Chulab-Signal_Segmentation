#!/usr/bin/env python3
"""
Batch volume inference with a synchronized Disk Manager pipeline.

Architecture:
1. Disk Manager Thread: Alternates between Reading the next window and Writing 
   the previous result. This prevents I/O contention and stabilizes RAM.
2. Main Process: Executes GPU inference.
"""
import warnings
# Suppress the cuda.cudart module deprecation warning (must be done before other imports)
warnings.filterwarnings("ignore", category=FutureWarning, module="cuda")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import argparse
import os
import json
import threading
import queue
import numpy as np
from pathlib import Path
from typing import Tuple, List

import torch
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.utility.dictionary import ToTensord

from IO import FileReader, FileWriter, InferenceMicroscopyDataset, TYPE_MAP
from utils.cropper import compute_z_plan
from utils.stitcher import stitch_image
from utils.visualization import visualize_predictions
from utils.concurrency import initialize_concurrency

# Standard transform
inference_transform = Compose([
    ToTensord(keys=["image"], dtype=torch.float32),
])

def load_checkpoint(model_path: str):
    """Load a torch model checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return torch.load(model_path, weights_only=False)

def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Execute model inference on a dataloader. Returns (N, D, H, W)."""
    model.eval()
    
    dataset = loader.dataset
    num_samples = len(dataset)
    patch_shape = dataset.image_tensors[0].shape[2:] 
    
    # Pre-allocate on GPU to avoid "jumping" and minimize CPU-GPU syncs
    # Note: If this hits OOM, we'd need to fall back to CPU pre-allocation.
    # But for a single chunk's worth of patches, it usually fits.
    try:
        outputs = torch.empty((num_samples, *patch_shape), device=device, dtype=torch.float32)
        use_gpu_acc = True
    except RuntimeError: # OOM on GPU
        logging.warning("OOM during GPU pre-allocation for inference, falling back to CPU.")
        outputs = np.empty((num_samples, *patch_shape), dtype=np.float32)
        use_gpu_acc = False
    
    curr_idx = 0
    with torch.no_grad():
        for inputs in loader:
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            
            batch_size = inputs.shape[0]
            inputs = inputs.to(device)
            preds = model(inputs)
            
            if preds.ndim == 5: # 3D (B, C, D, H, W)
                preds = preds.squeeze(1) 
            elif preds.ndim == 4: # 2D (B, C, H, W)
                preds = preds.squeeze(1)[:, np.newaxis, ...]
            
            if use_gpu_acc:
                outputs[curr_idx:curr_idx + batch_size] = preds
            else:
                outputs[curr_idx:curr_idx + batch_size] = preds.detach().cpu().numpy()
            
            curr_idx += batch_size
            
    # Single large transfer to CPU at the end
    if use_gpu_acc:
        return outputs.cpu().numpy()
    else:
        return outputs

def disk_manager_worker(
    full_config: dict,
    data_reader: FileReader,
    data_writer: FileWriter,
    z_plan: List[Tuple[int, int]],
    patch_size: Tuple[int, int, int],
    overlay_3d: Tuple[int, int, int],
    resize_factor: List[float],
    inf_queue: queue.Queue,
    stitch_queue: queue.Queue,
    output_type: str
):
    """
    Synchronized Disk Thread: Alternates between Reading and Writing.
    Ensures that Disk Read and Disk Write never happen at the same time.
    """
    prev_z_slices = None
    volume_shape = data_reader.volume_shape

    try:
        # Loop through the plan: Load N, then wait for results of N and Write N.
        # To maximize sequential IO, we want to Read N+1 while GPU is busy with N.
        # But wait, the user wants: Write i-1 THEN Read i+1 while GPU is busy with i.
        
        for i, (z_start, z_overlay_actual) in enumerate(z_plan):
            z_end = min(z_start + patch_size[0], volume_shape[0])
            
            # 1. READ Chunk i
            dataset = InferenceMicroscopyDataset(
                full_config=full_config,
                image_reader=data_reader,
                z_range=(z_start, z_end),
                patch_size=patch_size,
                overlap=overlay_3d, 
                transform=inference_transform
            )
            # Hand to GPU
            inf_queue.put((dataset, z_start, z_end, z_overlay_actual))

            # 2. WRITE Results of i-1
            # While the GPU starts working on chunk i, we can write the results of i-1.
            # We skip this for i=0 because there are no results yet.
            if i > 0:
                item = stitch_queue.get()
                if item is None: break
                
                mask_patches, data_position, res_z_start, res_z_end, res_z_overlay, dataset = item
                actual_chunk_depth = res_z_end - res_z_start
                pad_z_before = getattr(dataset, "pad_z_before", 0)
                
                # Adjust positions to account for symmetric Z-padding
                local_positions = [(pos[0] - res_z_start - pad_z_before, pos[1], pos[2]) for pos in data_position]
                
                logging.info(f"  Stitching Z: {res_z_start}-{res_z_end}...")
                stitched_volume, prev_z_slices = stitch_image(
                    patches=mask_patches, 
                    positions=local_positions,
                    original_shape=(actual_chunk_depth, volume_shape[1], volume_shape[2]),
                    patch_size=patch_size,
                    z_overlay=res_z_overlay,
                    prev_z_slices=prev_z_slices,
                    resize_factor=resize_factor,
                    output_dtype=data_writer.output_dtype
                )

                data_writer.write(stitched_volume, z_start=res_z_start, z_end=res_z_start+stitched_volume.shape[0])
                stitch_queue.task_done()

        # 3. FINAL CLEANUP: Write the very last chunk results
        item = stitch_queue.get()
        if item is not None:
            mask_patches, data_position, res_z_start, res_z_end, res_z_overlay, dataset = item
            actual_chunk_depth = res_z_end - res_z_start
            pad_z_before = getattr(dataset, "pad_z_before", 0)
            
            # Adjust positions to account for symmetric Z-padding
            local_positions = [(pos[0] - res_z_start - pad_z_before, pos[1], pos[2]) for pos in data_position]
            
            logging.info(f"  Stitching Z: {res_z_start}-{res_z_end}...")
            stitched_volume, _ = stitch_image(
                patches=mask_patches, 
                positions=local_positions,
                original_shape=(actual_chunk_depth, volume_shape[1], volume_shape[2]),
                patch_size=patch_size,
                z_overlay=0, 
                prev_z_slices=prev_z_slices,
                resize_factor=resize_factor,
                output_dtype=data_writer.output_dtype
            )

            data_writer.write(stitched_volume, z_start=res_z_start, z_end=res_z_start+stitched_volume.shape[0])
            stitch_queue.task_done()

        if output_type == "ome-zarr":
            data_writer.complete_ome()

    except Exception as e:
        logging.error(f"Disk Manager failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        inf_queue.put(None) 
    finally:
        inf_queue.put(None)

def process_volume(volume_path: Path, output_dir: Path, output_name: str, model: torch.nn.Module, device: torch.device, full_config: dict):
    """Processes a volume using sequential loading/inference and async stitching."""
    config = full_config.get("inference", {})
    preprocess_config = config.get("preprocess", {})
    output_config = config.get("output", {})
    registry = full_config.get("outputs", {})
    resources = full_config.get("resources", {})
    io_workers = resources.get("io_workers", 4)

    method = preprocess_config.get("normalize_mode", "z-score")
    data_reader = FileReader(
        volume_path,
        io_workers=io_workers,
        compute_stats=True,
        stats_sample_rate=preprocess_config.get("sample_rate", 1.0),
        low_cut=preprocess_config.get("low_cut"),
        high_cut=preprocess_config.get("high_cut"),
        compute_histogram=(method == "histogram")
    )
    
    os.makedirs(output_dir, exist_ok=True)
    output_type_str = output_config.get("type", "Scroll-Tiff")
    output_type = TYPE_MAP.get(output_type_str, output_type_str)
    
    # Merge parameters: Registry (global defaults) + Local Output Config
    # Use the unified internal type for registry lookup
    type_key = output_type
    type_defaults = registry.get(type_key, {})
    type_overrides = output_config.get(type_key, {})
    
    # Final params for the specific output type
    final_type_params = {**type_defaults, **type_overrides}
    
    # For OME-Zarr/Zarr: extract n_level, chunk_size, and resize_factor
    n_level = final_type_params.get("n_level", final_type_params.get("levels", 5))
    resize_factor = final_type_params.get("resize_factor", output_config.get("resize_factor", 2))
    chunk_size = final_type_params.get("chunk_size", [128, 128, 128])
    if isinstance(chunk_size, int):
        chunk_size = (chunk_size, chunk_size, chunk_size)
    
    data_writer = FileWriter(
        output_path=output_dir, output_name=output_name, output_type=output_type,
        output_dtype=output_config.get("dtype", "uint16"),
        full_res_shape=data_reader.volume_shape, file_name=data_reader.volume_files,
        chunk_size=tuple(chunk_size),
        n_level=n_level,
        resize_factor=resize_factor,
        resize_order=output_config.get("resize_order", 0),
        io_workers=io_workers,
    )
    
    patch_size = tuple(config.get("inference_patch_size", [16, 64, 64]))
    overlay_3d = tuple(config.get("inference_overlay", [2, 4, 4]))
    z_plan = compute_z_plan(data_reader.volume_shape[0], patch_size[0], overlay_3d[0])
    
    logging.info(f"Inference: {output_dir / output_name}")
    
    inf_queue = queue.Queue(maxsize=1)
    stitch_queue = queue.Queue(maxsize=1)
    disk_thread = threading.Thread(
        target=disk_manager_worker,
        args=(
            full_config, data_reader, data_writer, z_plan, patch_size, overlay_3d, 
            config.get("inference_resize_factor", [1.0, 1.0, 1.0]),
            inf_queue, stitch_queue, output_type
        )
    )
    disk_thread.start()

    import time

    while True:
        inf_data = inf_queue.get()
        if inf_data is None: break
            
        dataset, z_start, z_end, z_overlay_actual = inf_data

        # Use num_workers=0 because patches are already pre-extracted into a shared memory tensor.
        # This avoids the overhead of spawning/forking processes in every loop iteration.
        loader = DataLoader(dataset, batch_size=config.get("batch_size", 8), shuffle=False, num_workers=0)
        
        logging.info(f"  Inference Z:{z_start}-{z_end}")
        mask_patches = run_inference(model, loader, device)
        
        data_position = [meta.slices.global_coords for meta in dataset.patch_indices]
        stitch_queue.put((mask_patches, data_position, z_start, z_end, z_overlay_actual, dataset))
        
    disk_thread.join()

def main():
    parser = argparse.ArgumentParser(description="Batch Inference: Synchronized Disk Pipeline")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        full_config = json.load(f)
        config = full_config.get("inference", {})
    
    # Initialize concurrency settings
    initialize_concurrency(full_config)
    
    input_path_str = config.get("input_path")
    output_path_str = config.get("output_path", input_path_str)
    input_name = config.get("input_name", "Flatten_561")
    output_name = config.get("output_name", input_name)
    model_path = config.get("model_path")
    
    if not input_path_str or not model_path:
        logging.error("Missing mandatory paths in config."); return 1
        
    root_input = Path(input_path_str).resolve()
    root_output = Path(output_path_str).resolve()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(root_output / f"inference_{timestamp}.log", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Loading model: {model_path}")
    model = load_checkpoint(model_path).to(device)
    
    volumes_to_process = []
    if root_input.name == input_name:
        volumes_to_process.append(root_input)
    
    for p in root_input.rglob("*"):
        if p.is_dir() and p.name == input_name:
            volumes_to_process.append(p)
            
    if not volumes_to_process:
        logging.warning(f"No directories named '{input_name}' found under {root_input}")
        return 0

    logging.info(f"Found {len(volumes_to_process)} volumes to process.")

    for v_path in sorted(volumes_to_process):
        rel_parent = v_path.parent.relative_to(root_input)
        target_output_dir = root_output / rel_parent
        process_volume(v_path, target_output_dir, output_name, model, device, full_config)

    logging.info("Batch inference complete.")

if __name__ == "__main__":
    main()
