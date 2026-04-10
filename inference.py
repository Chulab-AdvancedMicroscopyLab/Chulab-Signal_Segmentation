"""
Batch volume inference with a synchronized Disk Manager pipeline.

Architecture:
1. Disk Manager Thread: Alternates between Reading the next window and Writing 
   the previous result. This prevents I/O contention and stabilizes RAM.
2. Main Process: Executes GPU inference.
"""
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
    outputs = []
    with torch.no_grad():
        for inputs in loader:
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            inputs = inputs.to(device)
            preds = model(inputs)
            if preds.ndim == 5: # 3D
                preds = preds.squeeze(1) 
            elif preds.ndim == 4: # 2D
                preds = preds.squeeze(1)[:, np.newaxis, ...]
            outputs.append(preds.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)

def disk_manager_worker(
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
        # But wait, the user wants: Write N-1 THEN Read N+1 while GPU is busy with N.
        
        for i, (z_start, z_overlay_actual) in enumerate(z_plan):
            z_end = min(z_start + patch_size[0], volume_shape[0])
            
            # 1. READ Chunk i
            # If i > 0, this happens while GPU is busy with i-1. 
            # BUT the user wants Write i-2 to happen BEFORE Read i.
            # (Note: i is the current chunk being loaded)
            
            dataset = InferenceMicroscopyDataset(
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
                
                mask_patches, data_position, res_z_start, res_z_end, res_z_overlay = item
                actual_chunk_depth = res_z_end - res_z_start
                local_positions = [(pos[0] - res_z_start, pos[1], pos[2]) for pos in data_position]
                
                logging.info(f"  Stitching Z: {res_z_start}-{res_z_end}...")
                stitched_volume, prev_z_slices = stitch_image(
                    patches=mask_patches, 
                    positions=local_positions,
                    original_shape=(actual_chunk_depth, volume_shape[1], volume_shape[2]),
                    patch_size=patch_size,
                    z_overlay=res_z_overlay,
                    prev_z_slices=prev_z_slices,
                    resize_factor=resize_factor,
                )
                data_writer.write(stitched_volume, z_start=res_z_start, z_end=res_z_start+stitched_volume.shape[0])
                stitch_queue.task_done()

        # 3. FINAL CLEANUP: Write the very last chunk results
        item = stitch_queue.get()
        if item is not None:
            mask_patches, data_position, res_z_start, res_z_end, res_z_overlay = item
            actual_chunk_depth = res_z_end - res_z_start
            local_positions = [(pos[0] - res_z_start, pos[1], pos[2]) for pos in data_position]
            
            logging.info(f"  Stitching Z: {res_z_start}-{res_z_end}...")
            stitched_volume, _ = stitch_image(
                patches=mask_patches, 
                positions=local_positions,
                original_shape=(actual_chunk_depth, volume_shape[1], volume_shape[2]),
                patch_size=patch_size,
                z_overlay=0, 
                prev_z_slices=prev_z_slices,
                resize_factor=resize_factor,
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
    resources = full_config.get("resources", {})
    io_workers = resources.get("io_workers", 4)

    data_reader = FileReader(
        volume_path, 
        io_workers=io_workers, 
        compute_stats=True, 
        stats_sample_rate=config.get("stats_sample_rate", 0.1)
    )
    os.makedirs(output_dir, exist_ok=True)
    
    output_type_str = config.get("output_type", "Scroll-Tif")
    output_type = TYPE_MAP.get(output_type_str, output_type_str)
    
    data_writer = FileWriter(
        output_path=output_dir, output_name=output_name, output_type=output_type,
        output_dtype=config.get("output_dtype", "uint16"),
        full_res_shape=data_reader.volume_shape, file_name=data_reader.volume_files,
        chunk_size=tuple(config.get("output_chunk_size", [128, 128, 128])),
        resize_factor=config.get("output_resize_factor", 2),
        io_workers=io_workers,
    )
    
    patch_size = tuple(config.get("inference_patch_size", [16, 64, 64]))
    overlay_3d = tuple(config.get("inference_overlay", [2, 4, 4]))
    z_plan = compute_z_plan(data_reader.volume_shape[0], patch_size[0], overlay_3d[0])
    
    inf_queue = queue.Queue(maxsize=1)
    stitch_queue = queue.Queue(maxsize=1)
    
    disk_thread = threading.Thread(
        target=disk_manager_worker,
        args=(data_reader, data_writer, z_plan, patch_size, overlay_3d,
              config.get("inference_resize_factor", [1.0, 1.0, 1.0]), 
              inf_queue, stitch_queue, output_type),
        daemon=True
    )
    disk_thread.start()
    
    logging.info(f"Inference: {output_dir / output_name}")
    
    # 1. Spot Check: Visualize middle chunk before starting the full run
    if config.get("visualize_preview", False):
        mid_idx = len(z_plan) // 2
        z_start, z_overlay_actual = z_plan[mid_idx]
        z_end = min(z_start + patch_size[0], data_reader.volume_shape[0])
        
        logging.info(f"Spot Check: Visualizing middle chunk Z:{z_start}-{z_end}")
        spot_ds = InferenceMicroscopyDataset(
            image_reader=data_reader,
            z_range=(z_start, z_end),
            patch_size=patch_size,
            overlap=overlay_3d, 
            transform=inference_transform
        )
        spot_loader = DataLoader(spot_ds, batch_size=config.get("batch_size", 8), shuffle=False, num_workers=0)
        spot_preds = run_inference(model, spot_loader, device)
        
        # Spot check visualization: pass predictions as the second argument (where 'masks' would be)
        # image_tensors[0] is (N, 1, D, H, W)
        visualize_predictions(
            images=spot_ds.image_tensors[0].numpy(), 
            masks=spot_preds, 
            predictions=None, 
            save_path=output_dir, 
            title=f"{output_name}_spot_check_Z{z_start}"
        )

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
        stitch_queue.put((mask_patches, data_position, z_start, z_end, z_overlay_actual))
        
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
