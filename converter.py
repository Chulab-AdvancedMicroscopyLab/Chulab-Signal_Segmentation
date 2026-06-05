#!/usr/bin/env python3
"""
Command-line entry point for converting large 3D volumes into various
output formats (OME-Zarr pyramids, flat Zarr, TIFF/NIfTI volumes, and
per-slice "scroll" exports).

The CLI streams the input volume using `IO.reader.FileReader` and writes
results incrementally via `IO.writer.FileWriter` to keep memory bounded.
"""

import sys
import json
from pathlib import Path

# 1. PRE-IMPORT CONCURRENCY SETUP
# We must set environment variables (OMP_NUM_THREADS, etc.) before 
# importing numpy, numba, or dask to ensure they are respected.
def _pre_init_concurrency():
    config_path = "configs/config.json"
    for i, arg in enumerate(sys.argv):
        if (arg == "--config" or arg == "-c") and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
    
    config = {}
    if Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception:
            pass
            
    from utils.concurrency import initialize_concurrency
    initialize_concurrency(config)

_pre_init_concurrency()

# 2. STANDARD IMPORTS
import argparse
import logging
import gc
import numpy as np

from IO import FileReader, FileWriter, TYPE_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_args():
    """Parse CLI arguments describing the input volume and desired outputs."""
    parser = argparse.ArgumentParser(description="Convert image volume to multiscale OME-Zarr or other formats.")
    parser.add_argument("--config", type=str, default="configs/config.json", required=True, help="Path to a JSON config file")
    return parser.parse_args()

def _write_pyramid(reader: FileReader, args, full_res_shape, chunk_tuple, io_output_type: str, io_workers: int = 4, output_dtype=None) -> bool:
    """Stream the full-resolution volume into a multiscale Zarr layout layout."""
    writer = FileWriter(
        output_path=args.output_path,
        output_name=reader.volume_name,
        output_type=io_output_type,
        full_res_shape=tuple(full_res_shape),
        output_dtype=output_dtype or reader.volume_dtype,
        chunk_size=chunk_tuple,
        n_level=args.levels,
        resize_factor=args.downscale_factor,
        resize_order=args.resize_order,
        input_shape=tuple(reader.volume_shape),
        io_workers=io_workers,
    )

    z_max = reader.volume_shape[0]
    for z0 in range(0, z_max, args.chunk_size):
        z1 = min(z0 + args.chunk_size, z_max)
        arr = reader.read(z_start=z0, z_end=z1)
        writer.write(arr, z_start=z0, z_end=z1)
        del arr

    try:
        writer.complete_resize()
    except Exception as e:
        logging.error(f"Failed to finalize resize into output: {e}")

    if io_output_type == "ome-zarr":
        writer.complete_ome()

    return True

def _write_single_volume(reader: FileReader, args, full_res_shape, io_output_type: str, io_workers: int = 4, output_dtype=None) -> bool:
    """Write a single full-resolution output volume for TIFF or NIfTI targets."""
    if tuple(full_res_shape) != tuple(reader.volume_shape):
        logging.error("resize-shape currently not supported for single outputs. Use input shape.")
        return False

    writer = FileWriter(
        output_path=args.output_path,
        output_name=reader.volume_name,
        output_type=io_output_type,
        full_res_shape=tuple(full_res_shape),
        output_dtype=output_dtype or reader.volume_dtype,
        input_shape=tuple(reader.volume_shape),
        io_workers=io_workers,
    )

    arr = reader.read(z_start=0, z_end=reader.volume_shape[0])
    writer.write(arr, z_start=0, z_end=reader.volume_shape[0])
    del arr
    gc.collect()
    return True

def _write_scroll_slices(reader: FileReader, args, full_res_shape, io_output_type: str, io_workers: int = 4, output_dtype=None) -> bool:
    """Emit individual 2D slices along the selected axis for scroll outputs."""
    if tuple(full_res_shape) != tuple(reader.volume_shape):
        logging.error("resize-shape currently not supported for scroll outputs. Use input shape.")
        return False
        
    scroll_axis = args.scroll_axis
    is_reverse = scroll_axis >= 3
    axis = scroll_axis - 3 if is_reverse else scroll_axis
    
    if axis < 0 or axis > 2:
        logging.error(f"Invalid scroll_axis: {scroll_axis}. Use 0-2 for forward, 3-5 for reverse.")
        return False

    axis_char = ["z", "y", "x"][axis]
    num_slices = reader.volume_shape[axis]
    
    # Calculate the transposed shape for the writer
    if axis == 0:
        writer_shape = reader.volume_shape
    elif axis == 1:
        writer_shape = (reader.volume_shape[1], reader.volume_shape[0], reader.volume_shape[2])
    else:  # axis == 2
        writer_shape = (reader.volume_shape[2], reader.volume_shape[0], reader.volume_shape[1])

    # Base names for the slices
    file_names = [Path(f"{reader.volume_name}_{axis_char}{i:05d}") for i in range(num_slices)]

    writer = FileWriter(
        output_path=args.output_path,
        output_name=reader.volume_name,
        output_type=io_output_type,
        full_res_shape=tuple(writer_shape),
        output_dtype=output_dtype or reader.volume_dtype,
        file_name=file_names,
        input_shape=tuple(writer_shape),
        io_workers=io_workers,
    )

    axis_length = reader.volume_shape[axis]
    axis_handlers = {
        0: lambda start, end: reader.read(z_start=start, z_end=end),
        1: lambda start, end: np.transpose(reader.read(y_start=start, y_end=end), (1, 0, 2)),
        2: lambda start, end: np.transpose(reader.read(x_start=start, x_end=end), (2, 0, 1)),
    }

    handler = axis_handlers[axis]
    step = args.chunk_size

    # Prepare ranges
    ranges = []
    for start in range(0, axis_length, step):
        end = min(start + step, axis_length)
        ranges.append((start, end))
    
    # If reverse, we process chunks from the end of the volume to the start
    # so that file_00000 corresponds to the LAST slice of the volume.
    if is_reverse:
        ranges.reverse()

    current_file_idx = 0
    for start, end in ranges:
        arr = handler(start, end)
        
        if is_reverse:
            # Flip the slices within the chunk so the last slice becomes the first
            arr = np.flip(arr, axis=0)
        
        # Write to the next available file slots
        num_in_chunk = end - start
        writer.write(arr, z_start=current_file_idx, z_end=current_file_idx + num_in_chunk)
        current_file_idx += num_in_chunk
        del arr
        gc.collect()

    return True

def run_task(task_config, full_config):
    """Processes a single task which may contain multiple input/output pairs and formats."""
    resources = full_config.get("resources", {})
    registry = full_config.get("outputs", {})
    io_workers = resources.get("io_workers", 4)
    memory_limit = resources.get("memory_limit", 64)
    
    # Gather input/output pairs
    io_pairs = []
    if "input_path" in task_config and "output_path" in task_config:
        io_pairs.append((task_config["input_path"], task_config["output_path"]))
    
    input_output_dict = task_config.get("input_output")
    if isinstance(input_output_dict, dict):
        for inp, outp in input_output_dict.items():
            io_pairs.append((inp, outp))
            
    if not io_pairs:
        logging.warning("No input/output pairs found for task. Skipping.")
        return

    # Gather output settings
    output_config = task_config.get("output", {})
    ot_raw = output_config.get("type", task_config.get("output_type")) # Fallback to old key for safety
    if not ot_raw:
        logging.error("Missing 'type' in 'output' or top-level 'output_type' in task configuration.")
        return
    output_types = [ot_raw] if isinstance(ot_raw, str) else ot_raw

    for input_path, output_path in io_pairs:
        logging.info(f"Processing input: {input_path}")
        logging.info(f"Output directory: {output_path}")

        try:
            reader = FileReader(
                input_path=input_path,
                memory_limit_gb=memory_limit,
                io_workers=io_workers,
            )
        except Exception as e:
            logging.error(f"Failed to initialize reader for {input_path}: {e}")
            continue

        resize_shape = output_config.get("resize_shape")
        full_res_shape = tuple(resize_shape) if resize_shape else reader.volume_shape
        output_dtype = output_config.get("dtype")
        
        for ot_str in output_types:
            io_output_type = TYPE_MAP.get(ot_str)
            if io_output_type is None:
                logging.error(f"Unsupported output_type: {ot_str}")
                continue

            logging.info(f"  Converting to: {ot_str}")
            Path(output_path).mkdir(parents=True, exist_ok=True)

            # Merge parameters: Registry (global defaults) + Local Output Config
            type_key = io_output_type
            type_defaults = registry.get(type_key, {})
            type_overrides = output_config.get(type_key, {})
            final_type_params = {**type_defaults, **type_overrides}

            chunk_size = final_type_params.get("chunk_size", output_config.get("chunk_size", task_config.get("chunk_size", 128)))
            if isinstance(chunk_size, int):
                chunk_tuple = (chunk_size, chunk_size, chunk_size)
            else:
                chunk_tuple = tuple(chunk_size)

            class ConfigArgs:
                def __init__(self, **entries):
                    self.__dict__.update(entries)
            
            helper_args = ConfigArgs(
                output_path=output_path,
                chunk_size=chunk_size if isinstance(chunk_size, int) else chunk_size[0], 
                levels=final_type_params.get("levels", final_type_params.get("n_level", 5)),
                downscale_factor=final_type_params.get("resize_factor", output_config.get("resize_factor", 2)),
                resize_order=output_config.get("resize_order", 0),
                scroll_axis=final_type_params.get("axis", task_config.get("scroll_axis", 0))
            )

            logging.info(
                f"  Output params: type={ot_str}, dtype={output_dtype}, "
                f"input_shape={reader.volume_shape}, resize_shape={full_res_shape}, "
                f"resize_order={helper_args.resize_order}, chunk_size={chunk_tuple}, "
                f"levels={helper_args.levels}, downscale_factor={helper_args.downscale_factor}, "
                f"scroll_axis={helper_args.scroll_axis}"
            )

            if io_output_type in ["ome-zarr", "zarr"]:
                _write_pyramid(reader, helper_args, full_res_shape, chunk_tuple, io_output_type, io_workers=io_workers, output_dtype=output_dtype)
            elif io_output_type in ["single-tiff", "single-nii"]:
                _write_single_volume(reader, helper_args, full_res_shape, io_output_type, io_workers=io_workers, output_dtype=output_dtype)
            elif io_output_type in ["scroll-tiff", "scroll-nii"]:
                _write_scroll_slices(reader, helper_args, full_res_shape, io_output_type, io_workers=io_workers, output_dtype=output_dtype)
            else:
                logging.error(f"Unsupported internal output_type: {io_output_type}")

def main():
    """Entry point that orchestrates reading, conversion, and writing."""
    args = parse_args()

    with open(args.config, 'r') as f:
        full_config = json.load(f)
        converter_config = full_config.get("converter", [])
    
    # converter_config can be a single dict or a list of dicts
    if isinstance(converter_config, dict):
        tasks = [converter_config]
    elif isinstance(converter_config, list):
        tasks = converter_config
    else:
        logging.error("Invalid 'converter' configuration format. Expected dict or list.")
        return

    logging.info(f"Starting conversion for {len(tasks)} task(s).")
    for i, task in enumerate(tasks):
        logging.info(f"Executing task {i+1}/{len(tasks)}")
        run_task(task, full_config)

    logging.info("All conversion tasks complete.")

if __name__ == "__main__":
    main()
