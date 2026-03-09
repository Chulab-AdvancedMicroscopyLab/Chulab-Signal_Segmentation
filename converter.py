"""
Command-line entry point for converting large 3D volumes into various
output formats (OME-Zarr pyramids, flat Zarr, TIFF/NIfTI volumes, and
per-slice "scroll" exports).

The CLI streams the input volume using `IO.reader.FileReader` and writes
results incrementally via `IO.writer.FileWriter` to keep memory bounded.
"""

import argparse
import logging
import json
from pathlib import Path
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

def _write_pyramid(reader: FileReader, args, full_res_shape, chunk_tuple, io_output_type: str) -> bool:
    """Stream the full-resolution volume into a multiscale Zarr layout."""
    writer = FileWriter(
        output_path=args.output_path,
        output_name=reader.volume_name,
        output_type=io_output_type,
        full_res_shape=tuple(full_res_shape),
        output_dtype=reader.volume_dtype,
        chunk_size=chunk_tuple,
        n_level=args.levels,
        resize_factor=args.downscale_factor,
        resize_order=args.resize_order,
        input_shape=tuple(reader.volume_shape),
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

def _write_single_volume(reader: FileReader, args, full_res_shape, io_output_type: str) -> bool:
    """Write a single full-resolution output volume for TIFF or NIfTI targets."""
    if tuple(full_res_shape) != tuple(reader.volume_shape):
        logging.error("resize-shape currently not supported for single outputs. Use input shape.")
        return False

    writer = FileWriter(
        output_path=args.output_path,
        output_name=reader.volume_name,
        output_type=io_output_type,
        full_res_shape=tuple(full_res_shape),
        output_dtype=reader.volume_dtype,
        input_shape=tuple(reader.volume_shape),
    )

    arr = reader.read(z_start=0, z_end=reader.volume_shape[0])
    writer.write(arr, z_start=0, z_end=reader.volume_shape[0])
    del arr
    return True

def _write_scroll_slices(reader: FileReader, args, full_res_shape, io_output_type: str) -> bool:
    """Emit individual 2D slices along the selected axis for scroll outputs."""
    if tuple(full_res_shape) != tuple(reader.volume_shape):
        logging.error("resize-shape currently not supported for scroll outputs. Use input shape.")
        return False
        
    axis = args.scroll_axis
    axis_char = ["z", "y", "x"][axis]
    num_slices = reader.volume_shape[axis]
    
    # Base names for the slices
    file_names = [Path(f"{reader.volume_name}_{axis_char}{i:05d}") for i in range(num_slices)]

    writer = FileWriter(
        output_path=args.output_path,
        output_name=reader.volume_name,
        output_type=io_output_type,
        full_res_shape=tuple(reader.volume_shape),
        output_dtype=reader.volume_dtype,
        file_name=file_names,
        input_shape=tuple(reader.volume_shape),
    )

    axis_length = reader.volume_shape[axis]
    axis_handlers = {
        0: lambda start, end: reader.read(z_start=start, z_end=end),
        1: lambda start, end: np.transpose(reader.read(y_start=start, y_end=end), (1, 0, 2)),
        2: lambda start, end: np.transpose(reader.read(x_start=start, x_end=end), (2, 0, 1)),
    }

    handler = axis_handlers[axis]
    step = args.chunk_size

    for start in range(0, axis_length, step):
        end = min(start + step, axis_length)
        arr = handler(start, end)
        writer.write(arr, z_start=start, z_end=end)
        del arr

    return True

def main():
    """Entry point that orchestrates reading, conversion, and writing."""
    args = parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f).get("converter", {})

    input_path = config.get("input_path")
    output_path = config.get("output_path")
    output_type_str = config.get("output_type")

    if not input_path or not output_path or not output_type_str:
        logging.error("Missing mandatory arguments in config (input_path, output_path, output_type).")
        return

    logging.info("Starting conversion process.")
    logging.info(f"Input: {input_path}")
    logging.info(f"Output: {output_path} ({output_type_str})")

    memory_limit = config.get("memory_limit", 64)
    transpose = config.get("transpose")
    
    reader = FileReader(
        input_path=input_path,
        memory_limit_gb=memory_limit,
        transpose_order=tuple(transpose) if transpose else None,
    )

    resize_shape = config.get("resize_shape")
    full_res_shape = tuple(resize_shape) if resize_shape else reader.volume_shape
    
    io_output_type = TYPE_MAP.get(output_type_str)
    if io_output_type is None:
        logging.error(f"Unsupported output_type: {output_type_str}")
        return

    Path(output_path).mkdir(parents=True, exist_ok=True)

    chunk_size = config.get("chunk_size", 128)
    chunk_tuple = (chunk_size, chunk_size, chunk_size)

    class ConfigArgs:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    
    helper_args = ConfigArgs(
        output_path=output_path,
        chunk_size=chunk_size,
        levels=config.get("levels", 5),
        downscale_factor=config.get("downscale_factor", 2),
        resize_order=config.get("resize_order", 0),
        scroll_axis=config.get("scroll_axis", 0)
    )

    if io_output_type in ["ome-zarr", "zarr"]:
        _write_pyramid(reader, helper_args, full_res_shape, chunk_tuple, io_output_type)
    elif io_output_type in ["single-tiff", "single-nii"]:
        _write_single_volume(reader, helper_args, full_res_shape, io_output_type)
    elif io_output_type in ["scroll-tiff", "scroll-nii"]:
        _write_scroll_slices(reader, helper_args, full_res_shape, io_output_type)
    else:
        logging.error(f"Unsupported output_type: {output_type_str}")
        return

    logging.info("Conversion complete.")

if __name__ == "__main__":
    main()
