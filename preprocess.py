#!/usr/bin/env python3
import argparse
import numpy as np
import tifffile
import json
import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm

from IO.reader import FileReader
from utils.normalization import build_normalizer_from_config
from utils.concurrency import initialize_concurrency

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Preprocess a TIFF stack using configuration-driven parameters.")
    parser.add_argument("--config", type=str, required=True, help="Path to the project config file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the directory containing 2D TIFF slices.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the normalized TIFF slices.")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train", 
                        help="Which configuration section to use for preprocessing settings (default: inference).")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        full_config = json.load(f)
    
    # Initialize concurrency settings
    initialize_concurrency(full_config)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get preprocess settings for the chosen mode
    mode_config = full_config.get(args.mode, {})
    preprocess_config = mode_config.get("preprocess", {})
    
    if not preprocess_config:
        logger.warning(f"No 'preprocess' block found in '{args.mode}' section. Using defaults.")

    # 1. Initialize FileReader
    logger.info(f"Reading volume info from {input_path}...")
    
    method = preprocess_config.get("normalize_mode", "z-score")
    reader = FileReader(
        input_path, 
        compute_stats=True, 
        stats_sample_rate=preprocess_config.get("sample_rate", 1.0),
        low_cut=preprocess_config.get("low_cut"),
        high_cut=preprocess_config.get("high_cut"),
        compute_histogram=(method == "histogram"),
        hist_bins=preprocess_config.get("bins", 1024)
    )

    # 2. Build Normalizer
    normalizer = build_normalizer_from_config(full_config, reader, mode=args.mode)

    logger.info("-" * 30)
    logger.info(f"Mode:   {args.mode}")
    logger.info(f"Method: {normalizer.method}")
    if normalizer.method == "z-score":
        logger.info(f"Mean:   {normalizer.mean:.6f}")
        logger.info(f"Std:    {normalizer.std:.6f} (Multiplier: {normalizer.std_multiplier})")
    elif normalizer.method in ("min-max", "min-max-gamma"):
        logger.info(f"Min:    {normalizer.min_v:.6f}")
        logger.info(f"Max:    {normalizer.max_v:.6f}")
        if normalizer.method == "min-max-gamma":
            logger.info(f"Gamma:  {normalizer.gamma:.4f}")
    
    if preprocess_config.get("low_cut") is not None:
        logger.info(f"Low Cut:  {preprocess_config.get('low_cut')}")
    if preprocess_config.get("high_cut") is not None:
        logger.info(f"High Cut: {preprocess_config.get('high_cut')}")
    logger.info("-" * 30)

    # 3. Process and Save
    chunk_size = 128
    z_max = reader.volume_shape[0]
    filenames = reader.volume_files

    logger.info(f"Preprocessing {z_max} slices...")
    for z0 in range(0, z_max, chunk_size):
        z1 = min(z0 + chunk_size, z_max)

        # Read a block
        data = reader.read(z_start=z0, z_end=z1, out_dtype=np.float32)

        # Apply normalization
        normalizer(data)
        
        # Save each slice
        for i in range(data.shape[0]):
            global_z = z0 + i
            original_file = filenames[global_z]
            save_file = output_path / original_file.name
            tifffile.imwrite(save_file, data[i])

    logger.info(f"Success! Preprocessed TIFFs saved to: {output_path}")

if __name__ == "__main__":
    main()
