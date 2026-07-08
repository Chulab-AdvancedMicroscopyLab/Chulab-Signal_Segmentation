#!/usr/bin/env python3
"""
Run GUSLModel inference using config_GUSL.json.

Usage:
    python inference_GUSL.py --config configs/config_GUSL.json

Architecture:
  - Z-chunked via compute_z_plan to bound RAM
  - Full XY passed to model.predict() — features extracted once per level,
    streamed in pixel batches (inference_pixel_batch) through RFT → LNT → XGBoost
  - Writes output via FileWriter
"""
import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from IO.reader import FileReader
from IO.writer import FileWriter
from IO.IO_types import TYPE_MAP
from models.GUSL import GUSLModel
from utils.cropper import compute_z_plan
from utils.normalization import build_normalizer_from_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config_GUSL.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    inf_cfg = cfg["inference"]
    out_cfg = inf_cfg.get("output", {})

    # --- Device ---
    device_str = inf_cfg.get("device", "cuda:0")
    device, gpu_id = (device_str.split(":") + ["0"])[:2]
    gpu_id = int(gpu_id)

    # --- Load model ---
    model_path = inf_cfg["model_path"]
    logger.info(f"[IO] Loading model: {model_path}")
    model: GUSLModel = GUSLModel.load(model_path)
    model.set_device(device_str)
    model.gpu_id = gpu_id

    # --- Z-chunk geometry ---
    z_chunk         = inf_cfg.get("z_chunk_depth", 16)
    z_overlap       = inf_cfg.get("z_chunk_overlap", 4)
    pixel_batch     = inf_cfg.get("inference_pixel_batch", 500_000)

    # --- Open reader ---
    input_path = inf_cfg["input_path"]
    input_name = inf_cfg.get("input_name", "")
    if input_name:
        import pathlib
        vol_path = str(pathlib.Path(input_path) / input_name)
    else:
        vol_path = input_path

    logger.info(f"[IO] Opening: {vol_path}")
    reader = FileReader(
        vol_path,
        io_workers=cfg.get("resources", {}).get("io_workers", 4),
        compute_stats=True,
        stats_sample_rate=inf_cfg.get("preprocess", {}).get("sample_rate", 0.1),
        low_cut=inf_cfg.get("preprocess", {}).get("low_cut"),
        high_cut=inf_cfg.get("preprocess", {}).get("high_cut"),
        compute_histogram=(inf_cfg.get("preprocess", {}).get("normalize_mode", "z-score") == "histogram"),
    )
    Z, H, W = reader.volume_shape
    logger.info(f"[IO] Volume shape: {reader.volume_shape}")

    normalizer = build_normalizer_from_config(cfg, reader, mode="inference")

    # --- Writer ---
    output_path = inf_cfg["output_path"]
    output_name = inf_cfg.get("output_name", reader.volume_name + "_gusl_pred")
    output_type = TYPE_MAP.get(out_cfg.get("type", "Tiff"), "single-tiff")
    out_dtype   = np.dtype(out_cfg.get("dtype", "uint8"))

    os.makedirs(output_path, exist_ok=True)
    writer = FileWriter(
        output_path=output_path,
        output_name=output_name,
        output_type=output_type,
        full_res_shape=(Z, H, W),
        output_dtype=out_dtype,
        file_name=reader.volume_files,
    )

    # --- Z-chunk loop ---
    z_plan = compute_z_plan(volume_depth=Z, patch_depth=z_chunk, z_overlap=z_overlap)
    logger.info(f"[Inference] {len(z_plan)} Z-chunks of depth={z_chunk}, overlap={z_overlap}")

    for chunk_idx, (z_start, _) in enumerate(z_plan):
        z_end = min(z_start + z_chunk, Z)
        logger.info(f"[Inference] Chunk {chunk_idx+1}/{len(z_plan)}: z=[{z_start},{z_end}]")

        raw = reader.read(z_start=z_start, z_end=z_end).astype(np.float32)
        imgs = normalizer(raw)
        del raw

        preds = model.predict(imgs, pixel_batch_size=pixel_batch)
        del imgs

        threshold = out_cfg.get("threshold", None)
        if threshold is not None:
            output = ((preds >= threshold) * 255).astype(out_dtype)
        else:
            output = np.clip(preds, 0, 255).astype(out_dtype)
        writer.write(output, z_start=z_start, z_end=z_end)
        del preds, output

    logger.info(f"[Done] Predictions written to {output_path}")


if __name__ == "__main__":
    main()
