#!/usr/bin/env python3
"""
Train a GUSLModel using config_GUSL.json.

Usage:
    python train_GUSL.py --config configs/config_GUSL.json
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

sys.path.insert(0, os.path.dirname(__file__))

from IO.datasets import GUSLDataset
from models.GUSL import GUSLModel


def _add_file_logger(log_path: str) -> None:
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config_GUSL.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    train_cfg      = cfg["train"]
    gusl_model_cfg = cfg["model"]["gusl"]
    gusl_train_cfg = train_cfg.get("gusl", {})

    device_str = train_cfg.get("device", "cuda:0")
    if ":" in device_str:
        device, gpu_id = device_str.split(":")
        gpu_id = int(gpu_id)
    else:
        device, gpu_id = device_str, 0

    patch_size  = tuple(train_cfg.get("training_patch_size", [32, 256, 256]))
    patch_depth = patch_size[0]
    base_size   = patch_size[1]  # auto-derived from patch HW

    saveroot = os.path.join(train_cfg["save_path"], train_cfg.get("model_name", "GUSL_v0"))
    os.makedirs(saveroot, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(saveroot, f"train_{timestamp}.log")
    _add_file_logger(log_path)

    logging.info(f"Config: {args.config}")
    logging.info(f"saveroot: {saveroot}")
    logging.info(f"log: {log_path}")

    # --- Build dataset ---
    logging.info("[Data] Building GUSLDataset from config...")
    full_ds = GUSLDataset.from_config(cfg)
    train_ds, val_ds = full_ds.split(
        patch_depth=patch_depth,
        val_ratio=train_cfg.get("val_ratio", 0.2),
        seed=train_cfg.get("seed", 42),
    )
    logging.info(f"[Data] train={train_ds.imgs.shape[0]} frames, val={val_ds.imgs.shape[0]} frames")

    # --- Build model ---
    model = GUSLModel(
        deepest_level=gusl_model_cfg.get("deepest_level", 4),
        base_size=base_size,
        kernel_size=gusl_model_cfg.get("kernel_size",   [3, 3, 5, 5]),
        kernel_depth=gusl_model_cfg.get("kernel_depth",  [3, 3, 3, 3]),
        neigh_size=gusl_model_cfg.get("neigh_size",    [3, 5, 5, 7]),
        neigh_depth=gusl_model_cfg.get("neigh_depth",   [1, 1, 1, 1]),
        neigh_stride=gusl_model_cfg.get("neigh_stride",  [2, 2, 2, 2]),
        use_grad=gusl_model_cfg.get("use_grad",      [True, True, True, True]),
        grad_size=gusl_model_cfg.get("grad_size",     [3, 5, 5, 5]),
        grad_depth=gusl_model_cfg.get("grad_depth",    [3, 3, 3, 3]),
        batch_centers=gusl_model_cfg.get("batch_centers", 100_000),
        n_bins=gusl_model_cfg.get("n_bins",       [32, 32, 32, 32]),
        n_selected=gusl_model_cfg.get("n_selected",  [400, 500, 600, 600]),
        lnt_depth=gusl_model_cfg.get("lnt_depth",    [3, 3, 4, 4]),
        lnt_num_tree=gusl_model_cfg.get("lnt_num_tree", [150, 150, 200, 200]),
        rft_sample_frac=gusl_model_cfg.get("rft_sample_frac", 0.5),
        lnt_sample_frac=gusl_model_cfg.get("lnt_sample_frac", 0.5),
        n_estimators=gusl_model_cfg.get("n_estimators",         5000),
        max_depth=gusl_model_cfg.get("max_depth",            4),
        learning_rate=gusl_model_cfg.get("learning_rate",        0.15),
        early_stopping_rounds=gusl_model_cfg.get("early_stopping_rounds", 5),
        keep_frac=gusl_train_cfg.get("keep_frac", 0.3),
        low1=gusl_train_cfg.get("low1",  0.0),
        low2=gusl_train_cfg.get("low2",  0.05),
        high1=gusl_train_cfg.get("high1", 0.1),
        high2=gusl_train_cfg.get("high2", 0.4),
        encode_percentage=gusl_train_cfg.get("encode_percentage",     [1.0, 1.0, 0.4, 0.1]),
        decode_percentage=gusl_train_cfg.get("decode_percentage",     [1.0, 1.0, 0.7, 0.3]),
        val_encode_percentage=gusl_train_cfg.get("val_encode_percentage", [1.0, 1.0, 1.0, 1.0]),
        val_decode_percentage=gusl_train_cfg.get("val_decode_percentage", [1.0, 1.0, 1.0, 1.0]),
        patch_depth=patch_depth,
        batch_size=gusl_train_cfg.get("batch_size", 2),
        device=device,
        gpu_id=gpu_id,
    )

    # --- Train ---
    model.fit(
        train_imgs=train_ds.imgs, train_msks=train_ds.msks,
        val_imgs=val_ds.imgs,     val_msks=val_ds.msks,
        saveroot=saveroot,
        train_names=train_ds.names,
        val_names=val_ds.names,
        run_level=train_cfg.get("run_level"),
    )

    # --- Save ---
    model_path = os.path.join(saveroot, "gusl_model.joblib")
    model.save(model_path)
    logging.info(f"[Done] Model saved to {model_path}")


if __name__ == "__main__":
    main()
