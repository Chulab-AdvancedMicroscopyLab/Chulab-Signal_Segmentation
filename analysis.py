#!/usr/bin/env python3
"""
Robust Analysis: Evaluates predicted masks against Ground Truth.
Dynamically discovers Image, GT, and Prediction triplets without hardcoded names.
"""
import argparse
import logging
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
from tqdm import tqdm

from utils.metrics import accumulate_confusion, compute_metrics, compute_cldice, compute_object_metrics, compute_pr_auc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("analysis")

VALID_EXTS = (".tif", ".tiff")

# Headers for the detailed data
DETAILED_HEADERS = [
    "parent_folder", "image_folder", "model_name", "files_evaluated", "pixels_total", 
    "tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1", "mcc", "hausdorff", "cldice", "obj_f1", "pr_auc", "gt_count", "pr_count"
]

# Headers for the summary rows
SUMMARY_HEADERS = [
    "model_name", "accuracy_mean", "accuracy_std", "precision_mean", "precision_std",
    "recall_mean", "recall_std", "f1_mean", "f1_std", "mcc_mean", "hausdorff_mean", "cldice_mean", "obj_f1_mean", "pr_auc_mean"
]

def list_files(path: Path, exts: Tuple[str, ...] = VALID_EXTS) -> List[Path]:
    if not path.exists(): return []
    allowed = tuple(e.lower() for e in exts)
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in allowed)

def to_binary(arr: np.ndarray) -> np.ndarray:
    return (arr > 0).astype(np.uint8)

def write_results(rows: List[Dict[str, object]], summary_rows: List[Dict[str, object]], out_path: Path):
    try:
        import openpyxl
        from openpyxl import Workbook
        wb = Workbook()
        ws_sum = wb.active
        ws_sum.title = "Model Summary"
        ws_sum.append(SUMMARY_HEADERS)
        for r in summary_rows: ws_sum.append([r.get(k, 0) for k in SUMMARY_HEADERS])
        ws_det = wb.create_sheet("Detailed Evaluation")
        ws_det.append(DETAILED_HEADERS)
        for r in rows: ws_det.append([r.get(k, 0) for k in DETAILED_HEADERS])
        wb.save(str(out_path.with_suffix(".xlsx")))
        logger.info(f"Results saved to {out_path.with_suffix('.xlsx')}")
    except ImportError:
        import csv
        csv_path = out_path.with_suffix(".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_HEADERS)
            w.writeheader()
            for r in rows: w.writerow({k: r.get(k) for k in SUMMARY_HEADERS})
        logger.info(f"Detailed results saved to {csv_path}")

def evaluate_triplet(gt_dir: Path, pred_dir: Path, metric_cfg: dict = None) -> Dict[str, float]:
    metric_cfg = metric_cfg or {}
    # Extract params from config
    hd_p = metric_cfg.get("hausdorff", {}).get("hd_percentile", 95.0)
    obj_p = metric_cfg.get("object_metrics", {})
    min_iou = obj_p.get("min_iou", 0.5)
    min_size = obj_p.get("min_size", 0)
    conn = obj_p.get("connectivity", None)

    gt_files = list_files(gt_dir)
    tp = fp = fn = tn = 0
    file_count = 0
    
    # Trackers for slice-wise complex metrics
    mcc_list = []
    hd_list = []
    cldice_list = []
    obj_f1_list = []
    prauc_list = []
    gt_count_total = 0
    pr_count_total = 0
    
    for gtf in gt_files:
        prf = pred_dir / gtf.name
        if not prf.exists():
            alt = gtf.stem + (".tiff" if gtf.suffix == ".tif" else ".tif")
            prf = pred_dir / alt
        if not prf.exists(): continue
        try:
            g_arr_raw = tifffile.imread(str(gtf))
            p_arr_raw = tifffile.imread(str(prf))
            
            g_arr = to_binary(g_arr_raw)
            p_arr = to_binary(p_arr_raw)
            
            if g_arr.shape != p_arr.shape: continue
            
            # Pixel-wise accumulation
            tp, fp, fn, tn = accumulate_confusion(tp, fp, fn, tn, g_arr, p_arr)
            
            # Slice-wise complex metrics
            slice_metrics = compute_metrics(0, 0, 0, 0, g_arr, p_arr, hd_percentile=hd_p)
            if not np.isnan(slice_metrics["mcc"]): mcc_list.append(slice_metrics["mcc"])
            if not np.isnan(slice_metrics["hausdorff"]): hd_list.append(slice_metrics["hausdorff"])
            
            cldice_list.append(compute_cldice(g_arr, p_arr))
            
            obj_metrics = compute_object_metrics(g_arr, p_arr, min_iou=min_iou, min_size=min_size, connectivity=conn)
            obj_f1_list.append(obj_metrics["obj_f1"])
            gt_count_total += obj_metrics["gt_count"]
            pr_count_total += obj_metrics["pr_count"]
            
            # PR-AUC uses raw scores
            prauc = compute_pr_auc(g_arr, p_arr_raw)
            if prauc > 0: prauc_list.append(prauc)
            
            file_count += 1
        except Exception as e:
            logger.warning(f"Error reading {gtf.name}: {e}")
            
    metrics = compute_metrics(tp, fp, fn, tn)
    metrics["files_evaluated"] = file_count
    metrics["mcc"] = np.mean(mcc_list) if mcc_list else 0.0
    metrics["hausdorff"] = np.mean(hd_list) if hd_list else 0.0
    metrics["cldice"] = np.mean(cldice_list) if cldice_list else 0.0
    metrics["obj_f1"] = np.mean(obj_f1_list) if obj_f1_list else 0.0
    metrics["pr_auc"] = np.mean(prauc_list) if prauc_list else 0.0
    metrics["gt_count"] = gt_count_total
    metrics["pr_count"] = pr_count_total
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Robustly evaluate masks.")
    parser.add_argument("--base_dir", type=str, required=True, help="Root directory to search")
    parser.add_argument("--output_name", type=str, default="evaluation_report", help="Output filename")
    parser.add_argument("--config", type=str, default=None, help="Path to config file for metric parameters")
    args = parser.parse_args()

    metric_cfg = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                metric_cfg = json.load(f).get("metrics", {})
            logger.info(f"Loaded metric configuration from {args.config}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")

    root = Path(args.base_dir).resolve()
    all_results = []

    # 1. Find potential GT folders (ending in _mask and NOT .scroll-tif)
    gt_folders = [
        p for p in root.rglob("*_mask") 
        if p.is_dir() and not p.name.endswith(".scroll-tif")
    ]
    
    logger.info(f"Found {len(gt_folders)} GT volumes to analyze.")

    for gt_dir in tqdm(gt_folders, desc="GT Folders"):
        parent = gt_dir.parent
        # The image folder is the one that shares the same prefix as the mask folder
        # e.g. Flatten_561_mask -> Flatten_561
        img_folder_name = gt_dir.name.replace("_mask", "")
        img_dir = parent / img_folder_name
        
        if not img_dir.exists() or not img_dir.is_dir():
            # Fallback: find ANY other folder in parent that isn't a mask or scroll-tif
            other_dirs = [d for d in parent.iterdir() if d.is_dir() and d != gt_dir and "_mask" not in d.name]
            if other_dirs: img_dir = other_dirs[0]
            else: continue

        # 2. Find prediction folders in the same parent
        # They should contain the model name and end with .scroll-tif
        pred_folders = [
            p for p in parent.iterdir() 
            if p.is_dir() and p.name.strip().endswith(".scroll-tif")
        ]
        
        for p_dir in pred_folders:
            p_name = p_dir.name.strip()
            # Try to extract model name: {img_name}_{model}_mask.scroll-tif
            # Robust extraction using regex
            model_match = re.search(f"{img_dir.name}_(.*)_mask", p_name)
            if model_match:
                model_name = model_match.group(1)
            else:
                # Fallback: just remove prefix and suffix
                model_name = p_name.replace(img_dir.name, "").replace("_mask.scroll-tif", "").strip("_")
            
            metrics = evaluate_triplet(gt_dir, p_dir, metric_cfg=metric_cfg)
            if metrics["files_evaluated"] > 0:
                all_results.append({
                    "parent_folder": parent.name,
                    "image_folder": img_dir.name,
                    "model_name": model_name,
                    **metrics
                })

    if not all_results:
        logger.warning("No matching GT/Prediction pairs found."); return

    # Horizontal Summary Aggregation
    model_names = sorted(set(r["model_name"] for r in all_results))
    summary_rows = []
    metrics_to_agg = ["accuracy", "precision", "recall", "f1", "mcc", "hausdorff", "cldice", "obj_f1", "pr_auc"]
    
    for m_name in model_names:
        m_results = [r for r in all_results if r["model_name"] == m_name]
        row = {"model_name": m_name}
        for metric in metrics_to_agg:
            values = [r[metric] for r in m_results]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        summary_rows.append(row)

    write_results(all_results, summary_rows, root / args.output_name)

if __name__ == "__main__":
    main()
