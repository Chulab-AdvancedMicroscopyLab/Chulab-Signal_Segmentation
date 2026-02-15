"""
Robust Analysis: Evaluates predicted masks against Ground Truth.
Dynamically discovers Image, GT, and Prediction triplets without hardcoded names.
"""
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
from tqdm import tqdm

from utils.metrics import accumulate_confusion, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("analysis")

VALID_EXTS = (".tif", ".tiff")

# Headers for the detailed data
DETAILED_HEADERS = [
    "parent_folder", "image_folder", "model_name", "files_evaluated", "pixels_total", 
    "tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1"
]

# Headers for the summary rows
SUMMARY_HEADERS = [
    "model_name", "accuracy_mean", "accuracy_std", "precision_mean", "precision_std",
    "recall_mean", "recall_std", "f1_mean", "f1_std"
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

def evaluate_triplet(gt_dir: Path, pred_dir: Path) -> Dict[str, float]:
    gt_files = list_files(gt_dir)
    tp = fp = fn = tn = 0
    file_count = 0
    for gtf in gt_files:
        prf = pred_dir / gtf.name
        if not prf.exists():
            alt = gtf.stem + (".tiff" if gtf.suffix == ".tif" else ".tif")
            prf = pred_dir / alt
        if not prf.exists(): continue
        try:
            g_arr = to_binary(tifffile.imread(str(gtf)))
            p_arr = to_binary(tifffile.imread(str(prf)))
            if g_arr.shape != p_arr.shape: continue
            tp, fp, fn, tn = accumulate_confusion(tp, fp, fn, tn, g_arr, p_arr)
            file_count += 1
        except Exception as e:
            logger.warning(f"Error reading {gtf.name}: {e}")
    metrics = compute_metrics(tp, fp, fn, tn)
    metrics["files_evaluated"] = file_count
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Robustly evaluate masks.")
    parser.add_argument("--base_dir", type=str, required=True, help="Root directory to search")
    parser.add_argument("--output_name", type=str, default="evaluation_report", help="Output filename")
    args = parser.parse_args()

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
            
            metrics = evaluate_triplet(gt_dir, p_dir)
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
    metrics_to_agg = ["accuracy", "precision", "recall", "f1"]
    
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
