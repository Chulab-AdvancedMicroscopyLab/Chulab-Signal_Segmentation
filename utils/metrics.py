import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Union, Callable, Optional
from sklearn.metrics import average_precision_score
from skimage.metrics import hausdorff_distance
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

# -----------------------------------------------------------------------------
# Torch-based Metrics (Used by Trainer)
# -----------------------------------------------------------------------------

class DiceScore(nn.Module):
    def __init__(self, smooth: float = 1e-5, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred = pred.sigmoid()
        
        target = target.float()
        dims = tuple(range(1, pred.dim()))

        intersection = (pred * target).sum(dim=dims)
        union = pred.sum(dim=dims) + target.sum(dim=dims)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice.mean()

class HardDiceScore(nn.Module):
    def __init__(self, smooth: float = 1e-5, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred = pred.sigmoid()
        
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        dims = tuple(range(1, pred.dim()))

        intersection = (pred * target).sum(dim=dims)
        union = pred.sum(dim=dims) + target.sum(dim=dims)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice.mean()

class BCEScore(nn.Module):
    def __init__(self, from_logits: bool = True, reduction: str = "mean"):
        super().__init__()
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred = pred.sigmoid()
        
        target = target.float()
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1.0 - eps)
        
        return F.binary_cross_entropy(pred, target, reduction=self.reduction)

class IoUScore(nn.Module):
    def __init__(self, smooth: float = 1e-5, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred = pred.sigmoid()
        
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        dims = tuple(range(1, pred.dim()))

        intersection = (pred * target).sum(dim=dims)
        union = pred.sum(dim=dims) + target.sum(dim=dims)
        iou = (intersection + self.smooth) / (union - intersection + self.smooth)
        return iou.mean()

from concurrent.futures import ThreadPoolExecutor

class NumpyMetricWrapper:
    """Wraps a numpy-based metric function to be callable with tensors, using parallel execution."""
    def __init__(self, func: Callable, threshold: Optional[float] = 0.5, needs_sigmoid: bool = True, num_workers: int = 8, **kwargs):
        self.func = func
        self.threshold = threshold
        self.needs_sigmoid = needs_sigmoid
        self.num_workers = num_workers
        self.kwargs = kwargs
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.needs_sigmoid:
            pred = pred.sigmoid()
        
        # Move to CPU and convert to numpy
        if self.threshold is not None:
            p_np = (pred.detach().cpu().numpy() > self.threshold).astype(np.uint8)
        else:
            p_np = pred.detach().cpu().numpy()
            
        t_np = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)
        
        batch_size = p_np.shape[0]
        
        def _run_single(i):
            try:
                res = self.func(t_np[i].squeeze(), p_np[i].squeeze(), **self.kwargs)
                return res if not np.isnan(res) else None
            except Exception:
                return None

        # Execute batch in parallel
        results = list(self._executor.map(_run_single, range(batch_size)))
        batch_results = [r for r in results if r is not None]
        
        if not batch_results:
            return torch.tensor(0.0)
        return torch.tensor(np.mean(batch_results))

def build_metrics_from_config(full_config: dict) -> Dict[str, Union[nn.Module, NumpyMetricWrapper]]:
    """
    Builds a dictionary of metric functions from config.
    """
    metric_params = full_config.get("metrics", {})
    train_config = full_config.get("train", {})
    resources = full_config.get("resources", {})
    
    # Priority: metrics_workers -> io_workers -> default 4
    metric_workers = resources.get("metrics_workers", resources.get("io_workers", 4))

    metrics_to_use = train_config.get("metrics", ["dice_soft", "dice_hard"])

    if isinstance(metrics_to_use, str):
        metrics_to_use = [metrics_to_use]

    # Registry of available metrics
    registry = {
        "dice_soft": lambda cfg: DiceScore(smooth=cfg.get("smooth", 1e-5)),
        "dice_hard": lambda cfg: HardDiceScore(smooth=cfg.get("smooth", 1e-5)),
        "iou": lambda cfg: IoUScore(smooth=cfg.get("smooth", 1e-5)),
        "bce": lambda _: BCEScore(),
        "mcc": lambda _: NumpyMetricWrapper(compute_mcc, num_workers=metric_workers),
        "cldice": lambda _: NumpyMetricWrapper(compute_cldice, num_workers=metric_workers),
        "hausdorff": lambda cfg: NumpyMetricWrapper(compute_robust_hausdorff, percentile=cfg.get("hd_percentile", 95.0), num_workers=metric_workers),
        "pr_auc": lambda _: NumpyMetricWrapper(compute_pr_auc, threshold=None, num_workers=metric_workers)
    }

    built_metrics = {}
    for name in metrics_to_use:
        name_lower = name.lower()
        if name_lower in registry:
            params = metric_params.get(name_lower, {})
            built_metrics[name_lower] = registry[name_lower](params)
        else:
            # Fallback for common names, but avoid mapping known numpy metrics to dice
            if name_lower in ["mcc", "cldice", "hausdorff", "hd95"]:
                continue # Should have been caught by registry if defined
                
            if "dice" in name_lower and "hard" not in name_lower:
                built_metrics[name_lower] = registry["dice_soft"]({})
            elif "dice" in name_lower and "hard" in name_lower:
                built_metrics[name_lower] = registry["dice_hard"]({})
            else:
                print(f"Warning: Unknown metric requested: {name}")

    return built_metrics


# -----------------------------------------------------------------------------
# Numpy-based Metrics (Used by Analysis)
# -----------------------------------------------------------------------------

def compute_mcc(gt: np.ndarray, pr: np.ndarray) -> float:
    """
    Calculates Matthews Correlation Coefficient safely.
    Avoids sklearn warnings when one or both classes are missing in a patch.
    """
    # Cast to float64 to prevent overflow in large volumes
    tp = np.sum((pr == 1) & (gt == 1)).astype(np.float64)
    fp = np.sum((pr == 1) & (gt == 0)).astype(np.float64)
    fn = np.sum((pr == 0) & (gt == 1)).astype(np.float64)
    tn = np.sum((pr == 0) & (gt == 0)).astype(np.float64)
    
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)

def confusion_from_arrays(gt: np.ndarray, pr: np.ndarray) -> Tuple[int, int, int, int]:
    """Calculates TP, FP, FN, TN for binary masks."""
    tp = int(np.sum((pr == 1) & (gt == 1)))
    fp = int(np.sum((pr == 1) & (gt == 0)))
    fn = int(np.sum((pr == 0) & (gt == 1)))
    tn = int(np.sum((pr == 0) & (gt == 0)))
    return tp, fp, fn, tn

def accumulate_confusion(tp: int, fp: int, fn: int, tn: int, gt: np.ndarray, pr: np.ndarray) -> Tuple[int, int, int, int]:
    """Accumulates confusion matrix values."""
    ctp, cfp, cfn, ctn = confusion_from_arrays(gt, pr)
    return tp + ctp, fp + cfp, fn + cfn, tn + ctn

def compute_metrics(tp: int, fp: int, fn: int, tn: int, gt: np.ndarray = None, pr: np.ndarray = None, hd_percentile: float = 95.0) -> Dict[str, float]:
    """Computes common metrics from confusion matrix."""
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "total": float(total),
    }

    if gt is not None and pr is not None:
        # Matthews Correlation Coefficient
        metrics["mcc"] = float(compute_mcc(gt, pr))
        
        # Robust Hausdorff Distance (HD95 by default)
        if np.any(gt) and np.any(pr):
            metrics["hausdorff"] = float(compute_robust_hausdorff(gt, pr, hd_percentile))
        else:
            metrics["hausdorff"] = float('nan')

    return metrics

def compute_robust_hausdorff(gt: np.ndarray, pr: np.ndarray, percentile: float = 95.0) -> float:
    """Computes percentile-based Hausdorff Distance (e.g., HD95)."""
    if not np.any(gt) or not np.any(pr):
        return float('nan')
    
    # Distance from pr points to gt surface
    dt_gt = distance_transform_edt(1 - gt)
    dist_pr_to_gt = dt_gt[pr == 1]
    
    # Distance from gt points to pr surface
    dt_pr = distance_transform_edt(1 - pr)
    dist_gt_to_pr = dt_pr[gt == 1]
    
    hd = max(np.percentile(dist_pr_to_gt, percentile), np.percentile(dist_gt_to_pr, percentile))
    return hd

def compute_pr_auc(gt: np.ndarray, pr_scores: np.ndarray) -> float:
    """Computes Pixel-wise Precision-Recall AUC (Average Precision)."""
    if not np.any(gt):
        return 0.0
    
    y_true = gt.flatten()
    y_scores = pr_scores.flatten()
    
    # If binary, AP doesn't provide a curve
    if np.unique(y_scores).size <= 2:
        return 0.0
        
    if y_scores.max() > 1.0:
        y_scores = y_scores.astype(np.float32) / 255.0
        
    return float(average_precision_score(y_true, y_scores))


def compute_cldice(gt: np.ndarray, pr: np.ndarray) -> float:
    """Computes Centerline Dice (clDice) for connectivity."""
    if not np.any(gt) or not np.any(pr):
        return 0.0
    
    # Note: skeletonize is already slice-wise or volume-wise depending on input dims
    t_skel = skeletonize(gt)
    p_skel = skeletonize(pr)
    
    tprec = np.sum(t_skel * pr) / (np.sum(t_skel) + 1e-6)
    psens = np.sum(p_skel * gt) / (np.sum(p_skel) + 1e-6)
    
    return 2 * (tprec * psens) / (tprec + psens) if (tprec + psens) else 0.0

def compute_object_metrics(gt: np.ndarray, pr: np.ndarray, min_iou: float = 0.5, min_size: int = 0, connectivity: int = None) -> Dict[str, float]:
    """Computes object-level detection metrics with noise filtering."""
    gt_labels = label(gt, connectivity=connectivity)
    pr_labels = label(pr, connectivity=connectivity)
    
    # Filter small objects (noise) if min_size > 0
    if min_size > 0:
        for i in range(1, gt_labels.max() + 1):
            if np.sum(gt_labels == i) < min_size:
                gt_labels[gt_labels == i] = 0
        for i in range(1, pr_labels.max() + 1):
            if np.sum(pr_labels == i) < min_size:
                pr_labels[pr_labels == i] = 0
        
        # Relabel after filtering
        gt_labels = label(gt_labels > 0, connectivity=connectivity)
        pr_labels = label(pr_labels > 0, connectivity=connectivity)

    num_gt = gt_labels.max()
    num_pr = pr_labels.max()
    
    if num_gt == 0:
        return {"obj_f1": 0.0, "obj_precision": 0.0, "obj_recall": 0.0, "gt_count": 0, "pr_count": num_pr}
    
    if num_pr == 0:
        return {"obj_f1": 0.0, "obj_precision": 0.0, "obj_recall": 0.0, "gt_count": num_gt, "pr_count": 0}

    # Match objects based on IoU
    tp = 0
    matched_gt = set()
    
    for i in range(1, num_pr + 1):
        pr_mask = (pr_labels == i)
        overlapping_gt_ids = np.unique(gt_labels[pr_mask])
        for gt_id in overlapping_gt_ids:
            if gt_id == 0 or gt_id in matched_gt:
                continue
            
            gt_mask = (gt_labels == gt_id)
            intersection = np.logical_and(pr_mask, gt_mask).sum()
            union = np.logical_or(pr_mask, gt_mask).sum()
            iou = intersection / union
            
            if iou >= min_iou:
                tp += 1
                matched_gt.add(gt_id)
                break
                
    fp = num_pr - tp
    fn = num_gt - tp
    
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    
    return {
        "obj_f1": float(f1),
        "obj_precision": float(prec),
        "obj_recall": float(rec),
        "gt_count": int(num_gt),
        "pr_count": int(num_pr)
    }
