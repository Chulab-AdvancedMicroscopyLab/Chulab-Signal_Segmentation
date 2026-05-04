import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

# -----------------------------------------------------------------------------
# Torch-based Metrics (Used by Trainer)
# -----------------------------------------------------------------------------

def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
    from_logits: bool = False,
    dims: tuple = None,
) -> torch.Tensor:
    """
    Computes the Soft Dice Score (differentiable-like).
    Does NOT threshold at 0.5, giving a smoother metric.
    """
    if from_logits:
        pred = pred.sigmoid()
    
    # Use soft probabilities for a more stable metric
    target = target.float()

    if dims is None:
        dims = tuple(range(1, pred.dim()))

    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()

def hard_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
    from_logits: bool = False,
    dims: tuple = None,
) -> torch.Tensor:
    """
    Computes the Hard Dice Score (non-differentiable).
    Thresholds prediction at 0.5 before computing.
    """
    if from_logits:
        pred = pred.sigmoid()
    
    # Binarize predictions and target
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    if dims is None:
        dims = tuple(range(1, pred.dim()))

    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()

def bce_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Binary Cross Entropy Score (Metric Only).
    Calculates standard BCE on probabilities.
    """
    if from_logits:
        pred = pred.sigmoid()
    
    target = target.float()
    
    # Clamp to avoid log(0)
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1.0 - eps)
    
    loss = F.binary_cross_entropy(pred, target, reduction=reduction)
    return loss

def hybrid_focal_tversky_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates a hybrid Focal + Tversky loss to handle extreme class imbalance.
    Tversky loss allows us to weight False Negatives more heavily to improve Recall.
    """
    # 1. Focal Loss (Stable implementation)
    gamma = 2.0
    alpha = 0.25
    
    # Binary Cross Entropy with Logits
    bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
    pt = torch.exp(-bce) # probability of the correct class
    focal_loss = (alpha * (1 - pt)**gamma * bce).mean()

    # 2. Tversky Loss (Differentiable)
    # alpha=0.3, beta=0.7 weights False Negatives more (improves Recall)
    # alpha + beta = 1.0 (standard Dice is 0.5/0.5)
    t_alpha = 0.5
    t_beta = 0.5
    
    pred_soft = pred.sigmoid()
    target_soft = target.float()
    
    dims = tuple(range(1, pred_soft.dim()))
    tp = (pred_soft * target_soft).sum(dim=dims)
    fp = (pred_soft * (1 - target_soft)).sum(dim=dims)
    fn = ((1 - pred_soft) * target_soft).sum(dim=dims)
    
    smooth = 1e-5
    tversky_index = (tp + smooth) / (tp + t_alpha * fp + t_beta * fn + smooth)
    tversky_loss = (1.0 - tversky_index).mean()

    # Hybrid weighting (Focus on Tversky for sparse signal)
    return 0.2 * focal_loss + 0.8 * tversky_loss

# -----------------------------------------------------------------------------
# Numpy-based Metrics (Used by Analysis)
# -----------------------------------------------------------------------------

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

def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Computes common metrics from confusion matrix."""
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return {
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
