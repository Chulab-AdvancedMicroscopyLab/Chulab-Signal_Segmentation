import logging
import numpy as np
import cv2
from typing import Dict, Tuple, List


def boundary_selection_mask(
    imgs: np.ndarray,
    msks: np.ndarray,
    window: int = 5,
    low: float = 0.1,
    high: float = 1,
    binarize_thresh: float = 1.0,
    img_window: int = 5,
    intensity_ranges: list = ((0.1, 0.4), (0.0, 0.05)),
    keep_fracs: list = (0.6, 0.6),
    seed: int = 42,
) -> np.ndarray:
    assert msks.ndim == 3
    assert window % 2 == 1
    assert img_window % 2 == 1
    assert 0.0 <= low <= high <= 1.0
    assert len(intensity_ranges) == len(keep_fracs)

    rng = np.random.default_rng(seed)

    if imgs.ndim == 4:
        x = imgs.astype(np.float32).mean(axis=-1)
    elif imgs.ndim == 3:
        x = imgs.astype(np.float32)
    else:
        raise ValueError("imgs must be (N,H,W) or (N,H,W,C)")

    N, H, W = x.shape
    x_norm = np.empty_like(x, dtype=np.float32)
    for i in range(N):
        xi = x[i]
        mn, mx = float(xi.min()), float(xi.max())
        x_norm[i] = (xi - mn) / (mx - mn) if mx > mn else np.zeros_like(xi, dtype=np.float32)

    bin_msks = (msks > binarize_thresh).astype(np.float32)

    local_means_mask = np.empty_like(bin_msks, dtype=np.float32)
    for i in range(N):
        local_means_mask[i] = cv2.blur(bin_msks[i], (window, window), borderType=cv2.BORDER_REFLECT)

    boundary = (local_means_mask >= low) & (local_means_mask <= high)
    positives = (bin_msks == 1)
    negatives = ~positives
    sel = positives | (negatives & boundary)

    local_means_img = np.empty_like(x_norm, dtype=np.float32)
    for i in range(N):
        local_means_img[i] = cv2.blur(x_norm[i], (img_window, img_window), borderType=cv2.BORDER_REFLECT)

    remaining = ~sel

    for (lo, hi), frac in zip(intensity_ranges, keep_fracs):
        if frac <= 0.0:
            continue
        for i in range(N):
            cand = remaining[i] & (local_means_img[i] >= lo) & (local_means_img[i] <= hi)
            idxs = np.flatnonzero(cand)
            if idxs.size == 0:
                continue
            k_keep = int(np.floor(frac * idxs.size))
            if k_keep <= 0:
                continue
            chosen = rng.choice(idxs, size=k_keep, replace=False)
            flat = sel[i].ravel()
            flat[chosen] = True
            sel[i] = flat.reshape(H, W)
            flat_rem = remaining[i].ravel()
            flat_rem[chosen] = False
            remaining[i] = flat_rem.reshape(H, W)

    return sel.astype(np.uint8)


def pos_neg_counts(msks: np.ndarray, sel_mask_flat: np.ndarray) -> Dict[str, int]:
    pos_flat = (msks > 0).reshape(-1).astype(bool)
    pos_total = int(pos_flat.sum())
    neg_total = int(pos_flat.size - pos_total)
    sel = sel_mask_flat.astype(bool)
    pos_sel = int(np.logical_and(pos_flat, sel).sum())
    neg_sel = int(np.logical_and(~pos_flat, sel).sum())
    return {
        "pos_total": pos_total, "neg_total": neg_total,
        "pos_sel": pos_sel, "neg_sel": neg_sel,
        "n_total": int(pos_flat.size), "n_sel": int(sel.sum()),
    }


def print_pos_neg_stats(msks: np.ndarray, sel_mask_flat: np.ndarray, prefix: str = "[Selection]") -> Dict[str, int]:
    stats = pos_neg_counts(msks, sel_mask_flat)
    logging.info(f"{prefix} Selected pixels: {stats['n_sel']} / {stats['n_total']}")
    logging.info(f"{prefix} Pos:Neg BEFORE = {stats['pos_total']}:{stats['neg_total']} "
                 f"({stats['pos_total'] / (stats['neg_total'] + 1e-8):.4f})")
    logging.info(f"{prefix} Pos:Neg AFTER  = {stats['pos_sel']}:{stats['neg_sel']} "
                 f"({stats['pos_sel'] / (stats['neg_sel'] + 1e-8):.4f})")
    return stats


def apply_random_thinning(
    sel_mask_flat: np.ndarray,
    select_perc: float,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    sel_mask_flat = sel_mask_flat.astype(bool, copy=False)

    if select_perc >= 1.0:
        return sel_mask_flat, np.where(sel_mask_flat)[0]

    if not (0.0 < select_perc <= 1.0):
        raise ValueError(f"select_perc must be in (0, 1], got {select_perc}")

    sel_idx = np.where(sel_mask_flat)[0]
    n_sel = sel_idx.size
    if n_sel == 0:
        return sel_mask_flat, sel_idx

    n_keep = max(1, int(round(select_perc * n_sel)))
    keep_idx = rng.choice(sel_idx, size=n_keep, replace=False)
    new_sel = np.zeros_like(sel_mask_flat, dtype=bool)
    new_sel[keep_idx] = True
    return new_sel, keep_idx
