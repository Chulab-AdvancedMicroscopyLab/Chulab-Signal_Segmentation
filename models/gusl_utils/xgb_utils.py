"""
Merged XGBoost pipeline utilities:
  - Feature stream extraction (stream_extract_features)
  - XGBoost training (train_xgboost_model)
  - Full prediction (run_full_prediction)
  - Residual target / input construction
  - Level-output mask saving
"""
import os
import gc
import glob
import io
import math
import struct
import time
import zipfile
import logging
import tempfile

import cv2
import numpy as np
import numpy.lib.format as npf
import xgboost as xgb
import joblib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from .featgen import (
    _NUMBA_AVAILABLE, _gather_neighbor_patches, _gather_center_patches,
    _gather_grad_patches, _reflect_index,
)
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch_index(path: str) -> int:
    name = os.path.splitext(os.path.basename(path))[0]
    return int(name.split("_")[-1])


def _load_npz(path: str):
    data = np.load(path)
    return data["X"], data["y"]


def _write_npz_from_memmap(save_path: str, X_mmap: np.memmap, y: np.ndarray, chunk_rows: int = 50_000):
    n_rows, n_cols = X_mmap.shape
    x_dtype = np.dtype(np.float32)

    header_str = "{'descr': '%s', 'fortran_order': False, 'shape': (%d, %d), }" % (
        x_dtype.str, n_rows, n_cols
    )
    padding = 64 - ((10 + len(header_str)) % 64)
    if padding == 64:
        padding = 0
    header_str = header_str + " " * (padding - 1) + "\n"
    header_len = struct.pack("<H", len(header_str))
    npy_header = b"\x93NUMPY" + bytes([1, 0]) + header_len + header_str.encode("latin1")

    with zipfile.ZipFile(save_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        buf = io.BytesIO()
        np.save(buf, y.astype(np.float32))
        zf.writestr("y.npy", buf.getvalue())

        with zf.open("X.npy", "w", force_zip64=True) as f:
            f.write(npy_header)
            for row_start in range(0, n_rows, chunk_rows):
                row_end = min(n_rows, row_start + chunk_rows)
                f.write(X_mmap[row_start:row_end].astype(np.float32).tobytes())


# ---------------------------------------------------------------------------
# Feature stream extraction
# ---------------------------------------------------------------------------

def _subbatch_generator(featgen, imgs_batch, msks_batch, featgen_res, res_batch, sub_batch_size):
    imgs = featgen._ensure_nhw(imgs_batch)
    N, H, W = imgs.shape
    total = N * H * W

    padded = featgen._pad_volume(imgs)
    if not _NUMBA_AVAILABLE:
        win = sliding_window_view(padded, window_shape=(featgen.kd, featgen.k, featgen.k))
        patch_view = win[featgen.r_nd:featgen.r_nd + N, featgen.r_n:featgen.r_n + H, featgen.r_n:featgen.r_n + W]

    if featgen.use_grad:
        gmax, gavg = featgen._precompute_grad_maps(imgs)
        if not _NUMBA_AVAILABLE:
            gv_max, gv_avg = featgen._grad_window_views(gmax, gavg)
        grad_len = featgen.grad_depth * featgen.grad_size * featgen.grad_size
    else:
        gmax = gavg = None
        grad_len = 0

    if res_batch is not None:
        res_imgs = featgen_res._ensure_nhw(res_batch)
        padded_r = featgen_res._pad_volume(res_imgs)
        if not _NUMBA_AVAILABLE:
            win_r = sliding_window_view(padded_r, window_shape=(featgen_res.kd, featgen_res.k, featgen_res.k))
            patch_view_r = win_r[featgen_res.r_nd:featgen_res.r_nd + N, featgen_res.r_n:featgen_res.r_n + H, featgen_res.r_n:featgen_res.r_n + W]
        if featgen_res.use_grad:
            gmax_r, gavg_r = featgen_res._precompute_grad_maps(res_imgs)
            if not _NUMBA_AVAILABLE:
                gv_max_r, gv_avg_r = featgen_res._grad_window_views(gmax_r, gavg_r)
            grad_len_r = featgen_res.grad_depth * featgen_res.grad_size * featgen_res.grad_size
        else:
            gmax_r = gavg_r = None
            grad_len_r = 0
    else:
        padded_r = gmax_r = gavg_r = None
        grad_len_r = 0

    msks_arr = featgen._ensure_nhw(msks_batch)
    y_full = msks_arr.reshape(-1).astype(np.float32, copy=False)

    neigh_offsets = [(dn, di, dj)
                     for dn in featgen._neigh_off_d_1d
                     for di in featgen._neigh_off_xy_1d
                     for dj in featgen._neigh_off_xy_1d]
    num_neighbors = len(neigh_offsets)
    out_per_patch = featgen._saab_out_dim()
    patch_dim = featgen.kd * featgen.k * featgen.k

    if res_batch is not None:
        neigh_offsets_r = [(dn, di, dj)
                           for dn in featgen_res._neigh_off_d_1d
                           for di in featgen_res._neigh_off_xy_1d
                           for dj in featgen_res._neigh_off_xy_1d]
        num_neighbors_r = len(neigh_offsets_r)
        out_per_patch_r = featgen_res._saab_out_dim()
        patch_dim_r = featgen_res.kd * featgen_res.k * featgen_res.k

    for sub_start in range(0, total, sub_batch_size):
        sub_end = min(total, sub_start + sub_batch_size)
        lin_b = np.arange(sub_start, sub_end, dtype=np.int64)
        nb = lin_b.size

        n0 = lin_b // (H * W)
        rem = lin_b % (H * W)
        i0 = rem // W
        j0 = rem % W

        neigh_n = np.stack([_reflect_index(n0 + dn, N) for dn, _, _ in neigh_offsets])
        neigh_i = np.stack([_reflect_index(i0 + di, H) for _, di, _ in neigh_offsets])
        neigh_j = np.stack([_reflect_index(j0 + dj, W) for _, _, dj in neigh_offsets])

        n0i = n0.astype(np.int64)
        i0i = i0.astype(np.int64)
        j0i = j0.astype(np.int64)

        if _NUMBA_AVAILABLE:
            P = _gather_neighbor_patches(padded, neigh_n.astype(np.int64), neigh_i.astype(np.int64), neigh_j.astype(np.int64),
                                         featgen.r_nd, featgen.r_n, featgen.kd, featgen.k)
        else:
            P = patch_view[neigh_n, neigh_i, neigh_j].reshape(num_neighbors * nb, patch_dim).astype(np.float32, copy=False)

        Y = featgen.saab.transform(P).astype(np.float32, copy=False)
        saab_feat = Y.reshape(num_neighbors, nb, out_per_patch).transpose(1, 0, 2).reshape(nb, -1)
        del P, Y, neigh_n, neigh_i, neigh_j

        if _NUMBA_AVAILABLE:
            raw_feat = _gather_center_patches(padded, n0i, i0i, j0i, featgen.r_nd, featgen.r_n, featgen.kd, featgen.k)
        else:
            raw_feat = patch_view[n0, i0, j0].reshape(nb, -1).astype(np.float32, copy=False)

        if featgen.use_grad:
            if _NUMBA_AVAILABLE:
                grad_feat = _gather_grad_patches(gmax, gavg, n0i, i0i, j0i,
                                                 featgen.r_gd, featgen.r_gxy,
                                                 featgen.grad_depth, featgen.grad_size,
                                                 N, H, W)
            else:
                gm = gv_max[n0, i0, j0].reshape(nb, grad_len)
                ga = gv_avg[n0, i0, j0].reshape(nb, grad_len)
                grad_feat = np.stack([gm, ga], axis=2).reshape(nb, grad_len * 2).astype(np.float32, copy=False)
                del gm, ga
            X_chunk = np.concatenate([saab_feat, raw_feat, grad_feat], axis=1)
            del grad_feat
        else:
            X_chunk = np.concatenate([saab_feat, raw_feat], axis=1)
        del saab_feat, raw_feat

        if res_batch is not None:
            neigh_n_r = np.stack([_reflect_index(n0 + dn, N) for dn, _, _ in neigh_offsets_r])
            neigh_i_r = np.stack([_reflect_index(i0 + di, H) for _, di, _ in neigh_offsets_r])
            neigh_j_r = np.stack([_reflect_index(j0 + dj, W) for _, _, dj in neigh_offsets_r])

            if _NUMBA_AVAILABLE:
                P_r = _gather_neighbor_patches(padded_r, neigh_n_r.astype(np.int64), neigh_i_r.astype(np.int64), neigh_j_r.astype(np.int64),
                                               featgen_res.r_nd, featgen_res.r_n, featgen_res.kd, featgen_res.k)
            else:
                P_r = patch_view_r[neigh_n_r, neigh_i_r, neigh_j_r].reshape(num_neighbors_r * nb, patch_dim_r).astype(np.float32, copy=False)

            Y_r = featgen_res.saab.transform(P_r).astype(np.float32, copy=False)
            saab_feat_r = Y_r.reshape(num_neighbors_r, nb, out_per_patch_r).transpose(1, 0, 2).reshape(nb, -1)
            del P_r, Y_r, neigh_n_r, neigh_i_r, neigh_j_r

            if _NUMBA_AVAILABLE:
                raw_feat_r = _gather_center_patches(padded_r, n0i, i0i, j0i, featgen_res.r_nd, featgen_res.r_n, featgen_res.kd, featgen_res.k)
            else:
                raw_feat_r = patch_view_r[n0, i0, j0].reshape(nb, -1).astype(np.float32, copy=False)

            if featgen_res.use_grad:
                if _NUMBA_AVAILABLE:
                    grad_feat_r = _gather_grad_patches(gmax_r, gavg_r, n0i, i0i, j0i,
                                                       featgen_res.r_gd, featgen_res.r_gxy,
                                                       featgen_res.grad_depth, featgen_res.grad_size,
                                                       N, H, W)
                else:
                    gm_r = gv_max_r[n0, i0, j0].reshape(nb, grad_len_r)
                    ga_r = gv_avg_r[n0, i0, j0].reshape(nb, grad_len_r)
                    grad_feat_r = np.stack([gm_r, ga_r], axis=2).reshape(nb, grad_len_r * 2).astype(np.float32, copy=False)
                    del gm_r, ga_r
                X_res = np.concatenate([saab_feat_r, raw_feat_r, grad_feat_r], axis=1)
                del grad_feat_r
            else:
                X_res = np.concatenate([saab_feat_r, raw_feat_r], axis=1)
            del saab_feat_r, raw_feat_r

            X_chunk = np.concatenate([X_chunk, X_res], axis=1).astype(np.float32, copy=False)
            del X_res

        yield X_chunk, y_full[sub_start:sub_end]
        del X_chunk


def stream_extract_features(
    imgs: np.ndarray,
    msks: np.ndarray,
    featgen,
    rft,
    lnt,
    save_dir: str,
    batch_size: int = 5,
    residual_imgs: Optional[np.ndarray] = None,
    featgen_res=None,
):
    os.makedirs(save_dir, exist_ok=True)
    n_imgs = imgs.shape[0]
    num_batches = math.ceil(n_imgs / batch_size)
    sub_batch_size = featgen.batch_centers

    t_featgen = t_rft = t_lnt = t_save = 0.0

    for batch_id, start in enumerate(range(0, n_imgs, batch_size)):
        end = min(start + batch_size, n_imgs)
        logging.info(f"[Feature Stream] batch {batch_id+1}/{num_batches}")

        imgs_batch = imgs[start:end]
        msks_batch = msks[start:end]
        res_batch = residual_imgs[start:end] if residual_imgs is not None else None

        N_b, H, W = imgs_batch.shape[0], imgs_batch.shape[-2], imgs_batch.shape[-1]
        n_total = N_b * H * W

        save_path = os.path.join(save_dir, f"batch_features_{batch_id:04d}.npz")

        X_buf = None
        y_all = np.empty(n_total, dtype=np.float32)
        offset = 0
        n_final = None

        _t0_feat = time.perf_counter()
        for X_chunk, y_chunk in _subbatch_generator(featgen, imgs_batch, msks_batch, featgen_res, res_batch, sub_batch_size):
            t_featgen += time.perf_counter() - _t0_feat

            t0 = time.perf_counter()
            X_rft = rft.transform(X_chunk)
            t_rft += time.perf_counter() - t0

            t0 = time.perf_counter()
            X_lnt = lnt.transform(X_rft)
            t_lnt += time.perf_counter() - t0

            nb = len(y_chunk)
            if X_buf is None:
                n_final = X_rft.shape[1] + X_lnt.shape[1]
                X_buf = np.empty((n_total, n_final), dtype=np.float32)

            n_rft = X_rft.shape[1]
            X_buf[offset:offset + nb, :n_rft] = X_rft
            X_buf[offset:offset + nb, n_rft:] = X_lnt
            y_all[offset:offset + nb] = y_chunk
            offset += nb
            del X_chunk, X_rft, X_lnt
            _t0_feat = time.perf_counter()

        logging.info(f"[Feature Stream] batch {batch_id+1} done: {n_total} pixel samples × {n_final} features")

        t0 = time.perf_counter()
        np.savez(save_path, X=X_buf, y=y_all)
        t_save += time.perf_counter() - t0

        logging.info(f"  FeatGen={t_featgen:.2f}s  RFT={t_rft:.2f}s  LNT={t_lnt:.2f}s  Save={t_save:.2f}s")
        del X_buf, y_all
        gc.collect()


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

class _TqdmCallback(xgb.callback.TrainingCallback):
    def __init__(self, n_estimators):
        self._bar = tqdm(total=n_estimators, desc="XGBoost", unit="tree", dynamic_ncols=True)

    def after_iteration(self, model, epoch, evals_log):
        metrics = {f"{ds}-{m}": f"{vals[-1]:.4f}" for ds, mv in evals_log.items() for m, vals in mv.items()}
        self._bar.set_postfix(metrics)
        self._bar.update(1)
        return False

    def after_training(self, model):
        self._bar.close()
        return model


def _load_selected_samples(feature_dir: str, sel_mask: np.ndarray, num_workers: int = 8):
    files = sorted(glob.glob(os.path.join(feature_dir, "batch_features_*.npz")), key=_batch_index)
    if not files:
        raise FileNotFoundError(f"No batch_features_*.npz found in {feature_dir}")

    logging.info(f"[Load] Reading {len(files)} feature batch(es)")

    # Pass 1: read only y per batch to count selections — keeps peak RAM to O(1 batch)
    offset = 0
    sel_counts = []
    for f in files:
        y = np.load(f)["y"]
        chunk = sel_mask[offset:offset + len(y)]
        if len(chunk) != len(y):
            raise ValueError(f"Selection mask length mismatch at offset {offset}")
        sel_counts.append(int(chunk.sum()))
        offset += len(y)

    n_total = sum(sel_counts)
    n_feat  = np.load(files[0])["X"].shape[1]
    X_out = np.empty((n_total, n_feat), dtype=np.float32)
    y_out = np.empty(n_total, dtype=np.float32)

    # Pass 2: stream one batch at a time — never hold more than 1 batch in RAM
    offset, out_offset = 0, 0
    for f, n_sel in tqdm(zip(files, sel_counts), total=len(files), desc="Loading batches", unit="file"):
        data = np.load(f)
        X, y = data["X"], data["y"]
        mask_chunk = sel_mask[offset:offset + len(y)]
        X_out[out_offset:out_offset + n_sel] = X[mask_chunk]
        y_out[out_offset:out_offset + n_sel] = y[mask_chunk]
        offset     += len(y)
        out_offset += n_sel
        del X, y, data

    logging.info(f"[Load] {n_total} pixel samples × {n_feat} features loaded")
    return X_out, y_out


def train_xgboost_model(
    train_feature_dir: str,
    val_feature_dir: str,
    sel_mask_train: np.ndarray,
    sel_mask_val: np.ndarray,
    model_dir: str,
    level: int,
    gpu_memory_limit_gb: float = 30,
    n_estimators: int = 5000,
    max_depth: int = 4,
    learning_rate: float = 0.15,
    early_stopping_rounds: int = 5,
    device: str = "cuda",
):
    model_path = os.path.join(model_dir, f"xgb_level{level}.ubj")

    try:
        X_train_sel, y_train_sel = _load_selected_samples(train_feature_dir, sel_mask_train)
        X_val_sel, y_val_sel = _load_selected_samples(val_feature_dir, sel_mask_val)

        memory_gb = (X_train_sel.nbytes + X_val_sel.nbytes) / 1e9
        logging.info(f"[XGBoost] Selected train+val size: {memory_gb:.2f} GB")

        actual_device = "cuda" if device == "cuda" and memory_gb < gpu_memory_limit_gb else "cpu"
        if device == "cuda" and actual_device == "cpu":
            logging.info("[XGBoost] Data exceeds GPU threshold. Falling back to CPU.")

        logging.info(f"[XGBoost] Training on device={actual_device}")
        model = xgb.XGBRegressor(
            tree_method="hist",
            device=actual_device,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            eval_metric="rmse",
            early_stopping_rounds=early_stopping_rounds,
            verbosity=0,
            callbacks=[_TqdmCallback(n_estimators)],
        )
        model.fit(
            X_train_sel, y_train_sel,
            eval_set=[(X_train_sel, y_train_sel), (X_val_sel, y_val_sel)],
            verbose=False,
        )
        model.set_params(callbacks=None)
        model.get_booster().save_model(model_path)
        logging.info(f"[XGBoost] Saved to {model_path}")
        return model

    except MemoryError as e:
        raise MemoryError(
            "[XGBoost] OOM. Reduce decode_percentage, n_selected, or use device='cpu'.\n"
            f"Original: {e}"
        ) from e
    except xgb.core.XGBoostError as e:
        msg = str(e).lower()
        if any(k in msg for k in ("out of memory", "bad allocation", "std::bad_alloc")):
            raise RuntimeError(
                "[XGBoost] XGBoost OOM. Reduce decode_percentage or use device='cpu'.\n"
                f"Original: {e}"
            ) from e
        raise


def run_full_prediction(feature_dir: str, xgb_model) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(feature_dir, "batch_features_*.npz")), key=_batch_index)
    preds = []
    for f in files:
        X = np.load(f)["X"]
        preds.append(xgb_model.predict(X))
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Residual targets and inputs
# ---------------------------------------------------------------------------

def build_level_targets(
    gt_imgs: np.ndarray,
    img_names: List[str],
    level: int,
    deepest_level: int,
    prev_mask_folder: str = None,
) -> np.ndarray:
    gt_imgs = np.asarray(gt_imgs, dtype=np.float32)
    N, H, W = gt_imgs.shape

    if level == deepest_level:
        return gt_imgs

    if prev_mask_folder is None:
        raise ValueError("prev_mask_folder required for non-deepest levels")

    if len(img_names) != N:
        raise ValueError(f"img_names length {len(img_names)} != N={N}")

    prev_preds = np.empty((N, H, W), dtype=np.float32)
    for i, name in enumerate(img_names):
        prev_path = os.path.join(prev_mask_folder, f"{name}.png")
        prev = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
        if prev is None:
            raise FileNotFoundError(f"Cannot read previous prediction: {prev_path}")
        if prev.shape != (H, W):
            prev = cv2.resize(prev, (W, H), interpolation=cv2.INTER_LANCZOS4)
        prev_preds[i] = prev.astype(np.float32)

    return gt_imgs - prev_preds


def build_residual_input_from_previous(
    img_names: List[str],
    prev_mask_folder: str,
    imgs: np.ndarray,
) -> np.ndarray:
    imgs = np.asarray(imgs, dtype=np.float32)
    N, H, W = imgs.shape

    prev_stack = np.empty((N, H, W), dtype=np.float32)
    for i, name in enumerate(img_names):
        prev_path = os.path.join(prev_mask_folder, f"{name}.png")
        prev = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
        if prev is None:
            raise FileNotFoundError(f"Cannot read previous prediction: {prev_path}")
        if prev.shape != (H, W):
            prev = cv2.resize(prev, (W, H), interpolation=cv2.INTER_LANCZOS4)
        prev_stack[i] = prev.astype(np.float32)

    return imgs - prev_stack


def concat_img_res_features(
    featgen_img,
    featgen_res,
    imgs: np.ndarray,
    residual_imgs: np.ndarray,
    targets: np.ndarray,
    sel_mask_flat=None,
    verbose: bool = False,
):
    X_img, y = featgen_img.transform(imgs, targets, sel_mask_flat=sel_mask_flat, verbose=verbose, name="FeatGen3D-Image")
    X_res, y_res = featgen_res.transform(residual_imgs, targets, sel_mask_flat=sel_mask_flat, verbose=verbose, name="FeatGen3D-Residual")

    if y.shape != y_res.shape:
        raise ValueError(f"Target mismatch img/res: {y.shape} vs {y_res.shape}")

    X = np.concatenate([X_img, X_res], axis=1).astype(np.float32, copy=False)
    return X, y.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Level-output mask saving
# ---------------------------------------------------------------------------

def save_pred_as_masks(
    img_names: List[str],
    y_pred_flat: np.ndarray,
    out_dir: str,
    input_size: List[int],
    clip_min: float = 0.0,
    clip_max: float = 255.0,
    out_dtype=np.uint8,
) -> None:
    H, W = input_size[0], input_size[1]
    N = len(img_names)
    expected = N * H * W
    if y_pred_flat.size != expected:
        raise ValueError(f"y_pred size mismatch: got {y_pred_flat.size}, expected {expected} (N={N}, H={H}, W={W})")

    y_pred = y_pred_flat.reshape(N, H, W).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)

    for i, name in enumerate(img_names):
        mask = np.clip(y_pred[i], clip_min, clip_max).astype(out_dtype)
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), mask)


def combine_pred_with_prev_and_save(
    img_names: List[str],
    prev_mask_folder: str,
    y_pred_flat: np.ndarray,
    out_dir: str,
    level: int,
    deepest_level: int,
    clip_min: float = 0.0,
    clip_max: float = 255.0,
    out_dtype=np.uint8,
) -> None:
    N = len(img_names)
    if N == 0:
        raise ValueError("img_names is empty")

    # Probe first mask to determine (H,W) after optional upsampling
    first_path = os.path.join(prev_mask_folder, f"{img_names[0]}.png")
    first_mask = cv2.imread(first_path, cv2.IMREAD_GRAYSCALE)
    if first_mask is None:
        raise FileNotFoundError(f"Cannot read: {first_path}")

    if level < deepest_level:
        first_mask = cv2.resize(first_mask, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

    H, W = first_mask.shape
    expected = N * H * W
    if y_pred_flat.size != expected:
        raise ValueError(f"y_pred size mismatch: got {y_pred_flat.size}, expected {expected}")

    y_pred = y_pred_flat.reshape(N, H, W).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)

    for i, name in enumerate(img_names):
        prev_path = os.path.join(prev_mask_folder, f"{name}.png")
        prev = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
        if prev is None:
            raise FileNotFoundError(f"Cannot read: {prev_path}")

        if level < deepest_level:
            prev = cv2.resize(prev, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        elif prev.shape != (H, W):
            prev = cv2.resize(prev, (W, H), interpolation=cv2.INTER_LANCZOS4)

        combined = np.clip(prev.astype(np.float32) + y_pred[i], clip_min, clip_max).astype(out_dtype)
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), combined)
