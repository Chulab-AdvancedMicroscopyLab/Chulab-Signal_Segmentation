"""
GUSLModel — coarse-to-fine 3D segmentation using Saab + RFT + LNT + XGBoost.
Drop-in model class for the Chulab-Signal_Segmentation pipeline.
"""
import gc
import logging
import os
from typing import Dict, List, Optional, Union

import cv2
import joblib
import numpy as np
import xgboost as xgb

from .gusl_utils.featgen import FeatGen3D
from .gusl_utils.rft import DualRFT, build_feature_masks, build_feature_masks_from_featgens
from .gusl_utils.lnt import LNT
from .gusl_utils.sample_selection import (
    boundary_selection_mask, apply_random_thinning, print_pos_neg_stats,
)
from .gusl_utils.xgb_utils import (
    stream_extract_features, train_xgboost_model, run_full_prediction,
    save_pred_as_masks, combine_pred_with_prev_and_save,
    build_level_targets, build_residual_input_from_previous, concat_img_res_features,
)

logger = logging.getLogger(__name__)


def _resolve_per_level(values, deepest_level: int, name: str, cast_fn=None) -> List:
    if not isinstance(values, (list, tuple)):
        values = [values]
    if cast_fn is not None:
        values = [cast_fn(v) for v in values]
    if len(values) == 1:
        return list(values) * deepest_level
    if len(values) != deepest_level:
        raise ValueError(
            f"{name} must have length 1 or deepest_level={deepest_level}, got {len(values)}: {values}"
        )
    return list(values)


def _feature_batches_ready(feature_dir: str, num_images: int, batch_size: int) -> bool:
    import math
    n_batches = math.ceil(num_images / batch_size)
    for i in range(n_batches):
        if not os.path.isfile(os.path.join(feature_dir, f"batch_features_{i:04d}.npz")):
            return False
    return True


class GUSLModel:
    """
    Coarse-to-fine Saab/XGBoost segmentation model.

    Training flow (per level, deepest → finest):
      1. FeatGen3D (Saab + neighborhood + gradient features)
      2. DualRFT  (histogram-based feature selection)
      3. LNT      (learnable node transform via shallow XGBoost leaf subsets)
      4. XGBoost  (final pixel-wise regressor)

    Residual stream: for levels < deepest_level, a second FeatGen3D is trained
    on (image − previous-level-prediction) and its features are concatenated.

    Args
    ----
    deepest_level : int
        Number of pyramid levels. Level deepest_level is coarsest, level 1 is finest.
    base_size : int
        Spatial resolution at level 1. Level k uses base_size / 2^(k-1).
    kernel_size, kernel_depth : int or list[int]
        Saab patch dimensions (per level, deepest → shallowest, or single value).
    neigh_size, neigh_depth, neigh_stride : int or list[int]
        Neighborhood sampling params.
    use_grad, grad_size, grad_depth : bool/int or list
        Gradient feature params.
    batch_centers : int
        Max voxels per FeatGen3D batch (controls GPU/CPU peak memory).
    n_bins, n_selected : int or list[int]
        RFT histogram bins and selected feature count.
    lnt_depth, lnt_num_tree : int or list[int]
        LNT XGBoost params.
    n_estimators, max_depth, learning_rate, early_stopping_rounds : int/float
        Final XGBoost regressor params.
    keep_frac, low1, low2, high1, high2 : float
        Boundary-aware sample selection params.
    encode_percentage, decode_percentage : float or list[float]
        Per-level pixel keep fractions for encoder / XGBoost training stages.
    val_encode_percentage, val_decode_percentage : float or list[float]
        Same for validation.
    patch_depth : int
        Depth (Z frames) of each training patch (= training_patch_size[0]).
    batch_size : int
        Number of whole patches per feature-stream batch. Frame stride = batch_size × patch_depth.
    device : str
        "cuda" or "cpu".
    gpu_id : int
        CUDA device index.
    """

    def __init__(
        self,
        deepest_level: int = 4,
        base_size: int = 1280,
        patch_depth: int = 16,
        # encoder per-level params (deepest → shallowest, or scalar)
        kernel_size: Union[int, List[int]] = [3, 3, 5, 5],
        kernel_depth: Union[int, List[int]] = [3, 3, 3, 3],
        neigh_size: Union[int, List[int]] = [3, 5, 5, 7],
        neigh_depth: Union[int, List[int]] = [1, 1, 1, 1],
        neigh_stride: Union[int, List[int]] = [2, 2, 2, 2],
        use_grad: Union[bool, List[bool]] = [True, True, True, True],
        grad_size: Union[int, List[int]] = [3, 5, 5, 5],
        grad_depth: Union[int, List[int]] = [3, 3, 3, 3],
        batch_centers: int = 100_000,
        # RFT per-level params
        n_bins: Union[int, List[int]] = [32, 32, 32, 32],
        n_selected: Union[int, List[int]] = [400, 500, 600, 600],
        rft_sample_frac: Optional[float] = 0.5,
        lnt_sample_frac: Optional[float] = 0.5,
        # LNT per-level params
        lnt_depth: Union[int, List[int]] = [3, 3, 4, 4],
        lnt_num_tree: Union[int, List[int]] = [150, 150, 200, 200],
        # Final XGBoost params
        n_estimators: int = 5000,
        max_depth: int = 4,
        learning_rate: float = 0.15,
        early_stopping_rounds: int = 5,
        # Sample selection
        keep_frac: float = 0.3,
        low1: float = 0.0,
        low2: float = 0.05,
        high1: float = 0.1,
        high2: float = 0.4,
        encode_percentage: Union[float, List[float]] = [1.0, 1.0, 0.4, 0.1],
        decode_percentage: Union[float, List[float]] = [1.0, 1.0, 0.7, 0.3],
        val_encode_percentage: Union[float, List[float]] = [1.0, 1.0, 1.0, 1.0],
        val_decode_percentage: Union[float, List[float]] = [1.0, 1.0, 1.0, 1.0],
        # I/O
        batch_size: int = 2,
        device: str = "cuda",
        gpu_id: int = 0,
    ):
        self.deepest_level = deepest_level
        self.base_size = base_size
        self.patch_depth = patch_depth

        # Resolve per-level lists
        self.kernel_sizes   = _resolve_per_level(kernel_size,   deepest_level, "kernel_size",   int)
        self.kernel_depths  = _resolve_per_level(kernel_depth,  deepest_level, "kernel_depth",  int)
        self.neigh_sizes    = _resolve_per_level(neigh_size,    deepest_level, "neigh_size",    int)
        self.neigh_depths   = _resolve_per_level(neigh_depth,   deepest_level, "neigh_depth",   int)
        self.neigh_strides  = _resolve_per_level(neigh_stride,  deepest_level, "neigh_stride",  int)
        self.use_grads      = _resolve_per_level(use_grad,      deepest_level, "use_grad",      bool)
        self.grad_sizes     = _resolve_per_level(grad_size,     deepest_level, "grad_size",     int)
        self.grad_depths    = _resolve_per_level(grad_depth,    deepest_level, "grad_depth",    int)
        self.n_bins_list    = _resolve_per_level(n_bins,        deepest_level, "n_bins",        int)
        self.n_selected_list = _resolve_per_level(n_selected,  deepest_level, "n_selected",    int)
        self.lnt_depth_list  = _resolve_per_level(lnt_depth,   deepest_level, "lnt_depth",     int)
        self.lnt_num_tree_list = _resolve_per_level(lnt_num_tree, deepest_level, "lnt_num_tree", int)
        self.encode_pcts    = _resolve_per_level(encode_percentage,     deepest_level, "encode_percentage",     float)
        self.decode_pcts    = _resolve_per_level(decode_percentage,     deepest_level, "decode_percentage",     float)
        self.val_encode_pcts = _resolve_per_level(val_encode_percentage, deepest_level, "val_encode_percentage", float)
        self.val_decode_pcts = _resolve_per_level(val_decode_percentage, deepest_level, "val_decode_percentage", float)

        self.rft_sample_frac = rft_sample_frac
        self.lnt_sample_frac = lnt_sample_frac
        self.batch_centers = batch_centers
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.keep_frac = keep_frac
        self.low1 = low1
        self.low2 = low2
        self.high1 = high1
        self.high2 = high2
        self.batch_size = batch_size
        self.device = device
        self.gpu_id = gpu_id

        # Per-level model state (index 0 = deepest level, index deepest_level-1 = level 1)
        self.featgen_img:  List[Optional[FeatGen3D]] = [None] * deepest_level
        self.featgen_res:  List[Optional[FeatGen3D]] = [None] * deepest_level
        self.rft_models:   List[Optional[DualRFT]]   = [None] * deepest_level
        self.lnt_models:   List[Optional[LNT]]       = [None] * deepest_level
        self.xgb_models:   List                       = [None] * deepest_level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_imgs: np.ndarray,
        train_msks: np.ndarray,
        val_imgs: np.ndarray,
        val_msks: np.ndarray,
        saveroot: str,
        train_names: Optional[List[str]] = None,
        val_names: Optional[List[str]] = None,
        run_level: Optional[int] = None,
    ) -> "GUSLModel":
        """
        Train the full coarse-to-fine pyramid.

        Args
        ----
        train_imgs, train_msks : (N, H, W) float32 numpy arrays (already normalised).
        val_imgs, val_msks     : (M, H, W) float32 numpy arrays.
        saveroot               : Root directory for intermediate features, level outputs, and models.
        train_names, val_names : Base filenames for intermediate PNG masks (default: frame_XXXX).
        run_level              : If set, train only this single level.
        """
        if self.device.startswith("cuda"):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        N_tr = train_imgs.shape[0]
        N_val = val_imgs.shape[0]
        train_names = train_names or [f"frame_{i:04d}" for i in range(N_tr)]
        val_names   = val_names   or [f"frame_{i:04d}" for i in range(N_val)]

        model_dir = os.path.join(saveroot, "models", f"Deepest_level_{self.deepest_level}")
        os.makedirs(model_dir, exist_ok=True)

        for level in range(self.deepest_level, 0, -1):
            if run_level is not None and level != run_level:
                logger.info(f"[Skip] Skipping level {level}")
                continue

            self._fit_level(
                level=level,
                train_imgs=train_imgs, train_msks=train_msks,
                val_imgs=val_imgs, val_msks=val_msks,
                saveroot=saveroot, model_dir=model_dir,
                train_names=train_names, val_names=val_names,
            )

        return self

    def predict(
        self,
        imgs: np.ndarray,
        pixel_batch_size: int = 500_000,
    ) -> np.ndarray:
        """
        Run coarse-to-fine inference on a full Z-chunk (N, H, W).

        Features are extracted once per level over the full spatial extent,
        then streamed through RFT → LNT → XGBoost in pixel_batch_size batches.
        No XY tiling — setup overhead paid once per level instead of once per tile.

        Level scale: level 1 = native resolution; level L = 1/(2^(L-1)) resolution.
        """
        imgs = np.asarray(imgs, dtype=np.float32)
        N, H_orig, W_orig = imgs.shape

        level_preds: Dict[int, np.ndarray] = {}

        for level in range(self.deepest_level, 0, -1):
            lvl_idx   = self.deepest_level - level
            fg        = self.featgen_img[lvl_idx]
            fg_r      = self.featgen_res[lvl_idx]
            rft       = self.rft_models[lvl_idx]
            lnt       = self.lnt_models[lvl_idx]
            xgb_model = self.xgb_models[lvl_idx]

            if any(m is None for m in [fg, rft, lnt, xgb_model]):
                raise RuntimeError(f"Level {level} not trained. Call fit() first.")

            # Coarser levels process at lower spatial resolution (same scale ratio as training)
            scale = 2 ** (level - 1)   # level 1: ×1, level 2: ×½, …
            Hl = max(1, H_orig // scale)
            Wl = max(1, W_orig // scale)

            if scale == 1:
                imgs_l = imgs
            else:
                imgs_l = np.stack([
                    cv2.resize(img, (Wl, Hl), interpolation=cv2.INTER_LANCZOS4)
                    for img in imgs
                ])

            if level < self.deepest_level:
                prev     = level_preds[level + 1]
                prev_up  = np.stack([
                    cv2.resize(p, (Wl, Hl), interpolation=cv2.INTER_LANCZOS4)
                    for p in prev
                ])
                res_imgs = (imgs_l - prev_up).astype(np.float32)
            else:
                res_imgs = None
                prev_up  = None

            n_pixels  = N * Hl * Wl
            n_batches = (n_pixels + pixel_batch_size - 1) // pixel_batch_size
            logger.info(f"[predict] level={level} shape=({N},{Hl},{Wl}) pixels={n_pixels} batches={n_batches}")

            # Setup FeatGen3D ONCE for this level — amortises pad/grad/view over all batches
            ctx_img = fg.prepare(imgs_l)
            ctx_res = fg_r.prepare(res_imgs) if res_imgs is not None else None

            y_pred_all = np.empty(n_pixels, dtype=np.float32)

            import time as _t
            for batch_idx, start in enumerate(range(0, n_pixels, pixel_batch_size), 1):
                end      = min(n_pixels, start + pixel_batch_size)
                keep_lin = np.arange(start, end, dtype=np.int64)

                t0 = _t.perf_counter()
                if level == self.deepest_level:
                    X = fg.transform_precomputed(ctx_img, keep_lin)
                else:
                    X_img = fg.transform_precomputed(ctx_img, keep_lin)
                    X_res = fg_r.transform_precomputed(ctx_res, keep_lin)
                    X     = np.concatenate([X_img, X_res], axis=1).astype(np.float32, copy=False)
                    del X_img, X_res
                t_feat = _t.perf_counter() - t0

                if self.device.startswith("cuda"):
                    import torch
                    # Upload X once, keep tensors on GPU through RFT → LNT → concat.
                    # Avoids repeated PCIe round-trips and eliminates the large CPU
                    # allocation + page-fault cost of np.concatenate on 600 MB+ arrays.
                    t0 = _t.perf_counter()
                    X_t   = torch.from_numpy(np.ascontiguousarray(X)).to(self.device)
                    del X
                    idx_t = torch.from_numpy(rft.selected_features).to(self.device)
                    X_rft_t = X_t[:, idx_t].contiguous()
                    del X_t, idx_t
                    torch.cuda.synchronize()
                    t_rft = _t.perf_counter() - t0

                    t0 = _t.perf_counter()
                    lnt_parts = [X_rft_t]
                    for kernel in lnt.svd:
                        k_t = torch.from_numpy(np.ascontiguousarray(kernel)).to(self.device)
                        lnt_parts.append(X_rft_t @ k_t)
                    X_final_t = torch.cat(lnt_parts, dim=1)
                    del X_rft_t, lnt_parts
                    X_final = X_final_t.cpu().numpy().astype(np.float32, copy=False)
                    del X_final_t
                    t_lnt = _t.perf_counter() - t0
                else:
                    t0 = _t.perf_counter()
                    X_rft = rft.transform(X)
                    del X
                    t_rft = _t.perf_counter() - t0

                    t0 = _t.perf_counter()
                    X_lnt = lnt.transform(X_rft)
                    X_final = np.concatenate([X_rft, X_lnt], axis=1).astype(np.float32, copy=False)
                    del X_rft, X_lnt
                    t_lnt = _t.perf_counter() - t0

                t0 = _t.perf_counter()
                dmat = xgb.DMatrix(X_final)
                y_pred_all[start:end] = xgb_model.get_booster().predict(dmat)
                del X_final, dmat
                t_xgb = _t.perf_counter() - t0

                if batch_idx <= 3:
                    logger.info(f"[predict] level={level} batch {batch_idx}/{n_batches} feat={t_feat:.1f}s rft={t_rft:.1f}s lnt={t_lnt:.1f}s xgb={t_xgb:.1f}s")
                else:
                    logger.info(f"[predict] level={level} batch {batch_idx}/{n_batches} ({100*end/n_pixels:.1f}%)")

            del ctx_img, ctx_res

            residual  = y_pred_all.reshape(N, Hl, Wl)
            pred_map  = residual if level == self.deepest_level else prev_up + residual
            level_preds[level] = np.clip(pred_map, 0, 255).astype(np.float32)

            del y_pred_all, residual
            if scale != 1:
                del imgs_l
            gc.collect()

        return level_preds[1]

    def set_device(self, device: str) -> None:
        """Propagate device to all sub-components.

        Call after load() to switch inference device — setting model.device directly
        does NOT update the Saab or LNT components baked in at training time.
        """
        import torch
        self.device = device
        torch_device = torch.device(device)
        lnt_mode = "gpu" if device.startswith("cuda") else "cpu"
        for fg in self.featgen_img + self.featgen_res:
            if fg is not None:
                fg.saab.device = torch_device
        for lnt in self.lnt_models:
            if lnt is not None:
                lnt.mode = lnt_mode
        # XGBoost GPU inference requires cupy; this workstation does not have it,
        # so always predict on CPU to avoid the device-mismatch fallback warning.
        for xgb_m in self.xgb_models:
            if xgb_m is not None:
                xgb_m.get_booster().set_param("device", "cpu")
        logger.info(f"[GUSLModel] device set to {device} (saab + lnt + xgb updated)")

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"[GUSLModel] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "GUSLModel":
        model = joblib.load(path)
        logger.info(f"[GUSLModel] Loaded from {path}")
        return model

    # ------------------------------------------------------------------
    # Internal: per-level training
    # ------------------------------------------------------------------

    def _fit_level(
        self,
        level: int,
        train_imgs: np.ndarray, train_msks: np.ndarray,
        val_imgs: np.ndarray,   val_msks: np.ndarray,
        saveroot: str, model_dir: str,
        train_names: List[str], val_names: List[str],
    ):
        logger.info(f"===== LEVEL {level} =====")
        lvl_idx = self.deepest_level - level
        size = self.base_size // (2 ** (level - 1))

        # Per-level hyperparams
        kernel_size  = self.kernel_sizes[lvl_idx]
        kernel_depth = self.kernel_depths[lvl_idx]
        neigh_size   = self.neigh_sizes[lvl_idx]
        neigh_depth  = self.neigh_depths[lvl_idx]
        neigh_stride = self.neigh_strides[lvl_idx]
        use_grad     = self.use_grads[lvl_idx]
        grad_size    = self.grad_sizes[lvl_idx]
        grad_depth   = self.grad_depths[lvl_idx]
        encode_pct   = self.encode_pcts[lvl_idx]
        decode_pct   = self.decode_pcts[lvl_idx]
        val_enc_pct  = self.val_encode_pcts[lvl_idx]
        val_dec_pct  = self.val_decode_pcts[lvl_idx]
        n_bins       = self.n_bins_list[lvl_idx]
        n_selected   = self.n_selected_list[lvl_idx]
        lnt_depth    = self.lnt_depth_list[lvl_idx]
        lnt_num_tree = self.lnt_num_tree_list[lvl_idx]

        # Resize to level resolution
        train_imgs_l = np.stack([cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4) for img in train_imgs])
        train_msks_l = np.stack([cv2.resize(msk, (size, size), interpolation=cv2.INTER_LANCZOS4) for msk in train_msks])
        val_imgs_l   = np.stack([cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4) for img in val_imgs])
        val_msks_l   = np.stack([cv2.resize(msk, (size, size), interpolation=cv2.INTER_LANCZOS4) for msk in val_msks])

        # --- Sample selection ---
        sel_win = {4: 3, 3: 5, 2: 11, 1: 21}.get(level, 5)
        sel_mask1_tr = boundary_selection_mask(
            imgs=train_imgs_l, msks=train_msks_l,
            intensity_ranges=((self.high1, self.high2), (self.low1, self.low2)),
            keep_fracs=(self.keep_frac, self.keep_frac),
            window=sel_win, img_window=sel_win,
        ).reshape(-1).astype(bool)
        print_pos_neg_stats(train_msks_l, sel_mask1_tr, prefix="[Train-Sel]")
        sel_mask2_tr, _ = apply_random_thinning(sel_mask1_tr, select_perc=encode_pct)
        print_pos_neg_stats(train_msks_l, sel_mask2_tr, prefix="[Train-Sel2]")

        sel_mask1_val = boundary_selection_mask(
            imgs=val_imgs_l, msks=val_msks_l,
            intensity_ranges=((self.high1, self.high2), (self.low1, self.low2)),
            keep_fracs=(self.keep_frac, self.keep_frac),
            window=sel_win, img_window=sel_win,
        ).reshape(-1).astype(bool)
        sel_mask2_val, _ = apply_random_thinning(sel_mask1_val, select_perc=val_enc_pct)

        # --- Residual targets / residual image streams ---
        if level == self.deepest_level:
            train_targets = train_msks_l.astype(np.float32)
            val_targets   = val_msks_l.astype(np.float32)
            train_res_imgs = None
            val_res_imgs   = None
        else:
            prev_tr_dir  = os.path.join(saveroot, "level_output", f"Deepest_level_{self.deepest_level}", f"level{level+1}", "train")
            prev_val_dir = os.path.join(saveroot, "level_output", f"Deepest_level_{self.deepest_level}", f"level{level+1}", "val")

            train_targets = build_level_targets(train_msks_l, train_names, level, self.deepest_level, prev_tr_dir)
            val_targets   = build_level_targets(val_msks_l,   val_names,   level, self.deepest_level, prev_val_dir)
            train_res_imgs = build_residual_input_from_previous(train_names, prev_tr_dir, train_imgs_l)
            val_res_imgs   = build_residual_input_from_previous(val_names,   prev_val_dir, val_imgs_l)

        # --- FeatGen-IMG ---
        featgen_path = os.path.join(model_dir, f"featgen_img_lvl{level}.joblib")
        if os.path.isfile(featgen_path):
            logger.info(f"[FeatGen-IMG] Loading existing: {featgen_path}")
            featgen = joblib.load(featgen_path)
        else:
            sel_indices = np.where(sel_mask2_tr)[0]
            featgen = FeatGen3D(
                kernel_size=kernel_size, kernel_depth=kernel_depth,
                neigh_size=neigh_size, neigh_depth=neigh_depth, neigh_stride=neigh_stride,
                use_grad=use_grad, grad_size=grad_size, grad_depth=grad_depth,
                batch_centers=self.batch_centers, device=self.device,
            )
            featgen.fit(train_imgs_l, train_targets, sel_indices=sel_indices)
            joblib.dump(featgen, featgen_path)
            logger.info(f"[FeatGen-IMG] Saved to {featgen_path}")
        self.featgen_img[lvl_idx] = featgen

        # --- FeatGen-RES ---
        featgen_res_path = os.path.join(model_dir, f"featgen_res_lvl{level}.joblib")
        featgen_res = None
        if level != self.deepest_level:
            if os.path.isfile(featgen_res_path):
                logger.info(f"[FeatGen-RES] Loading existing: {featgen_res_path}")
                featgen_res = joblib.load(featgen_res_path)
            else:
                sel_indices = np.where(sel_mask2_tr)[0]
                featgen_res = FeatGen3D(
                    kernel_size=kernel_size, kernel_depth=kernel_depth,
                    neigh_size=neigh_size, neigh_depth=neigh_depth, neigh_stride=neigh_stride,
                    use_grad=False,
                    grad_size=grad_size, grad_depth=grad_depth,
                    batch_centers=self.batch_centers, device=self.device,
                )
                featgen_res.fit(train_res_imgs, train_targets, sel_indices=sel_indices)
                joblib.dump(featgen_res, featgen_res_path)
                logger.info(f"[FeatGen-RES] Saved to {featgen_res_path}")
        self.featgen_res[lvl_idx] = featgen_res

        # --- RFT + LNT (need extracted features) ---
        rft_path = os.path.join(model_dir, f"rft_level{level}.joblib")
        lnt_path = os.path.join(model_dir, f"lnt_level{level}.joblib")

        if os.path.isfile(rft_path) and os.path.isfile(lnt_path):
            logger.info("[RFT/LNT] Loading existing models.")
            rft = joblib.load(rft_path)
            lnt = joblib.load(lnt_path)
        else:
            # Extract features on selected samples to fit RFT + LNT
            if level == self.deepest_level:
                X_tr, y_tr = featgen.transform(train_imgs_l, train_targets, sel_mask_flat=sel_mask2_tr, verbose=True, name="FeatGen3D-IMG")
                X_val, y_val = featgen.transform(val_imgs_l, val_targets, sel_mask_flat=sel_mask2_val, verbose=True, name="FeatGen3D-IMG")
            else:
                X_tr, y_tr = concat_img_res_features(featgen, featgen_res, train_imgs_l, train_res_imgs, train_targets,
                                                      sel_mask_flat=sel_mask2_tr, verbose=True)
                X_val, y_val = concat_img_res_features(featgen, featgen_res, val_imgs_l, val_res_imgs, val_targets,
                                                        sel_mask_flat=sel_mask2_val, verbose=True)

            X_tr  = X_tr.astype(np.float32, copy=False)
            y_tr  = y_tr.astype(np.float32, copy=False)
            X_val = X_val.astype(np.float32, copy=False)
            y_val = y_val.astype(np.float32, copy=False)

            if level == self.deepest_level:
                feature_masks = build_feature_masks(featgen, X_tr.shape[1])
            else:
                feature_masks = build_feature_masks_from_featgens(featgen, featgen_res, X_tr.shape[1])

            # RFT
            if os.path.isfile(rft_path):
                rft = joblib.load(rft_path)
            else:
                rft = DualRFT(n_bins=n_bins, n_selected=n_selected, feature_masks=feature_masks)
                frac = self.rft_sample_frac
                if frac and frac < 1.0:
                    rng = np.random.default_rng(42)
                    n_tr  = max(1, int(X_tr.shape[0]  * frac))
                    n_val = max(1, int(X_val.shape[0] * frac))
                    idx_tr  = rng.choice(X_tr.shape[0],  n_tr,  replace=False)
                    idx_val = rng.choice(X_val.shape[0], n_val, replace=False)
                    logger.info(f"[RFT] Subsampling for fit: {X_tr.shape[0]}→{n_tr} train, {X_val.shape[0]}→{n_val} val ({frac:.0%})")
                    rft.fit(X_tr[idx_tr], y_tr[idx_tr], X_val[idx_val], y_val[idx_val])
                else:
                    rft.fit(X_tr, y_tr, X_val, y_val)
                joblib.dump(rft, rft_path)
                logger.info(f"[RFT] Saved to {rft_path}")
                rft_plot_dir = os.path.join(saveroot, "plots", f"Deepest_level_{self.deepest_level}", f"level{level}", "rft")
                os.makedirs(rft_plot_dir, exist_ok=True)
                rft.plot(rft_plot_dir)
            del X_val, y_val
            gc.collect()

            # LNT
            if os.path.isfile(lnt_path):
                lnt = joblib.load(lnt_path)
            else:
                X_tr_rft = rft.transform(X_tr).astype(np.float32, copy=False)
                lnt = LNT(
                    depth=lnt_depth, num_tree=lnt_num_tree,
                    mode="gpu" if self.device.startswith("cuda") else "cpu",
                    lnt_sample_frac=self.lnt_sample_frac,
                )
                lnt.fit(X_tr_rft, y_tr)
                joblib.dump(lnt, lnt_path)
                logger.info(f"[LNT] Saved to {lnt_path}")
                del X_tr_rft

            del X_tr, y_tr
            gc.collect()

        self.rft_models[lvl_idx] = rft
        self.lnt_models[lvl_idx] = lnt

        # --- Feature extraction to disk (full dataset) ---
        tr_feat_dir  = os.path.join(saveroot, "features", f"Deepest_level_{self.deepest_level}", f"level{level}", "train")
        val_feat_dir = os.path.join(saveroot, "features", f"Deepest_level_{self.deepest_level}", f"level{level}", "val")
        os.makedirs(tr_feat_dir,  exist_ok=True)
        os.makedirs(val_feat_dir, exist_ok=True)

        frame_stride = self.batch_size * self.patch_depth
        tr_ready  = _feature_batches_ready(tr_feat_dir,  train_imgs_l.shape[0], frame_stride)
        val_ready = _feature_batches_ready(val_feat_dir, val_imgs_l.shape[0],   frame_stride)

        if not tr_ready:
            stream_extract_features(train_imgs_l, train_targets, featgen, rft, lnt,
                                     tr_feat_dir, frame_stride,
                                     residual_imgs=train_res_imgs, featgen_res=featgen_res)
        if not val_ready:
            stream_extract_features(val_imgs_l, val_targets, featgen, rft, lnt,
                                     val_feat_dir, frame_stride,
                                     residual_imgs=val_res_imgs, featgen_res=featgen_res)

        # --- XGBoost ---
        xgb_path = os.path.join(model_dir, f"xgb_level{level}.ubj")
        sel_mask2_tr_dec, _  = apply_random_thinning(sel_mask1_tr,  select_perc=decode_pct)
        sel_mask2_val_dec, _ = apply_random_thinning(sel_mask1_val, select_perc=val_dec_pct)

        if os.path.isfile(xgb_path):
            logger.info(f"[XGBoost] Loading existing: {xgb_path}")
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(xgb_path)
            xgb_model.get_booster().set_param("device", self.device)
        else:
            xgb_model = train_xgboost_model(
                train_feature_dir=tr_feat_dir,
                val_feature_dir=val_feat_dir,
                sel_mask_train=sel_mask2_tr_dec,
                sel_mask_val=sel_mask2_val_dec,
                model_dir=model_dir,
                level=level,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                early_stopping_rounds=self.early_stopping_rounds,
                device=self.device,
            )
            xgb_model.get_booster().set_param("device", self.device)
        self.xgb_models[lvl_idx] = xgb_model

        # --- Save level output PNGs (needed as residual inputs for finer levels) ---
        y_pred_tr  = run_full_prediction(tr_feat_dir,  xgb_model)
        y_pred_val = run_full_prediction(val_feat_dir, xgb_model)

        out_tr_dir  = os.path.join(saveroot, "level_output", f"Deepest_level_{self.deepest_level}", f"level{level}", "train")
        out_val_dir = os.path.join(saveroot, "level_output", f"Deepest_level_{self.deepest_level}", f"level{level}", "val")

        if level == self.deepest_level:
            save_pred_as_masks(train_names, y_pred_tr,  out_tr_dir,  input_size=[size, size])
            save_pred_as_masks(val_names,   y_pred_val, out_val_dir, input_size=[size, size])
        else:
            prev_tr_dir  = os.path.join(saveroot, "level_output", f"Deepest_level_{self.deepest_level}", f"level{level+1}", "train")
            prev_val_dir = os.path.join(saveroot, "level_output", f"Deepest_level_{self.deepest_level}", f"level{level+1}", "val")
            combine_pred_with_prev_and_save(train_names, prev_tr_dir,  y_pred_tr,  out_tr_dir,  level, self.deepest_level)
            combine_pred_with_prev_and_save(val_names,   prev_val_dir, y_pred_val, out_val_dir, level, self.deepest_level)

        logger.info(f"===== LEVEL {level} DONE =====")
