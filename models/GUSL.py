"""
GUSL — coarse-to-fine voxel regressor (Saab + RFT + LNT + XGBoost) as a drop-in model.

Same contract as the DL models: forward((B,1,D,H,W) or (B,1,H,W)) -> logits of the same shape,
so inference.py, the stitcher and the metrics work unchanged. Training is closed-form + boosting,
so train.py calls model.fit(train_ds, val_ds, full_config, device) instead of the epoch loop.

Levels run deepest -> 1; level L works at XY scale 1/2^(L-1) (level 1 = full resolution).
The deepest level regresses the mask; each finer level regresses the residual
(mask - upsampled coarser prediction) from features of the image and of (image - coarser prediction).
Everything is computed per patch on the GPU, identically in training and inference.
"""
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

from .gusl_utils.rft import DualRFT
from .gusl_utils.lnt import fit_lnt

try:
    import cupy  # optional: zero-copy GPU input for XGBoost predict
except ImportError:
    cupy = None

logger = logging.getLogger(__name__)


def _per_level(v, n, name):
    v = list(v) if isinstance(v, (list, tuple)) else [v]
    if len(v) == 1:
        return v * n
    if len(v) != n:
        raise ValueError(f"{name}: need 1 or {n} values (deepest -> level 1), got {v}")
    return v


def _down(x, s):
    return F.avg_pool3d(x, (1, s, s)) if s > 1 else x


def _window(x, kd, k):
    """(B,C,D,H,W) -> (B,C*kd*k*k,D,H,W): each voxel's kd×k×k neighbourhood (replicate-padded)."""
    C, P = x.shape[1], kd * k * k
    w = torch.eye(P, device=x.device, dtype=x.dtype).view(P, 1, kd, k, k).repeat(C, 1, 1, 1, 1)
    x = F.pad(x, (k // 2,) * 4 + (kd // 2,) * 2, mode="replicate")
    return F.conv3d(x, w, groups=C)


def _grad_maps(x):
    """Per-frame max and mean of |x - neighbour| over the 8 XY neighbours -> (B,2,D,H,W)."""
    H, W = x.shape[-2:]
    p = F.pad(x, (1, 1, 1, 1, 0, 0), mode="replicate")
    d = torch.stack([(p[..., 1 + dy:1 + dy + H, 1 + dx:1 + dx + W] - x).abs()
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    return torch.cat([d.amax(0), d.mean(0)], 1)


def _xgb_predict(booster, X):
    if cupy is not None and X.is_cuda:
        booster.set_param({"device": f"cuda:{X.device.index if X.device.index is not None else torch.cuda.current_device()}"})
        return torch.as_tensor(booster.inplace_predict(cupy.asarray(X)), device=X.device)
    # ponytail: without cupy, features round-trip through host RAM; `pip install cupy-cuda12x` keeps them on GPU
    booster.set_param({"device": "cpu"})
    return torch.from_numpy(booster.inplace_predict(X.cpu().numpy())).to(X.device)


class FeatGen(nn.Module):
    """Saab responses on a sparse neighbour grid + raw kd×k×k patch + optional gradient window."""

    def __init__(self, kernel_size, kernel_depth, neigh_size, neigh_depth, neigh_stride,
                 use_grad, grad_size, grad_depth):
        super().__init__()
        for name, v in [("kernel_size", kernel_size), ("kernel_depth", kernel_depth), ("neigh_size", neigh_size),
                        ("neigh_depth", neigh_depth), ("grad_size", grad_size), ("grad_depth", grad_depth)]:
            if v % 2 == 0:
                raise ValueError(f"{name} must be odd, got {v}")
        self.k, self.kd = kernel_size, kernel_depth
        self.use_grad, self.gs, self.gd = use_grad, grad_size, grad_depth
        self.r, self.rd = neigh_size // 2, neigh_depth // 2
        self.offsets = [(dz, dy, dx)
                        for dz in range(-self.rd, self.rd + 1, neigh_stride)
                        for dy in range(-self.r, self.r + 1, neigh_stride)
                        for dx in range(-self.r, self.r + 1, neigh_stride)]
        self.register_buffer("saab_w", torch.empty(0))
        self.register_buffer("saab_b", torch.empty(0))

    def patches(self, x):
        return _window(x, self.kd, self.k)

    @torch.no_grad()
    def fit(self, P):
        """Saab PCA on raw patches P (M, kd*k*k): DC kernel + AC principal components."""
        P = P.double()
        ac = P - P.mean(1, keepdim=True)
        mean0 = ac.mean(0, keepdim=True)
        X0 = ac - mean0
        _, eve = torch.linalg.eigh(X0.T @ X0)
        n = P.shape[1]
        dc = torch.full((1, n), n ** -0.5, dtype=P.dtype, device=P.device)
        K = torch.cat([dc, eve.T.flip(0)[:-1]])          # descending energy, drop the null (DC) direction
        self.saab_w = K.float().view(n, 1, self.kd, self.k, self.k)
        self.saab_b = -(K @ mean0.T).squeeze(1).float()  # transform = (x - mean0) @ K^T

    def forward(self, x):
        D, H, W = x.shape[-3:]
        r, rd = self.r, self.rd
        s = F.conv3d(F.pad(x, (self.k // 2,) * 4 + (self.kd // 2,) * 2, mode="replicate"), self.saab_w, self.saab_b)
        s = F.pad(s, (r, r, r, r, rd, rd), mode="replicate")
        feats = [s[:, :, rd + dz:rd + dz + D, r + dy:r + dy + H, r + dx:r + dx + W] for dz, dy, dx in self.offsets]
        feats.append(self.patches(x))
        if self.use_grad:
            feats.append(_window(_grad_maps(x), self.gd, self.gs))
        return torch.cat(feats, 1)


class Regressor(nn.Module):
    """RFT column pick -> append LNT projections -> XGBoost. (B,F,D,H,W) -> (B,1,D,H,W)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("rft_idx", torch.empty(0, dtype=torch.long))
        self.register_buffer("lnt_w", torch.empty(0))  # (m, F', 1, 1, 1)
        self.booster = None

    def project(self, f):
        f = f.index_select(1, self.rft_idx)
        return torch.cat([f, F.conv3d(f, self.lnt_w)], 1)

    def forward(self, f):
        if self.booster is None:
            raise RuntimeError("GUSL level not trained. Call fit() first.")
        f = self.project(f)
        B, C, D, H, W = f.shape
        y = _xgb_predict(self.booster, f.movedim(1, -1).reshape(-1, C).contiguous())
        return y.view(B, D, H, W).unsqueeze(1)


class GUSLLevel(nn.Module):
    def __init__(self, scale, residual, **fg_kwargs):
        super().__init__()
        self.scale = scale
        self.fg_img = FeatGen(**fg_kwargs)
        self.fg_res = FeatGen(**{**fg_kwargs, "use_grad": False}) if residual else None
        self.head = Regressor()

    def inputs(self, x, prev):
        """Full-res x + coarser prediction -> (x at this level's res, prediction upsampled to it)."""
        xl = _down(x, self.scale)
        if prev is None:
            return xl, None
        return xl, F.interpolate(prev, size=xl.shape[-3:], mode="trilinear", align_corners=False)

    def features(self, xl, prev_up):
        f = self.fg_img(xl)
        return f if prev_up is None else torch.cat([f, self.fg_res(xl - prev_up)], 1)

    def forward(self, x, prev):
        xl, prev_up = self.inputs(x, prev)
        r = self.head(self.features(xl, prev_up))
        return (r if prev_up is None else prev_up + r).clamp(0, 1)


class _Patches:
    """Iterates a Train dataset's shared patch tensor in batches, with the cached coarser prediction."""

    def __init__(self, ds, batch_size, device):
        self.imgs, self.msks = ds.image_tensors[0], ds.mask_tensors[0]
        self.idx = torch.tensor([m.volume_idx for m in ds.patch_indices])
        self.bs, self.device = batch_size, device
        self.prev = None  # (N,1,D,h,w) float16, prediction of the last trained level

    def __iter__(self):
        for i in range(0, len(self.idx), self.bs):
            j = self.idx[i:i + self.bs]
            x = self.imgs[j].to(self.device)
            m = (self.msks[j] > 0.5).float().to(self.device)
            p = None if self.prev is None else self.prev[i:i + self.bs].to(self.device).float()
            yield x, m, p


class GUSL(nn.Module):
    pad_div32 = False  # read by train.py / inference.py: keep patches at native size (no SwinUNETR padding)

    def __init__(
        self,
        levels=2,
        # per-level (deepest -> level 1, or one value for all)
        kernel_size=3, kernel_depth=3,
        neigh_size=3, neigh_depth=1, neigh_stride=2,
        use_grad=True, grad_size=3, grad_depth=3,
        n_bins=32, n_selected=500,
        lnt_depth=3, lnt_num_tree=150,
        boundary_window=5,
        # final XGBoost
        n_estimators=3000, max_depth=4, learning_rate=0.1, early_stopping_rounds=30, max_bin=256,
        # sampling (voxel counts per level)
        neg_keep_frac=0.15,
        saab_samples=200_000, encode_samples=2_000_000, decode_samples=8_000_000, val_samples=1_000_000,
        seed=42,
        spatial_dims=3,  # injected by train.py; 2 forces every depth extent to 1
    ):
        super().__init__()
        n = levels
        pl = lambda v, name: _per_level(v, n, name)
        if spatial_dims == 2:
            kernel_depth = neigh_depth = grad_depth = 1
        self.n_bins, self.n_selected = pl(n_bins, "n_bins"), pl(n_selected, "n_selected")
        self.lnt_depth, self.lnt_num_tree = pl(lnt_depth, "lnt_depth"), pl(lnt_num_tree, "lnt_num_tree")
        self.boundary_window = pl(boundary_window, "boundary_window")
        self.xgb_params = {"objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist",
                           "max_depth": max_depth, "eta": learning_rate, "max_bin": max_bin}
        self.n_estimators, self.early_stopping_rounds = n_estimators, early_stopping_rounds
        self.neg_keep_frac, self.seed = neg_keep_frac, seed
        self.saab_samples, self.encode_samples = saab_samples, encode_samples
        self.decode_samples, self.val_samples = decode_samples, val_samples

        fg = {k: pl(v, k) for k, v in dict(
            kernel_size=kernel_size, kernel_depth=kernel_depth, neigh_size=neigh_size, neigh_depth=neigh_depth,
            neigh_stride=neigh_stride, use_grad=use_grad, grad_size=grad_size, grad_depth=grad_depth).items()}
        self.levels = nn.ModuleList([
            GUSLLevel(scale=2 ** (n - 1 - i), residual=i > 0, **{k: v[i] for k, v in fg.items()})
            for i in range(n)
        ])

    def forward(self, x):
        is2d = x.ndim == 4
        if is2d:
            x = x.unsqueeze(2)
        pred = None
        for lvl in self.levels:
            pred = lvl(x, pred)
        p = pred.clamp(1e-4, 1 - 1e-4)
        logits = torch.log(p) - torch.log1p(-p)  # stitcher thresholds logits at 0 (= p 0.5)
        return logits.squeeze(2) if is2d else logits

    # ------------------------------------------------------------------ training

    def _weights(self, ml, li):
        """Sampling weight per voxel: positives and negatives near the mask = 1, other negatives = neg_keep_frac."""
        pos = ml > 0.5
        k = self.boundary_window[li]
        # ponytail: "near" = >=10% positives in a k×k window, same cut as the original boundary selection
        near = F.avg_pool3d(pos.float(), (1, k, k), stride=1, padding=(0, k // 2, k // 2), count_include_pad=False) >= 0.1
        return torch.where(pos | near, 1.0, self.neg_keep_frac)

    def _level_inputs(self, src, li):
        lvl = self.levels[li]
        for x, m, prev in src:
            xl, prev_up = lvl.inputs(x, prev)
            ml = _down(m, lvl.scale)
            yield xl, prev_up, (ml if prev_up is None else ml - prev_up), self._weights(ml, li)

    def _collect(self, src, li, cap, seed, fn):
        """Weighted random sample of ~cap voxels -> (fn(xl, prev_up) rows, targets) as numpy."""
        total = sum(float(w.sum()) for *_, w in self._level_inputs(src, li))
        rate = min(1.0, cap / max(total, 1.0))
        g = None
        Xs, ys = [], []
        for xl, prev_up, target, w in self._level_inputs(src, li):
            if g is None:
                g = torch.Generator(device=w.device).manual_seed(seed)
            keep = (torch.rand(w.shape, generator=g, device=w.device) < w * rate)[:, 0]
            Xs.append(fn(xl, prev_up).movedim(1, -1)[keep].cpu())
            ys.append(target[:, 0][keep].cpu())
        # ponytail: chunks + cat = 2x peak host RAM of the sample; preallocate if decode_samples grows huge
        return torch.cat(Xs).numpy(), torch.cat(ys).numpy()

    @torch.no_grad()
    def _cache_prediction(self, src, li):
        lvl = self.levels[li]
        src.prev = torch.cat([lvl(x, prev).half().cpu() for x, _, prev in src])

    @torch.no_grad()
    def fit(self, train_ds, val_ds, full_config, device):
        device = torch.device(device)
        self.to(device)
        bs = full_config.get("train", {}).get("training_batch_size", 16)
        tr, va = _Patches(train_ds, bs, device), _Patches(val_ds, bs, device)
        xgb_device = f"cuda:{device.index or 0}" if device.type == "cuda" else "cpu"
        logger.info(f"[GUSL] {len(tr.idx)} train / {len(va.idx)} val patches, batch={bs}, device={device}")

        for li, lvl in enumerate(self.levels):
            level = len(self.levels) - li
            seed = self.seed + 100 * li
            t0 = time.perf_counter()
            logger.info(f"[GUSL] ===== level {level} (XY scale 1/{lvl.scale}) =====")

            # 1. Saab kernels (image stream, + residual stream on finer levels)
            def raw(xl, prev_up):
                p = lvl.fg_img.patches(xl)
                return p if prev_up is None else torch.cat([p, lvl.fg_res.patches(xl - prev_up)], 1)
            P, _ = self._collect(tr, li, self.saab_samples, seed, raw)
            n = lvl.fg_img.kd * lvl.fg_img.k ** 2
            lvl.fg_img.fit(torch.from_numpy(P[:, :n]).to(device))
            if lvl.fg_res is not None:
                lvl.fg_res.fit(torch.from_numpy(P[:, n:]).to(device))
            del P
            logger.info(f"[GUSL] Saab fit ({time.perf_counter() - t0:.0f}s)")

            # 2. RFT feature selection + LNT projections on the encode sample
            X, y = self._collect(tr, li, self.encode_samples, seed + 1, lvl.features)
            Xv, yv = self._collect(va, li, self.val_samples, seed + 2, lvl.features)
            rft = DualRFT(n_bins=self.n_bins[li], n_selected=min(self.n_selected[li], X.shape[1]))
            rft.fit(X, y, Xv, yv)
            idx = rft.selected_features
            W = fit_lnt(X[:, idx], y, depth=self.lnt_depth[li], num_tree=self.lnt_num_tree[li], device=xgb_device)
            lvl.head.rft_idx = torch.from_numpy(idx).long().to(device)
            lvl.head.lnt_w = torch.from_numpy(W.T.copy()).float().view(W.shape[1], W.shape[0], 1, 1, 1).to(device)
            logger.info(f"[GUSL] features {X.shape[1]} -> RFT {len(idx)} + LNT {W.shape[1]}, "
                        f"encode {len(X)}/{len(Xv)} ({time.perf_counter() - t0:.0f}s)")
            del X, y, Xv, yv

            # 3. XGBoost on the decode sample
            project = lambda xl, prev_up: lvl.head.project(lvl.features(xl, prev_up))
            X, y = self._collect(tr, li, self.decode_samples, seed + 3, project)
            Xv, yv = self._collect(va, li, self.val_samples, seed + 4, project)
            logger.info(f"[GUSL] XGBoost decode {X.shape} train / {Xv.shape} val ({X.nbytes / 1e9:.1f} GB)")
            dtr = xgb.QuantileDMatrix(X, y, max_bin=self.xgb_params["max_bin"])
            dva = xgb.QuantileDMatrix(Xv, yv, ref=dtr)
            del X, y, Xv, yv
            booster = xgb.train({**self.xgb_params, "device": xgb_device}, dtr,
                                num_boost_round=self.n_estimators, evals=[(dva, "val")],
                                early_stopping_rounds=self.early_stopping_rounds, verbose_eval=100)
            lvl.head.booster = booster[: booster.best_iteration + 1]
            del dtr, dva
            logger.info(f"[GUSL] XGBoost best_iter={booster.best_iteration} val_rmse={booster.best_score:.4f} "
                        f"({time.perf_counter() - t0:.0f}s)")

            # 4. Dense prediction of this level, input to the next one
            if li < len(self.levels) - 1:
                self._cache_prediction(tr, li)
                self._cache_prediction(va, li)
            logger.info(f"[GUSL] level {level} done ({time.perf_counter() - t0:.0f}s)")
        return self
