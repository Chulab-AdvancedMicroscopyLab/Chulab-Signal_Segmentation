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


class MLPHead(nn.Module):
    """Standardise -> MLP -> scalar. hidden=[] is a plain linear model."""

    def __init__(self, mu, sd, hidden):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        layers, n = [], mu.numel()
        for h in hidden:
            layers += [nn.Linear(n, h), nn.ReLU()]
            n = h
        self.net = nn.Sequential(*layers, nn.Linear(n, 1))

    def forward(self, X):
        return self.net((X - self.mu) / self.sd).squeeze(1)


def _fit_mlp(X, y, Xv, yv, hidden, epochs, lr, device, seed):
    """Adam + MSE on the decode sample (kept on GPU), early stopping on validation MSE."""
    torch.manual_seed(seed)
    X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    Xv, yv = torch.from_numpy(Xv).to(device), torch.from_numpy(yv).to(device)
    head = MLPHead(X.mean(0), X.std(0).clamp_min(1e-6), hidden).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        with torch.enable_grad():
            for idx in torch.randperm(len(X), device=device).split(65536):
                loss = F.mse_loss(head(X[idx]), y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        val = sum(F.mse_loss(head(a), b, reduction="sum").item() for a, b in zip(Xv.split(262144), yv.split(262144))) / len(Xv)
        logger.info(f"[GUSL] MLP epoch {ep + 1}: val_rmse={val ** 0.5:.4f}")
        if val < best:
            best, best_state, bad = val, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:  # ponytail: fixed patience; make configurable if 5 epochs proves too short
                break
    head.load_state_dict(best_state)
    return head.requires_grad_(False), best ** 0.5


class Regressor(nn.Module):
    """RFT column pick -> append LNT projections -> XGBoost or MLP. (B,F,D,H,W) -> (B,1,D,H,W)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("rft_idx", torch.empty(0, dtype=torch.long))
        self.register_buffer("lnt_w", torch.empty(0))  # (m, F', 1, 1, 1)
        self.booster = None
        self.mlp = None

    def project(self, f):
        f = f.index_select(1, self.rft_idx)
        return torch.cat([f, F.conv3d(f, self.lnt_w)], 1)

    def forward(self, f):
        mlp = getattr(self, "mlp", None)  # absent in checkpoints saved before the MLP head existed
        if self.booster is None and mlp is None:
            raise RuntimeError("GUSL level not trained. Call fit() first.")
        f = self.project(f)
        B, C, D, H, W = f.shape
        X = f.movedim(1, -1).reshape(-1, C).contiguous()
        y = mlp(X) if mlp is not None else _xgb_predict(self.booster, X)
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
        finest_level=1,  # stop at this level (>1 = coarser output, upsampled to full resolution)
        # per-level (deepest -> level 1, or one value for all)
        kernel_size=3, kernel_depth=3,
        neigh_size=3, neigh_depth=1, neigh_stride=2,
        use_grad=True, grad_size=3, grad_depth=3,
        n_bins=32, n_selected=500,
        lnt_depth=3, lnt_num_tree=150,
        boundary_window=5,
        # final regressor: "xgboost" or "mlp" (mlp_hidden=[] -> linear)
        head="xgboost", mlp_hidden=(256, 128), mlp_epochs=50, mlp_lr=1e-3,
        n_estimators=3000, max_depth=4, learning_rate=0.1, early_stopping_rounds=30, max_bin=256,
        # sampling (voxel counts per level)
        neg_keep_frac=0.15,
        saab_samples=200_000, encode_samples=2_000_000, decode_samples=8_000_000, val_samples=1_000_000,
        seed=42,
        spatial_dims=3,  # injected by train.py; 2 forces every depth extent to 1
    ):
        super().__init__()
        if not 1 <= finest_level <= levels:
            raise ValueError(f"finest_level must be in 1..levels ({levels}), got {finest_level}")
        self.top_level = levels
        n = levels - finest_level + 1  # levels actually run: levels .. finest_level

        def pl(v, name):
            # per-level lists may list the run levels (n) or all configured levels (deepest first)
            if isinstance(v, (list, tuple)) and len(v) == levels and n < levels:
                v = list(v)[:n]
            return _per_level(v, n, name)
        if spatial_dims == 2:
            kernel_depth = neigh_depth = grad_depth = 1
        self.n_bins, self.n_selected = pl(n_bins, "n_bins"), pl(n_selected, "n_selected")
        self.lnt_depth, self.lnt_num_tree = pl(lnt_depth, "lnt_depth"), pl(lnt_num_tree, "lnt_num_tree")
        self.boundary_window = pl(boundary_window, "boundary_window")
        self.xgb_params = {"objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist",
                           "max_depth": max_depth, "eta": learning_rate, "max_bin": max_bin}
        if head not in ("xgboost", "mlp"):
            raise ValueError(f"head must be 'xgboost' or 'mlp', got {head!r}")
        self.head, self.mlp_hidden, self.mlp_epochs, self.mlp_lr = head, list(mlp_hidden), mlp_epochs, mlp_lr
        self.n_estimators, self.early_stopping_rounds = n_estimators, early_stopping_rounds
        self.neg_keep_frac, self.seed = neg_keep_frac, seed
        self.saab_samples, self.encode_samples = saab_samples, encode_samples
        self.decode_samples, self.val_samples = decode_samples, val_samples

        fg = {k: pl(v, k) for k, v in dict(
            kernel_size=kernel_size, kernel_depth=kernel_depth, neigh_size=neigh_size, neigh_depth=neigh_depth,
            neigh_stride=neigh_stride, use_grad=use_grad, grad_size=grad_size, grad_depth=grad_depth).items()}
        self.levels = nn.ModuleList([
            GUSLLevel(scale=2 ** (levels - 1 - i), residual=i > 0, **{k: v[i] for k, v in fg.items()})
            for i in range(n)
        ])

    def forward(self, x):
        is2d = x.ndim == 4
        if is2d:
            x = x.unsqueeze(2)
        pred = None
        for lvl in self.levels:
            pred = lvl(x, pred)
        if pred.shape[-3:] != x.shape[-3:]:  # finest_level > 1: upsample the coarse prediction
            pred = F.interpolate(pred, size=x.shape[-3:], mode="trilinear", align_corners=False)
        p = pred.clamp(1e-4, 1 - 1e-4)
        logits = torch.log(p) - torch.log1p(-p)  # stitcher thresholds logits at 0 (= p 0.5)
        return logits.squeeze(2) if is2d else logits

    # ------------------------------------------------------------------ cost accounting

    def flop_report(self):
        """
        Inference cost per output (full-resolution) voxel, per level, for a trained model.

        Returns a list of dicts (one per level) and a total. MACs count multiply-accumulates:
          macs_alg  - algorithmically required (Saab projection, LNT projection, MLP head); window
                      gathers (raw patch, gradient window, neighbour shifts) cost 0
          macs_impl - as implemented here (gathers run as identity convolutions)
          ops       - other elementwise ops (resampling, gradient maps, residual add, logit)
          tree_cmp  - XGBoost node comparisons (mean leaf depth x trees), not multiply-adds
        FLOPs = 2 x MACs. Coarser levels are scaled by the fraction of voxels they process.
        """
        rows = []
        for li, lvl in enumerate(self.levels):
            frac = 1.0 / lvl.scale ** 2                      # level processes 1/s^2 of the XY voxels
            macs_alg = macs_impl = ops = 0.0
            streams = [(lvl.fg_img, True)] + ([(lvl.fg_res, False)] if lvl.fg_res is not None else [])
            for fg, is_img in streams:
                P = fg.kd * fg.k ** 2
                n_k = fg.saab_w.shape[0] if fg.saab_w.numel() else P
                macs_alg += n_k * P                          # Saab projection
                macs_impl += n_k * P + P * P                 # + raw patch via identity conv
                if fg.use_grad:
                    G = fg.gd * fg.gs ** 2
                    ops += 8 * 4                             # 8 neighbour diffs: sub, abs, max, add
                    macs_impl += 2 * G * G                   # gradient window via identity conv
                if not is_img:
                    ops += 1 + 7                             # residual input (x - prev) + bilinear upsample
            if lvl.scale > 1:
                ops += lvl.scale ** 2                        # average-pool downsample
            h = lvl.head
            F_sel = int(h.rft_idx.numel())
            m = int(h.lnt_w.shape[0]) if h.lnt_w.numel() else 0
            macs_alg += F_sel * m; macs_impl += F_sel * m    # LNT projection
            tree_cmp = 0.0
            mlp = getattr(h, "mlp", None)
            if mlp is not None:
                lin = [mod for mod in mlp.net if isinstance(mod, nn.Linear)]
                mm = sum(mod.in_features * mod.out_features for mod in lin) + (F_sel + m)  # + standardise
                macs_alg += mm; macs_impl += mm
            elif h.booster is not None:
                df = h.booster.trees_to_dataframe()
                parents = dict(zip(df["Yes"].dropna(), df.loc[df["Yes"].notna(), "ID"]))
                parents.update(dict(zip(df["No"].dropna(), df.loc[df["No"].notna(), "ID"])))
                def d(node):
                    k = 0
                    while node in parents:
                        node = parents[node]; k += 1
                    return k
                leaves = df.loc[df["Feature"] == "Leaf", "ID"]
                mean_leaf_depth = float(sum(d(n) for n in leaves) / max(len(leaves), 1))
                n_trees = h.booster.num_boosted_rounds()
                tree_cmp = n_trees * mean_leaf_depth
                ops += n_trees                               # leaf-value accumulation
            ops += 1                                         # residual add / clamp
            rows.append({"level": getattr(self, "top_level", len(self.levels)) - li, "voxel_fraction": frac,
                         "macs_alg": macs_alg * frac, "macs_impl": macs_impl * frac,
                         "ops": ops * frac, "tree_cmp": tree_cmp * frac,
                         "features": F_sel, "lnt": m})
        total = {k: sum(r[k] for r in rows) for k in ("macs_alg", "macs_impl", "ops", "tree_cmp")}
        total["ops"] += 4                                    # final logit
        if self.levels[-1].scale > 1:
            total["ops"] += 7                                # upsample coarse output to full resolution
        return rows, total

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

    def _collect(self, src, li, cap, seed, fn, on_gpu=False):
        """Weighted random sample of ~cap voxels -> (fn(xl, prev_up) rows, targets) as numpy.

        Three passes over the patches: sampling-weight total, a cheap count of the kept voxels, then
        the feature pass writing straight into an exact-size array. The seeded generator draws the same
        selection in the last two passes, so the sample is held once (no chunk list + concat).
        on_gpu=True keeps it in GPU memory as torch tensors instead of host numpy arrays.
        """
        total = sum(float(w.sum()) for *_, w in self._level_inputs(src, li))
        rate = min(1.0, cap / max(total, 1.0))

        def kept():
            g = None
            for xl, prev_up, target, w in self._level_inputs(src, li):
                if g is None:
                    g = torch.Generator(device=w.device).manual_seed(seed)
                yield xl, prev_up, target, (torch.rand(w.shape, generator=g, device=w.device) < w * rate)[:, 0]

        n = sum(int(keep.sum()) for *_, keep in kept())
        X = y = None
        pos = 0
        for xl, prev_up, target, keep in kept():
            rows = fn(xl, prev_up).movedim(1, -1)[keep]
            if X is None:
                if on_gpu:
                    X = torch.empty((n, rows.shape[1]), dtype=torch.float32, device=rows.device)
                    y = torch.empty(n, dtype=torch.float32, device=rows.device)
                else:
                    X = np.empty((n, rows.shape[1]), dtype=np.float32)
                    y = np.empty(n, dtype=np.float32)
            X[pos:pos + len(rows)] = rows if on_gpu else rows.cpu().numpy()
            y[pos:pos + len(rows)] = target[:, 0][keep] if on_gpu else target[:, 0][keep].cpu().numpy()
            pos += len(rows)
        assert pos == n, f"sample count changed between passes ({pos} vs {n})"
        return X, y

    def _fit_xgb(self, lvl, data, xgb_device, t0):
        """data = [X, y, Xv, yv]; emptied once the quantile matrices are built so host RAM is freed."""
        X, y, Xv, yv = [cupy.asarray(a) if torch.is_tensor(a) else a for a in data]  # GPU tensors: zero-copy
        dtr = xgb.QuantileDMatrix(X, y, max_bin=self.xgb_params["max_bin"])
        dva = xgb.QuantileDMatrix(Xv, yv, ref=dtr)
        data.clear()
        del X, y, Xv, yv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        booster = xgb.train({**self.xgb_params, "device": xgb_device}, dtr,
                            num_boost_round=self.n_estimators, evals=[(dva, "val")],
                            early_stopping_rounds=self.early_stopping_rounds, verbose_eval=100)
        lvl.head.booster = booster[: booster.best_iteration + 1]
        logger.info(f"[GUSL] XGBoost best_iter={booster.best_iteration} val_rmse={booster.best_score:.4f} "
                    f"({time.perf_counter() - t0:.0f}s)")

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
            level = getattr(self, "top_level", len(self.levels)) - li
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
            n_feat, n_enc, n_val = X.shape[1], len(X), len(Xv)
            X = X[:, idx]          # LNT only needs the selected columns: free the full encode/val samples first
            del Xv, yv
            W = fit_lnt(X, y, depth=self.lnt_depth[li], num_tree=self.lnt_num_tree[li], device=xgb_device)
            del X, y
            lvl.head.rft_idx = torch.from_numpy(idx).long().to(device)
            lvl.head.lnt_w = torch.from_numpy(W.T.copy()).float().view(W.shape[1], W.shape[0], 1, 1, 1).to(device)
            logger.info(f"[GUSL] features {n_feat} -> RFT {len(idx)} + LNT {W.shape[1]}, "
                        f"encode {n_enc}/{n_val} ({time.perf_counter() - t0:.0f}s)")

            # 3. Final regressor on the decode sample
            project = lambda xl, prev_up: lvl.head.project(lvl.features(xl, prev_up))
            # XGBoost on GPU with cupy: keep the decode sample in GPU memory (no host copy at all)
            on_gpu = self.head == "xgboost" and device.type == "cuda" and cupy is not None
            X, y = self._collect(tr, li, self.decode_samples, seed + 3, project, on_gpu=on_gpu)
            Xv, yv = self._collect(va, li, self.val_samples, seed + 4, project, on_gpu=on_gpu)
            logger.info(f"[GUSL] {self.head} decode {tuple(X.shape)} train / {tuple(Xv.shape)} val "
                        f"({X.nbytes / 1e9:.1f} GB, {'GPU' if on_gpu else 'host'})")
            if self.head == "mlp":
                lvl.head.mlp, rmse = _fit_mlp(X, y, Xv, yv, self.mlp_hidden, self.mlp_epochs, self.mlp_lr, device, seed)
                del X, y, Xv, yv
                logger.info(f"[GUSL] MLP {self.mlp_hidden} val_rmse={rmse:.4f} ({time.perf_counter() - t0:.0f}s)")
            else:
                data = [X, y, Xv, yv]
                del X, y, Xv, yv
                self._fit_xgb(lvl, data, xgb_device, t0)

            # 4. Dense prediction of this level, input to the next one
            if li < len(self.levels) - 1:
                self._cache_prediction(tr, li)
                self._cache_prediction(va, li)
            logger.info(f"[GUSL] level {level} done ({time.perf_counter() - t0:.0f}s)")
        return self
