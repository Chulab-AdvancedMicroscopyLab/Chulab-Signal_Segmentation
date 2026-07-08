import logging
import numpy as np
from typing import List, Tuple, Optional, Union
from numpy.lib.stride_tricks import sliding_window_view

from .saab import SaabTorch

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

if _NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _gather_neighbor_patches(padded, neigh_n, neigh_i, neigh_j, r_nd, r_n, kd, k):
        K = neigh_n.shape[0]
        nb = neigh_n.shape[1]
        patch_dim = kd * k * k
        out = np.empty((K * nb, patch_dim), np.float32)
        for idx in prange(K * nb):
            ki = idx // nb
            bi = idx % nb
            n_start = neigh_n[ki, bi] + r_nd
            i_start = neigh_i[ki, bi] + r_n
            j_start = neigh_j[ki, bi] + r_n
            p = 0
            for d in range(kd):
                for y in range(k):
                    for x in range(k):
                        out[idx, p] = padded[n_start + d, i_start + y, j_start + x]
                        p += 1
        return out

    @njit(parallel=True, cache=True)
    def _gather_center_patches(padded, n0, i0, j0, r_nd, r_n, kd, k):
        """Gather center-pixel patches — replaces fancy indexing into sliding_window_view."""
        nb = n0.shape[0]
        patch_dim = kd * k * k
        out = np.empty((nb, patch_dim), np.float32)
        for bi in prange(nb):
            n_start = n0[bi] + r_nd
            i_start = i0[bi] + r_n
            j_start = j0[bi] + r_n
            p = 0
            for d in range(kd):
                for y in range(k):
                    for x in range(k):
                        out[bi, p] = padded[n_start + d, i_start + y, j_start + x]
                        p += 1
        return out

    @njit(parallel=True, cache=True)
    def _gather_grad_patches(gmax, gavg, n0, i0, j0, r_gd, r_gxy, gd, gs, N, H, W):
        """Gather gradient windows directly from gmax/gavg with reflect padding.
        Replaces sliding_window_view fancy indexing — avoids 12GB of padded arrays
        and the ~150MB-stride cache-miss pattern on the view's depth axis.
        Output layout matches np.stack([gm, ga], axis=2).reshape(nb, gd*gs*gs*2):
        [gmax[d,y,x], gavg[d,y,x], gmax[d,y,x+1], ...] (interleaved per position).
        """
        nb = n0.shape[0]
        grad_len = gd * gs * gs
        out = np.empty((nb, grad_len * 2), np.float32)
        for bi in prange(nb):
            p = 0
            for d in range(gd):
                nd = n0[bi] + d - r_gd
                if nd < 0:
                    nd = -nd
                elif nd >= N:
                    nd = 2 * (N - 1) - nd
                for y in range(gs):
                    iy = i0[bi] + y - r_gxy
                    if iy < 0:
                        iy = -iy
                    elif iy >= H:
                        iy = 2 * (H - 1) - iy
                    for x in range(gs):
                        jx = j0[bi] + x - r_gxy
                        if jx < 0:
                            jx = -jx
                        elif jx >= W:
                            jx = 2 * (W - 1) - jx
                        out[bi, 2 * p]     = gmax[nd, iy, jx]
                        out[bi, 2 * p + 1] = gavg[nd, iy, jx]
                        p += 1
        return out


def _reflect_index(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x)
    if L <= 1:
        return np.zeros_like(x, dtype=np.int64)
    period = 2 * L - 2
    xm = np.abs(x) % period
    return np.where(xm < L, xm, period - xm).astype(np.int64)


class FeatGen3D:
    """
    3D feature generator: Saab patches + neighborhood + optional gradient features.
    Input images expected as (N, H, W) where N is treated as depth.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        neigh_size: int = 3,
        neigh_stride: int = 1,
        kernel_depth: int = 3,
        neigh_depth: int = 1,
        needBias: bool = True,
        saab_num_kernels: int = -1,
        saab_bias: float = 0.0,
        random_state: int = 0,
        max_saab_samples: Optional[int] = None,
        grad_size: int = 3,
        grad_depth: int = 3,
        use_grad: bool = True,
        batch_centers: int = 1_000_000,
        device: Optional[str] = None,
    ):
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        assert neigh_size % 2 == 1, "neigh_size must be odd"
        assert isinstance(neigh_stride, int) and neigh_stride >= 1
        assert kernel_depth % 2 == 1, "kernel_depth must be odd"
        assert neigh_depth >= 1

        self.k = int(kernel_size)
        self.kd = int(kernel_depth)
        self.nw = int(neigh_size)
        self.nw_d = int(neigh_depth)
        self.neigh_stride = int(neigh_stride)

        self.use_grad = bool(use_grad)
        if self.use_grad:
            assert grad_size % 2 == 1, "grad_size must be odd"
            assert grad_depth >= 1
        self.grad_size = int(grad_size)
        self.grad_depth = int(grad_depth)

        self.batch_centers = int(batch_centers)

        self.r_k = (self.k - 1) // 2
        self.r_kd = (self.kd - 1) // 2
        self.r_n = (self.nw - 1) // 2
        self.r_nd = (self.nw_d - 1) // 2
        self.r_gxy = (self.grad_size - 1) // 2
        self.r_gd = (self.grad_depth - 1) // 2

        self.pad_xy = self.r_k + self.r_n
        self.pad_d = self.r_kd + self.r_nd

        self.saab = SaabTorch(num_kernels=saab_num_kernels, needBias=needBias,
                              bias=saab_bias, device=device)
        self.trained = False

        self.random_state = int(random_state)
        self.max_saab_samples = max_saab_samples

        self._neigh_idx_xy_1d: List[int] = list(range(0, self.nw, self.neigh_stride))
        self._neigh_off_xy_1d: List[int] = [idx - self.r_n for idx in self._neigh_idx_xy_1d]
        self._neigh_idx_d_1d: List[int] = list(range(0, self.nw_d, self.neigh_stride))
        self._neigh_off_d_1d: List[int] = [idx - self.r_nd for idx in self._neigh_idx_d_1d]

        self.saab_dim = None
        self.raw_dim = None
        self.grad_dim = None
        self.feature_dim = None

    @staticmethod
    def _ensure_nhw(x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("Expected (N,H,W) with N as depth.")
        return np.asarray(x, dtype=np.float32)

    def _pad_volume(self, imgs: np.ndarray) -> np.ndarray:
        imgs = self._ensure_nhw(imgs)
        return np.pad(
            imgs,
            pad_width=((self.pad_d, self.pad_d),
                       (self.pad_xy, self.pad_xy),
                       (self.pad_xy, self.pad_xy)),
            mode="reflect",
        ).astype(np.float32, copy=False)

    def _center_patch_view(self, imgs: np.ndarray) -> np.ndarray:
        imgs = self._ensure_nhw(imgs)
        N, H, W = imgs.shape
        padded = self._pad_volume(imgs)
        win = sliding_window_view(padded, window_shape=(self.kd, self.k, self.k))
        return win[self.r_nd:self.r_nd + N, self.r_n:self.r_n + H, self.r_n:self.r_n + W]

    def _precompute_grad_maps(self, imgs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        imgs = self._ensure_nhw(imgs)
        N, H, W = imgs.shape
        pad = 1
        padded = np.pad(imgs, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
        center = padded[:, 1:1 + H, 1:1 + W]
        shifts = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        # Running max/mean: O(2×volume) peak instead of O(16×volume) from np.stack.
        gmax = np.zeros((N, H, W), dtype=np.float32)
        gavg = np.zeros((N, H, W), dtype=np.float32)
        for dy, dx in shifts:
            nbr = padded[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            diff = np.abs(nbr - center).astype(np.float32)
            np.maximum(gmax, diff, out=gmax)
            gavg += diff
        gavg *= (1.0 / len(shifts))
        return gmax, gavg

    def _grad_window_views(self, gmax: np.ndarray, gavg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gmax = self._ensure_nhw(gmax)
        gavg = self._ensure_nhw(gavg)
        N, H, W = gmax.shape
        pad_d, pad_xy = self.r_gd, self.r_gxy
        gmax_p = np.pad(gmax, ((pad_d, pad_d), (pad_xy, pad_xy), (pad_xy, pad_xy)), mode="reflect")
        gavg_p = np.pad(gavg, ((pad_d, pad_d), (pad_xy, pad_xy), (pad_xy, pad_xy)), mode="reflect")
        vw_max = sliding_window_view(gmax_p, window_shape=(self.grad_depth, self.grad_size, self.grad_size))
        vw_avg = sliding_window_view(gavg_p, window_shape=(self.grad_depth, self.grad_size, self.grad_size))
        return vw_max[0:N, 0:H, 0:W], vw_avg[0:N, 0:H, 0:W]

    def _saab_out_dim(self) -> int:
        if getattr(self.saab, "Kernels", None) is not None:
            return int(self.saab.Kernels.shape[0])
        dummy = np.zeros((1, self.kd * self.k * self.k), dtype=np.float32)
        return int(self.saab.transform(dummy).shape[1])

    def fit(self, images: np.ndarray, gt: np.ndarray,
            sel_indices: Optional[np.ndarray] = None) -> "FeatGen3D":
        imgs = self._ensure_nhw(images)
        _ = self._ensure_nhw(gt)
        N, H, W = imgs.shape
        total = N * H * W
        patch_view = self._center_patch_view(imgs)

        if sel_indices is not None:
            lin = np.asarray(sel_indices, dtype=np.int64).reshape(-1)
        else:
            if self.max_saab_samples is not None and total > self.max_saab_samples:
                rng = np.random.RandomState(self.random_state)
                lin = rng.choice(total, size=self.max_saab_samples, replace=False)
            else:
                lin = np.arange(total)

        n = lin // (H * W)
        rem = lin % (H * W)
        i = rem // W
        j = rem % W

        patches_fit = patch_view[n, i, j].reshape(len(lin), -1).astype(np.float32)
        self.saab.fit(patches_fit)
        self.trained = True

        out_per_patch = self._saab_out_dim()
        num_neighbors = len(self._neigh_off_d_1d) * (len(self._neigh_off_xy_1d) ** 2)
        self.saab_dim = out_per_patch * num_neighbors
        self.raw_dim = self.kd * self.k * self.k
        self.grad_dim = (self.grad_depth * self.grad_size * self.grad_size * 2 if self.use_grad else 0)
        self.feature_dim = self.saab_dim + self.raw_dim + self.grad_dim
        return self

    def prepare(self, images: np.ndarray) -> dict:
        """Precompute padded volume and gradient maps once for the full input.
        Pass the returned context to transform_precomputed() for each pixel batch.
        Stores raw gmax/gavg (not padded views) — avoids ~12GB of duplicate
        padded gradient arrays and eliminates the depth-stride cache-miss pattern."""
        imgs = self._ensure_nhw(images)
        N, H, W = imgs.shape
        padded = self._pad_volume(imgs)
        if self.use_grad:
            gmax, gavg = self._precompute_grad_maps(imgs)
            grad_len = self.grad_depth * self.grad_size * self.grad_size
        else:
            gmax = gavg = None
            grad_len = 0
        return {
            "N": N, "H": H, "W": W,
            "padded": padded,
            "gmax": gmax,
            "gavg": gavg,
            "grad_len": grad_len,
        }

    def transform_precomputed(self, ctx: dict, keep_lin: np.ndarray) -> np.ndarray:
        """Extract features for pixels at keep_lin using a context from prepare()."""
        if not self.trained:
            raise RuntimeError("Saab not trained. Call fit() first.")
        N, H, W = ctx["N"], ctx["H"], ctx["W"]
        padded   = ctx["padded"]
        gmax     = ctx["gmax"]
        gavg     = ctx["gavg"]
        grad_len = ctx["grad_len"]

        keep_lin = np.asarray(keep_lin, dtype=np.int64)
        nb = keep_lin.size
        n0 = keep_lin // (H * W)
        rem = keep_lin % (H * W)
        i0 = rem // W
        j0 = rem % W

        neigh_offsets = [(dn, di, dj)
                         for dn in self._neigh_off_d_1d
                         for di in self._neigh_off_xy_1d
                         for dj in self._neigh_off_xy_1d]
        num_neighbors = len(neigh_offsets)
        patch_dim = self.kd * self.k * self.k

        neigh_n = np.stack([_reflect_index(n0 + dn, N) for dn, di, dj in neigh_offsets])
        neigh_i = np.stack([_reflect_index(i0 + di, H) for dn, di, dj in neigh_offsets])
        neigh_j = np.stack([_reflect_index(j0 + dj, W) for dn, di, dj in neigh_offsets])

        if _NUMBA_AVAILABLE:
            P = _gather_neighbor_patches(
                padded,
                neigh_n.astype(np.int64), neigh_i.astype(np.int64), neigh_j.astype(np.int64),
                self.r_nd, self.r_n, self.kd, self.k,
            )
        else:
            win = sliding_window_view(padded, window_shape=(self.kd, self.k, self.k))
            patch_view = win[self.r_nd:self.r_nd + N, self.r_n:self.r_n + H, self.r_n:self.r_n + W]
            P = patch_view[neigh_n, neigh_i, neigh_j].reshape(num_neighbors * nb, patch_dim).astype(np.float32, copy=False)

        out_per_patch = self._saab_out_dim()
        Y = self.saab.transform(P).astype(np.float32, copy=False)
        saab_feat = Y.reshape(num_neighbors, nb, out_per_patch).transpose(1, 0, 2).reshape(nb, -1)
        del P, Y

        if _NUMBA_AVAILABLE:
            raw_feat = _gather_center_patches(padded, n0, i0, j0, self.r_nd, self.r_n, self.kd, self.k)
        else:
            win = sliding_window_view(padded, window_shape=(self.kd, self.k, self.k))
            patch_view = win[self.r_nd:self.r_nd + N, self.r_n:self.r_n + H, self.r_n:self.r_n + W]
            raw_feat = patch_view[n0, i0, j0].reshape(nb, -1).astype(np.float32, copy=False)

        if self.use_grad:
            if _NUMBA_AVAILABLE:
                grad_feat = _gather_grad_patches(
                    gmax, gavg, n0, i0, j0,
                    self.r_gd, self.r_gxy,
                    self.grad_depth, self.grad_size,
                    N, H, W,
                )
            else:
                grad_vw_max, grad_vw_avg = self._grad_window_views(gmax, gavg)
                gm = grad_vw_max[n0, i0, j0].reshape(nb, grad_len)
                ga = grad_vw_avg[n0, i0, j0].reshape(nb, grad_len)
                grad_feat = np.stack([gm, ga], axis=2).reshape(nb, grad_len * 2).astype(np.float32, copy=False)
            return np.concatenate([saab_feat, raw_feat, grad_feat], axis=1).astype(np.float32, copy=False)
        return np.concatenate([saab_feat, raw_feat], axis=1).astype(np.float32, copy=False)

    def transform(
        self,
        images: np.ndarray,
        gt: Optional[np.ndarray] = None,
        sel_mask_flat: Optional[np.ndarray] = None,
        verbose: bool = True,
        print_every: int = 1,
        name: str = "FeatGen3D",
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if not self.trained:
            raise RuntimeError("Saab not trained. Call fit() first.")

        imgs = self._ensure_nhw(images)
        N, H, W = imgs.shape
        total = N * H * W

        if sel_mask_flat is not None:
            sel_mask_flat = np.asarray(sel_mask_flat, dtype=bool).reshape(-1)
            if sel_mask_flat.size != total:
                raise ValueError(f"sel_mask_flat size {sel_mask_flat.size} != {total}")
            keep_lin = np.where(sel_mask_flat)[0].astype(np.int64)
        else:
            keep_lin = np.arange(total, dtype=np.int64)

        n_keep = int(keep_lin.size)
        F = int(self.feature_dim)
        feats = np.empty((n_keep, F), dtype=np.float32)

        y_full = None
        if gt is not None:
            gt_arr = self._ensure_nhw(gt)
            y_full = gt_arr.reshape(-1).astype(np.float32, copy=False)

        padded = self._pad_volume(imgs)
        win = sliding_window_view(padded, window_shape=(self.kd, self.k, self.k))
        patch_view = win[self.r_nd:self.r_nd + N, self.r_n:self.r_n + H, self.r_n:self.r_n + W]

        if self.use_grad:
            gmax, gavg = self._precompute_grad_maps(imgs)
            grad_vw_max, grad_vw_avg = self._grad_window_views(gmax, gavg)
            grad_len = self.grad_depth * self.grad_size * self.grad_size
        else:
            grad_vw_max = grad_vw_avg = None
            grad_len = 0

        out_per_patch = self._saab_out_dim()
        off_xy = self._neigh_off_xy_1d
        off_d = self._neigh_off_d_1d
        num_neighbors = len(off_d) * (len(off_xy) ** 2)
        patch_dim = self.kd * self.k * self.k
        neigh_offsets = [(dn, di, dj) for dn in off_d for di in off_xy for dj in off_xy]

        write_pos = 0
        B = self.batch_centers
        num_batches = (n_keep + B - 1) // B

        for batch_idx, start in enumerate(range(0, n_keep, B), start=1):
            end = min(n_keep, start + B)
            lin_b = keep_lin[start:end]
            nb = lin_b.size

            n0 = lin_b // (H * W)
            rem = lin_b % (H * W)
            i0 = rem // W
            j0 = rem % W

            neigh_n = np.stack([_reflect_index(n0 + dn, N) for dn, di, dj in neigh_offsets])
            neigh_i = np.stack([_reflect_index(i0 + di, H) for dn, di, dj in neigh_offsets])
            neigh_j = np.stack([_reflect_index(j0 + dj, W) for dn, di, dj in neigh_offsets])

            if _NUMBA_AVAILABLE:
                P = _gather_neighbor_patches(
                    padded,
                    neigh_n.astype(np.int64), neigh_i.astype(np.int64), neigh_j.astype(np.int64),
                    self.r_nd, self.r_n, self.kd, self.k,
                )
            else:
                P = patch_view[neigh_n, neigh_i, neigh_j].reshape(num_neighbors * nb, patch_dim).astype(np.float32, copy=False)

            Y = self.saab.transform(P).astype(np.float32, copy=False)
            saab_feat = Y.reshape(num_neighbors, nb, out_per_patch).transpose(1, 0, 2).reshape(nb, -1)

            _n0i = n0.astype(np.int64); _i0i = i0.astype(np.int64); _j0i = j0.astype(np.int64)
            if _NUMBA_AVAILABLE:
                raw_feat = _gather_center_patches(padded, _n0i, _i0i, _j0i, self.r_nd, self.r_n, self.kd, self.k)
            else:
                raw_feat = patch_view[n0, i0, j0].reshape(nb, -1).astype(np.float32, copy=False)
            if self.use_grad:
                gm = grad_vw_max[n0, i0, j0].reshape(nb, grad_len)
                ga = grad_vw_avg[n0, i0, j0].reshape(nb, grad_len)
                grad_feat = np.stack([gm, ga], axis=2).reshape(nb, grad_len * 2).astype(np.float32, copy=False)
                feats_b = np.concatenate([saab_feat, raw_feat, grad_feat], axis=1).astype(np.float32, copy=False)
            else:
                feats_b = np.concatenate([saab_feat, raw_feat], axis=1).astype(np.float32, copy=False)

            feats[write_pos:write_pos + nb, :] = feats_b
            write_pos += nb

            if verbose and (batch_idx == 1 or batch_idx % print_every == 0 or batch_idx == num_batches):
                pct = 100.0 * write_pos / max(n_keep, 1)
                logging.info(f"[{name}] batch {batch_idx}/{num_batches} | {write_pos}/{n_keep} ({pct:.1f}%)")

            del lin_b, n0, i0, j0, neigh_n, neigh_i, neigh_j, P, Y, saab_feat, raw_feat, feats_b
            if self.use_grad:
                del gm, ga, grad_feat

        if write_pos != n_keep:
            raise RuntimeError(f"[FeatGen3D] wrote {write_pos} rows, expected {n_keep}")

        if gt is None:
            return feats

        y = y_full[keep_lin] if sel_mask_flat is not None else y_full
        return feats, y
