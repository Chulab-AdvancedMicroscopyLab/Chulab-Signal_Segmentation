import logging
from typing import Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD
from joblib import Parallel, delayed
import xgboost as xgb


def _select_comb(gt):
    num = len(gt)
    gt_comb = np.zeros((num, 2))
    gt_comb[:, 0] = 1 - gt
    gt_comb[:, 1] = gt
    return gt_comb


class LNT:
    def __init__(self, depth, num_tree, mode, lnt_sample_frac: Optional[float] = 0.5):
        self.svd = []
        self.depth = depth
        self.num_tree = num_tree
        self.mode = mode
        self.lnt_sample_frac = lnt_sample_frac

    def fit(self, x, y):
        frac = self.lnt_sample_frac
        if frac and frac < 1.0:
            rng = np.random.default_rng(42)
            n = max(1, int(x.shape[0] * frac))
            idx = rng.choice(x.shape[0], n, replace=False)
            logging.info(f"[LNT] Subsampling for fit: {x.shape[0]}→{n} ({frac:.0%})")
            x, y = x[idx], y[idx]

        params = {
            "objective": "reg:squarederror",
            "max_depth": self.depth,
            "min_child_weight": 1,
            "subsample": 1,
            "colsample_bytree": 1,
            "n_estimators": self.num_tree,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "device": "cuda" if self.mode == "gpu" else "cpu",
        }

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        model = xgb.XGBRegressor(**params)
        model.fit(x, y)

        trees_df = model.get_booster().trees_to_dataframe()
        split_nodes = trees_df[trees_df["Feature"] != "Leaf"]
        per_tree_features = (
            split_nodes.groupby("Tree")["Feature"]
            .apply(lambda s: tuple(sorted(int(f[1:]) for f in s.unique())))
        )
        tree_features = [np.array(list(fs)) for fs in set(per_tree_features)]
        logging.info(f"[LNT] feature dimension {len(tree_features)}")

        num_features = x.shape[-1]
        non_empty = [list(map(int, fs)) for fs in tree_features if len(fs) > 0]

        def _fit_one(selected):
            lr = LinearRegression()
            lr.fit(x[:, selected], y)
            theta = np.zeros(num_features, dtype=np.float32)
            theta[selected] = lr.coef_.astype(np.float32, copy=False)
            return theta

        thetas = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_fit_one)(sel) for sel in non_empty
        )
        A = np.column_stack(thetas).astype(np.float32) if thetas else np.zeros((num_features, 0), dtype=np.float32)
        self.svd.append(A.astype(np.float32, copy=False))

        x0 = x - np.mean(x, axis=0, dtype=np.float32)
        y_comb = _select_comb(y).astype(np.float32, copy=False)
        y0 = y_comb - np.mean(y_comb, axis=0, dtype=np.float32)
        # Normal equations: avoid single-threaded LAPACK SVD in LinearRegression.
        # x0.T @ x0 is a multi-threaded BLAS dgemm (p×p), solve is trivial.
        XtX = x0.T @ x0 + 1e-6 * np.eye(x0.shape[1], dtype=np.float32)
        Xty = x0.T @ y0
        T = np.linalg.solve(XtX, Xty).astype(np.float32)
        svd = TruncatedSVD(n_components=T.shape[1] - 1, random_state=42).fit(T.astype(np.float32, copy=False))
        U = svd.transform(T).astype(np.float32, copy=False)
        U = U[:, np.argsort(svd.explained_variance_ratio_)[::-1]].astype(np.float32, copy=False)
        self.svd.append(U)

    def transform(self, x, y=None):
        import time as _t
        x = np.asarray(x, dtype=np.float32)
        _diag = not getattr(self, "_transform_logged", False)
        if _diag:
            self._transform_logged = True
            logging.info(f"[LNT] mode={self.mode} x={x.shape} svd_shapes={[s.shape for s in self.svd]}")
        if self.mode == "gpu":
            import torch
            device = "cuda"
            t0 = _t.perf_counter()
            x_t = torch.from_numpy(np.ascontiguousarray(x)).to(device)
            if _diag:
                logging.info(f"[LNT] to_cuda: {_t.perf_counter()-t0:.3f}s")
            res = []
            for i, temp_kernel in enumerate(self.svd):
                t0 = _t.perf_counter()
                k_t = torch.from_numpy(np.ascontiguousarray(temp_kernel)).to(device)
                out = (x_t @ k_t).cpu().numpy().astype(np.float32)
                res.append(out)
                if _diag:
                    logging.info(f"[LNT] gpu svd[{i}] {temp_kernel.shape}: {_t.perf_counter()-t0:.3f}s → {out.shape}")
            del x_t
            return np.concatenate(res, axis=-1).astype(np.float32, copy=False)
        res = []
        for i, temp_kernel in enumerate(self.svd):
            t0 = _t.perf_counter()
            k = np.asarray(temp_kernel, dtype=np.float32)
            out = (x @ k).astype(np.float32, copy=False)
            res.append(out)
            if _diag:
                logging.info(f"[LNT] cpu svd[{i}] {k.shape}: {_t.perf_counter()-t0:.3f}s → {out.shape}")
        return np.concatenate(res, axis=-1).astype(np.float32, copy=False)
