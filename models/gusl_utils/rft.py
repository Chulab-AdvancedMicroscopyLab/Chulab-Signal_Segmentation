import numpy as np
from joblib import Parallel, delayed


class FeatureTest:
    def __init__(self, loss="bce"):
        assert loss in ["bce", "ce", "rmse"]
        self.loss = loss
        self.dim_loss = dict()
        self.sorted_features = None
        self.dim = 0

    def fit(self, X, y, n_bins, outliers=False):
        self.dim = X.shape[1]
        losses = Parallel(n_jobs=-1, prefer="threads")(
            delayed(self.get_min_partition_loss)(X[:, d], y, n_bins, outliers)
            for d in range(self.dim)
        )
        self.dim_loss = dict(enumerate(losses))
        self.dim_loss = {k: v for k, v in sorted(self.dim_loss.items(), key=lambda item: item[1])}
        self.sorted_features = np.array(list(self.dim_loss.keys()))

    def transform(self, X, n_selected):
        assert self.sorted_features is not None
        assert X.shape[1] == self.dim
        return X[:, self.sorted_features[np.arange(n_selected)]]

    def fit_transform(self, X, y, n_bins, n_selected):
        self.fit(X, y, n_bins)
        return self.transform(X, n_selected)

    def get_min_partition_loss(self, f_1d, y, n_bins, outliers=False):
        if outliers:
            f_1d, y = self.remove_outliers(f_1d, y)
        f_min, f_max = float(f_1d.min()), float(f_1d.max())
        if f_max == f_min or self.loss not in ("rmse", "bce"):
            min_partition_loss = float("inf")
            bin_width = (f_max - f_min) / n_bins if f_max != f_min else 1.0
            for i in range(1, n_bins):
                partition_point = f_min + i * bin_width
                y_l, y_r = y[f_1d <= partition_point], y[f_1d > partition_point]
                partition_loss = self.get_loss(y_l, y_r)
                if partition_loss < min_partition_loss:
                    min_partition_loss = partition_loss
            return min_partition_loss

        order = np.argsort(f_1d, kind="quicksort")
        f_s = f_1d[order]
        y_s = y[order].astype(np.float64)
        n = len(y_s)

        bin_width = (f_max - f_min) / n_bins
        thresholds = f_min + np.arange(1, n_bins) * bin_width
        ks = np.searchsorted(f_s, thresholds, side="right")

        valid = (ks > 0) & (ks < n)
        ks = ks[valid]
        if len(ks) == 0:
            return float("inf")

        prefix_y = np.concatenate([[0.0], np.cumsum(y_s)])
        prefix_y2 = np.concatenate([[0.0], np.cumsum(y_s ** 2)])

        s_l = prefix_y[ks]
        s2_l = prefix_y2[ks]
        s_r = prefix_y[n] - prefix_y[ks]
        s2_r = prefix_y2[n] - prefix_y2[ks]
        n_l = ks.astype(np.float64)
        n_r = (n - ks).astype(np.float64)

        if self.loss == "rmse":
            mse_l = s2_l - s_l ** 2 / n_l
            mse_r = s2_r - s_r ** 2 / n_r
            losses = np.sqrt(np.maximum(mse_l + mse_r, 0.0) / n)
        else:  # bce
            p_l = np.clip(s_l / n_l, 1e-15, 1 - 1e-15)
            p_r = np.clip(s_r / n_r, 1e-15, 1 - 1e-15)
            h_l = n_l * (-p_l * np.log2(p_l) - (1 - p_l) * np.log2(1 - p_l))
            h_r = n_r * (-p_r * np.log2(p_r) - (1 - p_r) * np.log2(1 - p_r))
            losses = (h_l + h_r) / n

        return float(losses.min())

    def get_loss(self, y_l, y_r):
        n1, n2 = len(y_l), len(y_r)
        if self.loss == "bce":
            lp = y_l.mean()
            lh = 0.0 if lp in (0, 1) else np.sum(-y_l * np.log2(lp) - (1 - y_l) * np.log2(1 - lp))
            rp = y_r.mean()
            rh = 0.0 if rp in (0, 1) else np.sum(-y_r * np.log2(rp) - (1 - y_r) * np.log2(1 - rp))
            return (lh + rh) / (n1 + n2)
        elif self.loss == "rmse":
            left_mse = ((y_l - y_l.mean()) ** 2).sum()
            right_mse = ((y_r - y_r.mean()) ** 2).sum()
            return np.sqrt((left_mse + right_mse) / (n1 + n2))

    @staticmethod
    def remove_outliers(f_1d, y, n_std=2.0):
        f_mean, f_std = f_1d.mean(), f_1d.std()
        mask = np.abs(f_1d - f_mean) <= n_std * f_std
        return f_1d[mask], y[mask]


class DualRFT:
    def __init__(self, n_bins=32, n_selected=1000):
        self.n_bins = n_bins
        self.n_selected = n_selected
        self.ft_train = FeatureTest(loss="rmse")
        self.ft_val = FeatureTest(loss="rmse")
        self.selected_features = None
        self.train_ranks = None
        self.val_ranks = None
        self.train_losses = None
        self.val_losses = None

    def fit(self, X_train, y_train, X_val, y_val):
        self.ft_train.fit(X_train, y_train, n_bins=self.n_bins)
        self.ft_val.fit(X_val, y_val, n_bins=self.n_bins)

        self.train_ranks = np.argsort([self.ft_train.dim_loss[i] for i in range(X_train.shape[1])])
        self.val_ranks = np.argsort([self.ft_val.dim_loss[i] for i in range(X_val.shape[1])])

        train_rank_map = np.empty_like(self.train_ranks)
        val_rank_map = np.empty_like(self.val_ranks)
        train_rank_map[self.train_ranks] = np.arange(X_train.shape[1])
        val_rank_map[self.val_ranks] = np.arange(X_val.shape[1])

        distances = train_rank_map ** 2 + val_rank_map ** 2
        self.selected_features = np.sort(np.argsort(distances)[: self.n_selected])

        self.train_losses = np.array([self.ft_train.dim_loss[i] for i in range(X_train.shape[1])])
        self.val_losses = np.array([self.ft_val.dim_loss[i] for i in range(X_val.shape[1])])

    def transform(self, X, device: str = "cpu"):
        assert self.selected_features is not None, "Call fit() first."
        if device.startswith("cuda"):
            import torch
            X_t   = torch.from_numpy(np.ascontiguousarray(X)).to(device)
            idx_t = torch.from_numpy(self.selected_features).to(device)
            out   = X_t[:, idx_t].contiguous().cpu().numpy()
            del X_t, idx_t
            return out
        return X[:, self.selected_features]

    def fit_transform(self, X_train, y_train, X_val, y_val):
        self.fit(X_train, y_train, X_val, y_val)
        return self.transform(X_train), self.transform(X_val)
