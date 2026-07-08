import numpy as np
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt


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
    def __init__(self, n_bins=32, n_selected=1000, feature_masks=None):
        self.n_bins = n_bins
        self.n_selected = n_selected
        self.ft_train = FeatureTest(loss="rmse")
        self.ft_val = FeatureTest(loss="rmse")
        self.selected_features = None
        self.train_ranks = None
        self.val_ranks = None
        self.train_losses = None
        self.val_losses = None
        self.feature_masks = feature_masks or {}

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

    def plot(self, save_root):
        os.makedirs(save_root, exist_ok=True)
        assert self.train_losses is not None

        num_features = len(self.train_losses)
        train_ranks = np.argsort(self.train_losses)
        val_ranks = np.argsort(self.val_losses)

        for name, ranks, losses, color in [
            ("train", train_ranks, self.train_losses, "blue"),
            ("val", val_ranks, self.val_losses, "blue"),
        ]:
            plt.figure(figsize=(8, 5))
            plt.scatter(np.arange(num_features), losses[ranks], color=color, s=15)
            plt.title(f"{'Train' if name == 'train' else 'Validation'} RMSE vs Feature Rank")
            plt.xlabel("Feature Rank")
            plt.ylabel("RMSE")
            plt.grid(True)
            plt.tight_layout()
            path = os.path.join(save_root, f"{name}_rmse_rank_split.png")
            plt.savefig(path)
            plt.close()

        train_rank_map = np.empty_like(train_ranks)
        val_rank_map = np.empty_like(val_ranks)
        train_rank_map[train_ranks] = np.arange(num_features)
        val_rank_map[val_ranks] = np.arange(num_features)

        plt.figure(figsize=(6, 6))
        plt.scatter(train_rank_map, val_rank_map, s=20, alpha=0.25, label="All")

        color_map = {"saab_img": "blue", "saab_laws": "purple", "raw": "green", "grad": "red"}
        if self.feature_masks:
            for name, mask in self.feature_masks.items():
                if name not in color_map:
                    continue
                idx = np.where(np.asarray(mask, dtype=bool))[0]
                if idx.size == 0:
                    continue
                plt.scatter(train_rank_map[idx], val_rank_map[idx], s=30,
                            label=name, color=color_map[name], alpha=0.8)
        else:
            plt.scatter(train_rank_map[self.selected_features], val_rank_map[self.selected_features],
                        color="red", s=40, label="Selected")

        plt.xlabel("Train Rank")
        plt.ylabel("Validation Rank")
        plt.title("Train vs Validation Feature Rank")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_root, "train_vs_val_rank.png"))
        plt.close()


def build_feature_masks(featgen, num_feat_total: int):
    saab_dim = int(getattr(featgen, "saab_dim", 0) or 0)
    raw_dim = int(getattr(featgen, "raw_dim", 0) or 0)
    grad_dim = int(getattr(featgen, "grad_dim", 0) or 0)
    feat_dim = int(getattr(featgen, "feature_dim", 0) or 0)

    if feat_dim <= 0:
        raise ValueError("FeatGen must be fitted before building feature masks.")
    if feat_dim != num_feat_total:
        raise ValueError(f"FeatGen feature_dim ({feat_dim}) != num_feat_total ({num_feat_total})")
    if saab_dim + raw_dim + grad_dim != feat_dim:
        raise ValueError(f"Inconsistent FeatGen dims: saab={saab_dim}, raw={raw_dim}, grad={grad_dim}")

    masks = {
        "saab_img": np.zeros(num_feat_total, dtype=bool),
        "saab_laws": np.zeros(num_feat_total, dtype=bool),
        "raw": np.zeros(num_feat_total, dtype=bool),
        "grad": np.zeros(num_feat_total, dtype=bool),
    }
    off = 0
    masks["saab_img"][off:off + saab_dim] = True; off += saab_dim
    masks["raw"][off:off + raw_dim] = True; off += raw_dim
    if grad_dim > 0:
        masks["grad"][off:off + grad_dim] = True; off += grad_dim
    if off != num_feat_total:
        raise ValueError(f"Final offset {off} != total {num_feat_total}")
    return masks


def build_feature_masks_from_featgens(fg_img, fg_res, num_feat_total):
    masks = {
        "saab": np.zeros(num_feat_total, dtype=bool),
        "raw": np.zeros(num_feat_total, dtype=bool),
        "grad": np.zeros(num_feat_total, dtype=bool),
    }
    offset = 0
    for fg in [fg_img, fg_res]:
        if fg is None:
            continue
        saab_dim = int(fg.saab_dim)
        raw_dim = int(fg.raw_dim)
        grad_dim = int(fg.grad_dim)
        masks["saab"][offset:offset + saab_dim] = True; offset += saab_dim
        masks["raw"][offset:offset + raw_dim] = True; offset += raw_dim
        if grad_dim > 0:
            masks["grad"][offset:offset + grad_dim] = True; offset += grad_dim
    if offset != num_feat_total:
        raise ValueError(f"Feature mask mismatch: offset={offset}, total={num_feat_total}")
    return masks
