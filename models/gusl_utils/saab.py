import numpy as np
import numba
import torch


@numba.jit(nopython=True, parallel=True)
def pca_cal(X: np.ndarray):
    cov = X.transpose() @ X
    eva, eve = np.linalg.eigh(cov)
    inds = eva.argsort()[::-1]
    eva = eva[inds]
    kernels = eve.transpose()[inds]
    return kernels, eva / (X.shape[0] - 1)


@numba.jit(forceobj=True, parallel=True)
def remove_mean(X: np.ndarray, feature_mean: np.ndarray):
    return X - feature_mean


@numba.jit(nopython=True, parallel=False)
def feat_transform(X: np.ndarray, kernel: np.ndarray):
    return X @ kernel.transpose()


class Saab:
    def __init__(self, num_kernels=-1, needBias=True, bias=0):
        self.num_kernels = num_kernels
        self.needBias = needBias
        self.Bias_previous = bias
        self.Bias_current = []
        self.Kernels = []
        self.Mean0 = []
        self.Energy = []
        self.trained = False

    def fit(self, X):
        assert len(X.shape) == 2, "Input must be a 2D array!"
        X = X.astype("float32")

        if self.needBias:
            X += self.Bias_previous

        dc = np.mean(X, axis=1, keepdims=True)
        X = remove_mean(X, dc)

        self.Bias_current = np.max(np.linalg.norm(X, axis=1))

        self.Mean0 = np.mean(X, axis=0, keepdims=True)
        X = remove_mean(X, self.Mean0)

        if self.num_kernels == -1:
            self.num_kernels = X.shape[-1]

        kernels, eva = pca_cal(X)

        dc_kernel = 1 / np.sqrt(X.shape[-1]) * np.ones((1, X.shape[-1]))
        kernels = np.concatenate((dc_kernel, kernels[:-1]), axis=0)

        largest_ev = np.var(dc * np.sqrt(X.shape[-1]))
        energy = np.concatenate((np.array([largest_ev]), eva[:-1]), axis=0)
        energy = energy / np.sum(energy)

        self.Kernels, self.Energy = kernels.astype("float32"), energy
        self.trained = True

    def transform(self, X):
        assert self.trained, "Must call fit first!"
        X = X.astype("float32")

        if self.needBias:
            X += self.Bias_previous

        X = remove_mean(X, self.Mean0)
        X = feat_transform(X, self.Kernels)
        return X


class SaabTorch:
    """GPU-accelerated Saab via PyTorch. Drop-in replacement for Saab."""

    def __init__(self, num_kernels=-1, needBias=True, bias=0.0, device=None):
        self.num_kernels = int(num_kernels)
        self.needBias = bool(needBias)
        self.Bias_previous = float(bias)
        self.Bias_current = None
        self.Kernels = None
        self.Mean0 = None
        self.Energy = None
        self.trained = False

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    @staticmethod
    def _to_np(x_t: torch.Tensor) -> np.ndarray:
        return x_t.detach().cpu().numpy()

    def fit(self, X: np.ndarray):
        assert len(X.shape) == 2, "Input must be a 2D array!"
        X_np = np.asarray(X, dtype=np.float32, order="C")
        M, D = X_np.shape

        X_t = torch.from_numpy(X_np).to(self.device)

        if self.needBias:
            X_t = X_t + self.Bias_previous

        dc = torch.mean(X_t, dim=1, keepdim=True)
        X_ac = X_t - dc

        self.Bias_current = float(torch.max(torch.linalg.norm(X_ac, dim=1)).item())

        mean0 = torch.mean(X_ac, dim=0, keepdim=True)
        X0 = X_ac - mean0

        if self.num_kernels == -1:
            self.num_kernels = D

        cov = X0.transpose(0, 1) @ X0

        eva, eve = torch.linalg.eigh(cov)

        inds = torch.argsort(eva, descending=True)
        eva = eva[inds]
        eve = eve[:, inds]

        kernels = eve.transpose(0, 1)
        eva = eva / float(M - 1)

        dc_kernel = (1.0 / np.sqrt(D)) * torch.ones((1, D), device=self.device, dtype=torch.float32)
        kernels_all = torch.cat([dc_kernel, kernels[:-1, :]], dim=0)

        dc_scaled = dc * np.sqrt(D)
        largest_ev = torch.var(dc_scaled, dim=0, unbiased=False).squeeze(0)

        energy = torch.cat([largest_ev.view(1), eva[:-1]], dim=0)
        energy = energy / torch.sum(energy)

        out_dim = max(1, min(int(self.num_kernels), kernels_all.shape[0]))
        kernels_all = kernels_all[:out_dim, :]
        energy = energy[:out_dim]

        self.Kernels = self._to_np(kernels_all).astype(np.float32, copy=False)
        self.Energy = self._to_np(energy).astype(np.float32, copy=False)
        self.Mean0 = self._to_np(mean0).astype(np.float32, copy=False)

        self.trained = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.trained, "Must call fit first!"
        X_np = np.asarray(X, dtype=np.float32, order="C")
        X_t = torch.from_numpy(X_np).to(self.device)

        if self.needBias:
            X_t = X_t + self.Bias_previous

        mean0_t = torch.from_numpy(self.Mean0).to(self.device)
        X_t = X_t - mean0_t

        K_t = torch.from_numpy(self.Kernels).to(self.device)
        Y = X_t @ K_t.transpose(0, 1)

        return self._to_np(Y).astype(np.float32, copy=False)
