import logging

import numpy as np
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression


def fit_lnt(x, y, depth, num_tree, device="cuda"):
    """
    Least-squares Normal Transform. Returns W (n_features, m); new features = x @ W.

    Columns: one linear regressor per distinct feature subset used by a shallow XGBoost tree,
    plus the SVD direction of the joint linear map to (1-y, y).
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    model = xgb.XGBRegressor(objective="reg:squarederror", max_depth=depth, n_estimators=num_tree,
                             learning_rate=0.1, tree_method="hist", device=device)
    model.fit(x, y)
    trees = model.get_booster().trees_to_dataframe()
    splits = trees[trees["Feature"] != "Leaf"]
    subsets = {tuple(sorted(int(f[1:]) for f in s.unique())) for _, s in splits.groupby("Tree")["Feature"]}
    subsets = [list(s) for s in subsets if s]
    logging.info(f"[LNT] {len(subsets)} distinct tree feature subsets")

    def _fit_one(sel):
        theta = np.zeros(x.shape[1], dtype=np.float32)
        theta[sel] = LinearRegression().fit(x[:, sel], y).coef_
        return theta

    cols = Parallel(n_jobs=-1, prefer="threads")(delayed(_fit_one)(s) for s in subsets)

    x0 = x - x.mean(0)
    y2 = np.stack([1 - y, y], 1)
    y0 = y2 - y2.mean(0)
    # Normal equations (multi-threaded BLAS); lstsq on the small p×p system tolerates duplicate columns
    T = np.linalg.lstsq(x0.T @ x0, x0.T @ y0, rcond=None)[0]
    cols.append(TruncatedSVD(n_components=1, random_state=42).fit_transform(T)[:, 0])
    return np.column_stack(cols).astype(np.float32)
