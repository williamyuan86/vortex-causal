# src/graph_generation/notears_al.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

def notears_al_placeholder(X, lambda1=0.1, max_iter=100, tol=1e-8):
    """
    Minimal placeholder NOTEARS-like routine for v0.1.
    This is NOT a full NOTEARS implementation. It constructs a simple
    weighted adjacency by Lasso regressions (neighborhood selection).
    Replace with a real NOTEARS-AL implementation for production.
    """
    n, d = X.shape
    W = np.zeros((d, d))
    from sklearn.linear_model import Lasso
    for j in range(d):
        # predict column j from others
        idx = [k for k in range(d) if k != j]
        Xj = X[:, idx]
        y = X[:, j]
        lasso = Lasso(alpha=lambda1, max_iter=1000)
        lasso.fit(Xj, y)
        coef = lasso.coef_
        # insert into W: coef_k corresponds to edge k -> j
        for ii, k in enumerate(idx):
            W[k, j] = coef[ii]
    # keep absolute weights
    return W

def get_weighted_adjacency(df, **kwargs):
    X = df.values
    W = notears_al_placeholder(X, **kwargs)
    # threshold small weights
    W[np.abs(W) < 1e-4] = 0.0
    return W
