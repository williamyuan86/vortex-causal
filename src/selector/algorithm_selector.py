# src/selector/algorithm_selector.py
import numpy as np
from src.graph_generation.notears_al import get_weighted_adjacency
from sklearn.covariance import GraphicalLassoCV
import logging

logger = logging.getLogger(__name__)

def pc_like_partialcorr(df, alpha=0.01):
    """
    A very simple constraint-based proxy: estimate precision matrix with GraphicalLasso,
    translate non-zero to undirected edges, then orient heuristically by marginal correlations.
    Returns asymmetric weight matrix W_pc (k->j positive weight if suggest k->j)
    """
    X = df.values
    model = GraphicalLassoCV()
    model.fit(X)
    prec = model.precision_
    d = X.shape[1]
    W = np.zeros((d, d))
    # undirected adjacency proxy
    adj = (np.abs(prec) > 1e-6).astype(float)
    # orientation: if corr(k,j) > 0 then k->j weight = abs(corr)
    corr = np.corrcoef(X, rowvar=False)
    for i in range(d):
        for j in range(d):
            if i == j: continue
            if adj[i, j]:
                W[i, j] = abs(corr[i, j])  # simple heuristic
    return W

def consensus_scores(W_list):
    """
    W_list: list of adjacency weight matrices (same shape).
    Returns consensus matrix in [0,1] = normalized mean agreement.
    Also compute direction_consensus: edges with consistent sign/direction.
    """
    W_stack = np.stack(W_list, axis=0)  # (algos, d, d)
    # presence indicator
    presence = (np.abs(W_stack) > 1e-8).astype(float)
    presence_mean = presence.mean(axis=0)  # fraction of algos that propose edge
    # weight mean among nonzero
    weight_mean = np.where(presence.sum(axis=0)>0,
                           W_stack.sum(axis=0) / (presence.sum(axis=0)+1e-12),
                           0.0)
    return presence_mean, weight_mean

def run_selector(df, thresh_hard=0.9, thresh_ambiguous=0.4):
    """
    Run two/lightweight algorithms and produce:
      - hard_edges: list of (i,j) to be forced
      - ambiguous_edges: list to send to LLM
      - low_confidence: list to ignore
    """
    W1 = get_weighted_adjacency(df, lambda1=0.05)  # NOTEARS-like
    W2 = pc_like_partialcorr(df)  # constraint-style proxy
    presence_mean, weight_mean = consensus_scores([W1, W2])
    d = presence_mean.shape[0]
    hard = []
    ambiguous = []
    low = []
    for i in range(d):
        for j in range(d):
            if i==j: continue
            score = presence_mean[i, j]  # 0/0.5/1 in this two-algo case
            if score >= thresh_hard:
                hard.append((i, j, weight_mean[i, j], score))
            elif score <= thresh_ambiguous:
                low.append((i, j, weight_mean[i, j], score))
            else:
                ambiguous.append((i, j, weight_mean[i, j], score))
    return {'hard': hard, 'ambiguous': ambiguous, 'low': low, 'W1': W1, 'W2': W2}
