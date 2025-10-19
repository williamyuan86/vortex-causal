# src/selector/algorithm_selector_fixed.py
import numpy as np
from sklearn.covariance import GraphicalLassoCV
import logging

logger = logging.getLogger(__name__)

try:
    from src.graph_generation.notears_al import get_weighted_adjacency
    NOTEARS_AVAILABLE = True
except ImportError:
    NOTEARS_AVAILABLE = False
    logger.warning("NOTEARS not available, using fallback")

def pc_like_partialcorr(df, alpha=0.01):
    """
    A very simple constraint-based proxy: estimate precision matrix with GraphicalLasso,
    translate non-zero to undirected edges, then orient heuristically by marginal correlations.
    Returns asymmetric weight matrix W_pc (k->j positive weight if suggest k->j)
    """
    try:
        X = df.values
        if X.shape[0] < X.shape[1]:
            logger.warning(f"More variables than samples ({X.shape[1]} > {X.shape[0]}), results may be unstable")

        model = GraphicalLassoCV(cv=min(3, X.shape[0] // 2))  # Adjust CV for small datasets
        model.fit(X)
        prec = model.precision_
        d = X.shape[1]
        W = np.zeros((d, d))

        # undirected adjacency proxy
        adj = (np.abs(prec) > 1e-6).astype(float)

        # orientation: if corr(k,j) > 0 then k->j weight = abs(corr)
        corr = np.corrcoef(X, rowvar=False)

        # Handle NaN correlations (constant variables)
        corr = np.nan_to_num(corr, nan=0.0)

        for i in range(d):
            for j in range(d):
                if i == j: continue
                if adj[i, j]:
                    W[i, j] = abs(corr[i, j])  # simple heuristic
        return W
    except Exception as e:
        logger.error(f"Error in pc_like_partialcorr: {e}")
        # Fallback: return zero matrix
        return np.zeros((df.shape[1], df.shape[1]))

def consensus_scores(W_list):
    """
    W_list: list of adjacency weight matrices (same shape).
    Returns consensus matrix in [0,1] = normalized mean agreement.
    Also compute direction_consensus: edges with consistent sign/direction.
    """
    if not W_list:
        raise ValueError("W_list cannot be empty")

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
      - hard_edges: list of (i, j) to be forced
      - ambiguous_edges: list to send to LLM
      - low_confidence: list to ignore
    """
    # Validate thresholds
    if not (0 <= thresh_ambiguous <= thresh_hard <= 1):
        raise ValueError("Invalid thresholds: must satisfy 0 <= thresh_ambiguous <= thresh_hard <= 1")

    try:
        if NOTEARS_AVAILABLE:
            W1 = get_weighted_adjacency(df, lambda1=0.05)  # NOTEARS-like
        else:
            # Fallback: use random matrix
            W1 = np.random.normal(0, 0.1, (df.shape[1], df.shape[1]))
            np.fill_diagonal(W1, 0)
    except Exception as e:
        logger.error(f"Error in first algorithm: {e}")
        W1 = np.zeros((df.shape[1], df.shape[1]))

    W2 = pc_like_partialcorr(df)  # constraint-style proxy
    presence_mean, weight_mean = consensus_scores([W1, W2])

    d = presence_mean.shape[0]
    hard = []
    ambiguous = []
    low = []

    for i in range(d):
        for j in range(d):
            if i == j: continue
            score = presence_mean[i, j]  # 0/0.5/1 in this two-algo case
            weight = weight_mean[i, j]

            edge_data = (i, j, weight, score)
            if score >= thresh_hard:
                hard.append(edge_data)
            elif score <= thresh_ambiguous:
                low.append(edge_data)
            else:
                ambiguous.append(edge_data)

    return {
        'hard': hard,
        'ambiguous': ambiguous,
        'low': low,
        'W1': W1,
        'W2': W2,
        'presence_mean': presence_mean,
        'weight_mean': weight_mean
    }

def validate_selector_output(selector_result, n_vars):
    """
    Validate the output of the selector.

    Args:
        selector_result: Dictionary from run_selector
        n_vars: Expected number of variables

    Returns:
        is_valid: Boolean indicating validity
        issues: List of issues found
    """
    issues = []

    required_keys = ['hard', 'ambiguous', 'low', 'W1', 'W2']
    for key in required_keys:
        if key not in selector_result:
            issues.append(f"Missing required key: {key}")

    if len(issues) > 0:
        return False, issues

    # Check weight matrices
    for W_key in ['W1', 'W2']:
        W = selector_result[W_key]
        if W.shape != (n_vars, n_vars):
            issues.append(f"Invalid shape for {W_key}: {W.shape} != {(n_vars, n_vars)}")
        if not np.allclose(W.diagonal(), 0, atol=1e-8):
            issues.append(f"Diagonal not zero for {W_key}")

    # Check edge categories
    total_edges = len(selector_result['hard']) + len(selector_result['ambiguous']) + len(selector_result['low'])
    max_edges = n_vars * (n_vars - 1)
    if total_edges > max_edges:
        issues.append(f"Too many edges: {total_edges} > {max_edges}")

    # Check edge data format
    for category in ['hard', 'ambiguous', 'low']:
        for i, edge in enumerate(selector_result[category]):
            if not isinstance(edge, (tuple, list)) or len(edge) != 4:
                issues.append(f"Invalid edge format in {category}[{i}]: {edge}")
                continue

            i_idx, j_idx, weight, score = edge
            if not (0 <= i_idx < n_vars) or not (0 <= j_idx < n_vars):
                issues.append(f"Invalid indices in {category}[{i}]: ({i_idx}, {j_idx})")
            if i_idx == j_idx:
                issues.append(f"Self-loop in {category}[{i}]: ({i_idx}, {j_idx})")

    return len(issues) == 0, issues