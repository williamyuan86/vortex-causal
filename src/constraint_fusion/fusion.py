# src/constraint_fusion/fusion_fixed.py
import numpy as np


def apply_hard_constraints(W_init, hard_edges, node_count=None):
    """
    Apply hard constraints to the weight matrix.

    Args:
        W_init: Initial weight matrix (numpy array)
        hard_edges: List of tuples (i, j, weight, score) representing forced edges
        node_count: Number of nodes (inferred from W_init if None)

    Returns:
        mask_fixed: Boolean mask indicating which edges are fixed
        W_fixed: Weight matrix with hard constraints applied
    """
    if node_count is None:
        node_count = W_init.shape[0]

    # Ensure W_init has correct shape
    if W_init.shape != (node_count, node_count):
        W_init = np.zeros((node_count, node_count))

    mask_fixed = np.zeros_like(W_init, dtype=bool)
    W_fixed = np.array(W_init, copy=True)

    # Fix diagonal to zero (no self-loops)
    np.fill_diagonal(mask_fixed, True)
    np.fill_diagonal(W_fixed, 0)

    # Apply hard edges
    for (i, j, w, s) in hard_edges:
        # Validate indices
        if 0 <= i < node_count and 0 <= j < node_count and i != j:
            mask_fixed[i, j] = True
            # Ensure non-zero weight for fixed edges
            W_fixed[i, j] = w if abs(w) > 1e-8 else 1e-3

    return mask_fixed, W_fixed


def apply_soft_priors_loss_term(W, priors_dict, lam=1.0):
    """
    Compute soft priors loss term for the given weight matrix.

    Args:
        W: Current weight matrix (numpy array)
        priors_dict: Dictionary mapping (i, j) -> prior confidence (0-1)
        lam: Regularization strength

    Returns:
        loss: Scalar loss value
    """
    if not priors_dict:
        return 0.0

    penalty = 0.0
    for (i, j), p in priors_dict.items():
        # Validate indices
        if 0 <= i < W.shape[0] and 0 <= j < W.shape[1]:
            # Loss encourages consistency with priors
            # If prior confidence is high, penalize deviation more
            deviation_loss = lam * (1.0 - p) * abs(W[i, j])
            penalty += deviation_loss

    return penalty


def validate_weight_matrix(W):
    """
    Validate weight matrix properties.

    Args:
        W: Weight matrix to validate

    Returns:
        is_valid: Boolean indicating if matrix is valid
        issues: List of identified issues
    """
    issues = []

    if not isinstance(W, np.ndarray):
        issues.append("Weight matrix must be numpy array")
        return False, issues

    if W.ndim != 2:
        issues.append("Weight matrix must be 2D")
        return False, issues

    if W.shape[0] != W.shape[1]:
        issues.append("Weight matrix must be square")
        return False, issues

    if np.any(np.isnan(W)):
        issues.append("Weight matrix contains NaN values")
        return False, issues

    if np.any(np.isinf(W)):
        issues.append("Weight matrix contains infinite values")
        return False, issues

    return len(issues) == 0, issues


def merge_constraints(hard_edges, soft_priors, conflict_resolution="hard_wins"):
    """
    Merge hard and soft constraints, resolving conflicts.

    Args:
        hard_edges: List of hard edge constraints
        soft_priors: Dictionary of soft priors
        conflict_resolution: Strategy for resolving conflicts

    Returns:
        merged_priors: Dictionary of merged priors
        conflicts: List of detected conflicts
    """
    conflicts = []
    merged_priors = dict(soft_priors)

    for (i, j, w, conf) in hard_edges:
        if (i, j) in soft_priors:
            conflicts.append(((i, j), "hard_vs_soft", w, soft_priors[(i, j)]))
            if conflict_resolution == "hard_wins":
                # Remove conflicting soft prior
                merged_priors.pop((i, j), None)

    return merged_priors, conflicts