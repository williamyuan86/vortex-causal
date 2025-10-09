# src/constraint_fusion/fusion.py
import numpy as np

def apply_hard_constraints(W_init, hard_edges, node_count):
    """
    W_init: current weight matrix
    hard_edges: list of tuples (i, j, weight, score)
    Lock edges in W_init by setting large weight or fixed mask.
    Returns mask_fixed (bool matrix) and W_fixed (values).
    """
    mask_fixed = np.zeros_like(W_init, dtype=bool)
    W_fixed = np.array(W_init, copy=True)
    for (i,j,w,s) in hard_edges:
        mask_fixed[i,j] = True
        W_fixed[i,j] = w if w!=0 else 1e-3
    return mask_fixed, W_fixed

def apply_soft_priors_loss_term(W, priors_dict, lam=1.0):
    """
    Compute extra penalty for edges in priors_dict: { (i,j): p_confidence }
    In practice you'd augment NOTEARS loss; here we compute a penalty score for diagnostics.
    """
    penalty = 0.0
    for (i,j), p in priors_dict.items():
        # if model suggests opposite sign / absent, penalize proportionally
        penalty += lam * (1.0 - p) * abs(W[i,j])
    return penalty
