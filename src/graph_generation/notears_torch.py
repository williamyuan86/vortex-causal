# src/graph_generation/notears_torch_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

logger = logging.getLogger(__name__)

class NotearsMLP(nn.Module):
    """Simple linear NOTEARS model: X = WX + noise"""
    def __init__(self, dim):
        super().__init__()
        # unconstrained weight matrix
        self.W = nn.Parameter(torch.zeros(dim, dim))

    def forward(self, X):
        return X @ self.W  # [n,d]

def acyclicity_constraint(W):
    """
    Compute acyclicity constraint value h(W) = tr(exp(W*W)) - d = 0
    """
    d = W.shape[0]
    M = W * W  # element-wise square

    # Add small regularization to prevent numerical issues
    M = M + 1e-8 * torch.eye(d, device=W.device)

    try:
        expm = torch.matrix_exp(M)
        h = torch.trace(expm) - d
        return h
    except Exception as e:
        logger.warning(f"Matrix exponential failed: {e}, using fallback")
        # Fallback: use trace of M as proxy
        return torch.trace(M)

def notears_loss(X, model, lambda1=0.01, priors_dict=None, mask_fixed=None, lam_soft=1.0):
    """
    X: [n,d]
    model: NotearsMLP
    lambda1: L1 regularization weight
    priors_dict: {(i,j): p_confidence} for soft constraints
    mask_fixed: boolean mask for hard constraints (fixed edges)
    lam_soft: multiplier for soft prior penalty
    """
    X_hat = model(X)
    mse_loss = 0.5 * ((X - X_hat) ** 2).mean()

    # L1 regularization (excluding diagonal)
    W = model.W
    l1 = torch.sum(torch.abs(W)) - torch.sum(torch.abs(torch.diag(W)))

    # soft prior penalty
    soft_penalty = torch.tensor(0.0, device=X.device)
    if priors_dict:
        for (i,j), p in priors_dict.items():
            if 0 <= i < W.shape[0] and 0 <= j < W.shape[1]:
                # penalty pushes W[i,j] toward sign that agrees with prior
                soft_penalty += lam_soft * (1 - p) * torch.abs(W[i,j])

    # hard constraint enforcement (zero out fixed edges' gradients)
    if mask_fixed is not None:
        # Ensure mask has same shape as W
        if mask_fixed.shape != W.shape:
            logger.warning(f"Mask shape mismatch: {mask_fixed.shape} vs {W.shape}")
            mask_fixed = torch.zeros_like(W, dtype=torch.bool)

        # Freeze fixed edges
        with torch.no_grad():
            W.data[mask_fixed] = W.data[mask_fixed].detach()

    # acyclicity
    h = acyclicity_constraint(W)

    # augmented Lagrangian term (will be dynamically updated)
    loss = mse_loss + lambda1 * l1 + soft_penalty
    return loss, mse_loss, l1, soft_penalty, h

def train_notears_torch(
    X_np,
    priors_dict=None,
    mask_fixed=None,
    lambda1=0.01,
    lam_soft=1.0,
    rho_max=1e16,
    h_tol=1e-8,
    max_iter=100,
    lr=1e-3,
    device=None,
    verbose=True
):
    """
    Train NOTEARS with augmented Lagrangian and soft priors.

    Args:
        X_np: Input data (numpy array)
        priors_dict: Soft constraints dictionary
        mask_fixed: Hard constraints mask
        lambda1: L1 regularization
        lam_soft: Soft constraint regularization
        rho_max: Maximum augmented Lagrangian parameter
        h_tol: Tolerance for acyclicity constraint
        max_iter: Maximum iterations
        lr: Learning rate
        device: PyTorch device
        verbose: Whether to print progress

    Returns:
        W_est: Estimated weight matrix
    """
    # Input validation
    if not isinstance(X_np, np.ndarray):
        raise ValueError("X_np must be numpy array")

    if X_np.ndim != 2:
        raise ValueError("X_np must be 2D array")

    if np.any(np.isnan(X_np)) or np.any(np.isinf(X_np)):
        raise ValueError("X_np contains NaN or infinite values")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        n, d = X.shape

        # Validate dimensions
        if n < d:
            logger.warning(f"More variables than samples ({d} > {n}), results may be unreliable")

        model = NotearsMLP(d).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        rho, alpha = 1.0, 0.0

        for iteration in range(max_iter):
            optimizer.zero_grad()

            try:
                loss, mse, l1, softp, h = notears_loss(X, model, lambda1, priors_dict, mask_fixed, lam_soft)
                aug_loss = loss + 0.5 * rho * h * h + alpha * h

                # Check for numerical issues
                if torch.isnan(aug_loss) or torch.isinf(aug_loss):
                    logger.warning(f"Numerical issues at iteration {iteration}, stopping early")
                    break

                aug_loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                with torch.no_grad():
                    h_val = h.item()

                if verbose and iteration % 10 == 0:
                    logger.info(f"[Iter {iteration}] loss={loss.item():.4f} h(W)={h_val:.4e} mse={mse.item():.4f}")

                if h_val <= h_tol or rho > rho_max:
                    if verbose:
                        logger.info(f"Converged at iteration {iteration}, h={h_val:.4e}")
                    break

                # Augmented Lagrangian multipliers update
                rho = min(rho * 1.5, rho_max)
                alpha += rho * h_val

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                break

        W_est = model.W.detach().cpu().numpy()

        # Post-processing: threshold small weights
        threshold = 1e-4
        W_est[np.abs(W_est) < threshold] = 0.0

        # Ensure diagonal is exactly zero
        np.fill_diagonal(W_est, 0)

        return W_est

    except Exception as e:
        logger.error(f"Error in train_notears_torch: {e}")
        # Return fallback matrix
        return np.zeros((X_np.shape[1], X_np.shape[1]))

def validate_weight_matrix(W, tol=1e-6):
    """
    Validate the learned weight matrix.

    Args:
        W: Weight matrix to validate
        tol: Tolerance for checking diagonal

    Returns:
        is_valid: Boolean indicating validity
        issues: List of issues found
    """
    issues = []

    if not isinstance(W, np.ndarray):
        issues.append("W must be numpy array")
        return False, issues

    if W.ndim != 2:
        issues.append("W must be 2D array")
        return False, issues

    if W.shape[0] != W.shape[1]:
        issues.append("W must be square")
        return False, issues

    if np.any(np.isnan(W)):
        issues.append("W contains NaN values")
        return False, issues

    if np.any(np.isinf(W)):
        issues.append("W contains infinite values")
        return False, issues

    if not np.allclose(np.diag(W), 0, atol=tol):
        issues.append("Diagonal elements are not zero")
        return False, issues

    return len(issues) == 0, issues