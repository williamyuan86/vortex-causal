# src/graph_generation/notears_torch.py
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
    expm = torch.matrix_exp(M)
    return torch.trace(expm) - d

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

    # L1 regularization
    l1 = torch.sum(torch.abs(model.W))

    # soft prior penalty
    soft_penalty = torch.tensor(0.0, device=X.device)
    if priors_dict:
        for (i,j), p in priors_dict.items():
            # penalty pushes W[i,j] toward sign that agrees with prior
            soft_penalty += lam_soft * (1 - p) * torch.abs(model.W[i,j])

    # hard constraint enforcement (zero out fixed edges’ gradients)
    if mask_fixed is not None:
        model.W.data[mask_fixed] = model.W.data[mask_fixed].detach()  # freeze fixed edges

    # acyclicity
    h = acyclicity_constraint(model.W)
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
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train NOTEARS with augmented Lagrangian and soft priors.
    """
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    n, d = X.shape
    model = NotearsMLP(d).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    rho, alpha = 1.0, 0.0

    for iteration in range(max_iter):
        optimizer.zero_grad()
        loss, mse, l1, softp, h = notears_loss(X, model, lambda1, priors_dict, mask_fixed, lam_soft)
        aug_loss = loss + 0.5 * rho * h * h + alpha * h
        aug_loss.backward()
        optimizer.step()

        with torch.no_grad():
            h_val = h.item()
        if iteration % 10 == 0:
            logger.info(f"[Iter {iteration}] loss={loss.item():.4f} h(W)={h_val:.4e} mse={mse.item():.4f}")
        if h_val <= h_tol or rho > rho_max:
            break
        # Augmented Lagrangian multipliers update
        rho *= 1.5
        alpha += rho * h_val

    W_est = model.W.detach().cpu().numpy()
    W_est[np.abs(W_est) < 1e-4] = 0.0
    return W_est
