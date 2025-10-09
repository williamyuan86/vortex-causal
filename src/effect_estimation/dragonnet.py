# src/effect_estimation/dragonnet.py
"""
Minimal DragonNet implementation (PyTorch) for ATE/CATE estimation.

Paper reference: "DragonNet: Estimating Heterogeneous Treatment Effects using Neural Networks"
This file contains:
- DragonNet model (shared representation + propensity head + outcome heads)
- training loop (binary treatment assumed)
- predict function to get potential outcomes, propensity scores
- metrics: ATE, RMSE on potential outcomes (if both potential outcomes known), PEHE

Notes:
- This is a simplified but functional implementation for v0.1.
- For production, consider using targeted regularization (TARNet-style targeting step),
  early stopping, cross-validation, and richer architectures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[200,100], activate=nn.ReLU):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activate())
            prev = h
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class DragonNet(nn.Module):
    def __init__(self, input_dim, repr_dims=[200,100], head_dims=[100]):
        """
        Shared representation -> propensity head (sigmoid) + two outcome heads (for y0,y1)
        """
        super().__init__()
        # shared representation
        self.repr = MLP(input_dim, hidden_dims=repr_dims)
        # propensity head
        self.propensity_head = nn.Sequential(
            nn.Linear(repr_dims[-1], head_dims[0]),
            nn.ReLU(),
            nn.Linear(head_dims[0], 1),
            nn.Sigmoid()
        )
        # outcome heads
        self.outcome_head0 = nn.Sequential(
            nn.Linear(repr_dims[-1]+1, head_dims[0]),  # concat treatment indicator
            nn.ReLU(),
            nn.Linear(head_dims[0], 1)
        )
        self.outcome_head1 = nn.Sequential(
            nn.Linear(repr_dims[-1]+1, head_dims[0]),
            nn.ReLU(),
            nn.Linear(head_dims[0], 1)
        )

    def forward(self, x):
        """
        x: [batch, features]
        returns: propensity p(t=1|x), mu0(x), mu1(x)
        """
        z = self.repr(x)  # [batch, repr_dim]
        p = self.propensity_head(z).squeeze(-1)  # [batch]
        # prepare for outcome heads: concatenated with treatment placeholder (0/1)
        t0 = torch.zeros((z.shape[0],1), device=z.device)
        t1 = torch.ones((z.shape[0],1), device=z.device)
        z0 = torch.cat([z, t0], dim=1)
        z1 = torch.cat([z, t1], dim=1)
        mu0 = self.outcome_head0(z0).squeeze(-1)
        mu1 = self.outcome_head1(z1).squeeze(-1)
        return p, mu0, mu1

# Loss components
bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()

def dragonnet_train(model, X, T, Y, epochs=200, batch_size=128, lr=1e-3,
                    alpha=1.0, beta=1.0, verbose=True):
    """
    Train DragonNet.
    - X: numpy array [n, d]
    - T: binary treatment vector [n] (0/1)
    - Y: outcome vector [n]
    - alpha: weight for propensity loss
    - beta: weight for outcome loss (both arms)
    Returns trained model.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    idx = np.arange(n)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)

    for ep in range(epochs):
        np.random.shuffle(idx)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            xb = X_t[batch_idx]
            tb = T_t[batch_idx]
            yb = Y_t[batch_idx]
            optimizer.zero_grad()
            p_pred, mu0_pred, mu1_pred = model(xb)  # p: [b], mu*: [b]
            # propensity loss
            prop_loss = bce_loss(p_pred, tb)
            # outcome loss: only observed outcomes
            mu_t = tb * mu1_pred + (1 - tb) * mu0_pred
            outcome_loss = mse_loss(mu_t, yb)
            loss = alpha * prop_loss + beta * outcome_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)
        if verbose and (ep % 20 == 0 or ep==epochs-1):
            avg = epoch_loss / n
            print(f"[DragonNet] Epoch {ep} loss {avg:.6f}")
    return model

def predict_dragonnet(model, X):
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        p, mu0, mu1 = model(X_t)
        return p.cpu().numpy(), mu0.cpu().numpy(), mu1.cpu().numpy()

# Evaluation metrics
def ate_from_preds(mu0, mu1):
    return np.mean(mu1 - mu0)

def rmse_potential(y_true0, y_true1, mu0, mu1):
    """
    If both potential outcomes available (e.g., in synthetic), compute RMSE.
    y_true0,y_true1 can be None if unknown -> skip.
    """
    res = {}
    if y_true0 is not None:
        res['rmse0'] = np.sqrt(np.mean((y_true0 - mu0)**2))
    if y_true1 is not None:
        res['rmse1'] = np.sqrt(np.mean((y_true1 - mu1)**2))
    return res

def pehe(mu0, mu1, y0_true=None, y1_true=None):
    """
    PEHE = sqrt( mean ( (mu1-mu0) - (y1 - y0) )^2 )
    Requires both potential outcomes.
    """
    if y0_true is None or y1_true is None:
        return None
    tau_pred = mu1 - mu0
    tau_true = y1_true - y0_true
    return np.sqrt(np.mean((tau_pred - tau_true)**2))

# Utility wrapper: train + eval on dataset dict
def fit_and_eval(X, T, Y, X_val=None, T_val=None, Y_val=None, **kwargs):
    input_dim = X.shape[1]
    model = DragonNet(input_dim)
    dragonnet_train(model, X, T, Y, **kwargs)
    p, mu0, mu1 = predict_dragonnet(model, X if X_val is None else X_val)
    ate = ate_from_preds(mu0, mu1)
    metrics = {'ate': ate, 'p_mean': float(np.mean(p))}
    # If val truth available and synthetic potential outcomes known, compute PEHE/RMSE
    if X_val is not None and T_val is not None and Y_val is not None:
        # evaluate on validation set
        p_val, mu0_val, mu1_val = predict_dragonnet(model, X_val)
        ate_val = ate_from_preds(mu0_val, mu1_val)
        metrics['ate_val'] = ate_val
    return model, metrics
