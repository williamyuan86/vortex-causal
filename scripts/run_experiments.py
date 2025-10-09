# scripts/run_experiments.py
import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from src.utils.data_loader import load_sachs
from src.selector.algorithm_selector import run_selector
from src.graph_generation.notears_torch import train_notears_torch
from src.ensemble.ensemble_cc_agent_openai import ensemble_decide
from src.constraint_fusion.fusion import apply_hard_constraints
from src.effect_estimation.dragonnet import DragonNet, dragonnet_train, predict_dragonnet, ate_from_preds, pehe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("exp")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def eval_against_groundtruth(W_est, W_true, threshold=0.1):
    W_est_bin = (np.abs(W_est) > threshold).astype(int)
    W_true_bin = (np.abs(W_true) > 1e-8).astype(int)
    TP = np.sum(W_est_bin * W_true_bin)
    FP = np.sum(W_est_bin * (1 - W_true_bin))
    FN = np.sum((1 - W_est_bin) * W_true_bin)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    shd = int(np.sum(np.abs(W_est_bin - W_true_bin)))
    return {"SHD": shd, "F1": float(f1)}

def make_synthetic_pehe_dataset(n=2000, d=10, seed=42):
    """
    Create a simple synthetic dataset with known potential outcomes.
    Data generating process:
      X ~ N(0,I)
      T ~ Bernoulli(sigmoid(Xw_t))
      Y0 = X @ beta0 + noise
      Y1 = Y0 + tau(X)  (treatment effect tau(X) is nonlinear)
    Returns: X, T, Y_obs, Y0_true, Y1_true
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    w_t = rng.normal(size=(d,))
    logits = X @ w_t
    p = 1 / (1 + np.exp(-logits))
    T = rng.binomial(1, p)
    beta0 = rng.normal(scale=0.5, size=(d,))
    noise = rng.normal(scale=0.5, size=(n,))
    Y0 = X @ beta0 + noise
    # define heterogeneous treatment effect
    tau = 1.0 + 0.5 * np.tanh(X[:, 0])  # simple non-linear effect
    Y1 = Y0 + tau
    Y = T * Y1 + (1 - T) * Y0
    return X.astype(np.float32), T.astype(np.int64), Y.astype(np.float32), Y0.astype(np.float32), Y1.astype(np.float32)

def run_mode(df, mode="baseline", model_list=None, use_synthetic=False):
    logger.info(f"Running mode: {mode} (synthetic={use_synthetic})")
    # Step 1: selector
    sel = run_selector(df)
    mask_fixed, _ = apply_hard_constraints(np.zeros((df.shape[1], df.shape[1])), sel['hard'], df.shape[1])

    priors = None
    if mode == "soft":
        # If real LLMs used, call them for ambiguous edges; here we mock priors=0.8
        priors = {(i,j): 0.8 for (i,j,_,_) in sel['ambiguous']}

    # Step 2: run NOTEARS (torch)
    W = train_notears_torch(df.values, priors_dict=priors if mode=="soft" else None,
                            mask_fixed=mask_fixed if mode!="baseline" else None,
                            lambda1=0.02, lam_soft=1.0, max_iter=200, lr=1e-3)

    # Step 3: Evaluate structure if groundtruth provided (not provided for Sachs here)
    # For demo/PEHE we optionally use synthetic dataset
    metrics = {}
    llm_calls = len(sel['ambiguous']) if mode!="baseline" else 0

    if use_synthetic:
        X, T, Y, Y0_true, Y1_true = make_synthetic_pehe_dataset(n=2000, d=df.shape[1], seed=123)
        # train DragonNet on synthetic (X,T,Y)
        model = DragonNet(X.shape[1])
        dragonnet_train(model, X, T, Y, epochs=100, batch_size=256, lr=1e-3, verbose=False)
        p_pred, mu0_pred, mu1_pred = predict_dragonnet(model, X)
        ate_est = ate_from_preds(mu0_pred, mu1_pred)
        pehe_val = pehe(mu0_pred, mu1_pred, Y0_true, Y1_true)
        metrics.update({"ate": float(ate_est), "pehe": float(pehe_val), "llm_calls": llm_calls})
        # save preds for inspection
        np.save(os.path.join(OUTPUT_DIR, f"mu0_{mode}.npy"), mu0_pred)
        np.save(os.path.join(OUTPUT_DIR, f"mu1_{mode}.npy"), mu1_pred)
    else:
        # No groundtruth potential outcomes: just run DragonNet on df if possible
        # We attempt to identify a treatment and outcome automatically (fallback)
        cols = df.columns.tolist()
        if len(cols) >= 3:
            treatment_idx = 0
            outcome_idx = 1
            treatment_col = cols[treatment_idx]
            outcome_col = cols[outcome_idx]
            # binarize treatment by median as default
            df_mod = df.copy()
            df_mod[treatment_col] = (df_mod[treatment_col] > df_mod[treatment_col].median()).astype(int)
            covs = [c for c in cols if c not in [treatment_col, outcome_col]]
            X = df_mod[covs].values.astype(np.float32)
            T = df_mod[treatment_col].values.astype(np.int64)
            Y = df_mod[outcome_col].values.astype(np.float32)
            model = DragonNet(X.shape[1])
            dragonnet_train(model, X, T, Y, epochs=100, batch_size=128, lr=1e-3, verbose=False)
            p_pred, mu0_pred, mu1_pred = predict_dragonnet(model, X)
            ate_est = ate_from_preds(mu0_pred, mu1_pred)
            metrics.update({"ate": float(ate_est), "pehe": None, "llm_calls": llm_calls})
            np.save(os.path.join(OUTPUT_DIR, f"mu0_{mode}.npy"), mu0_pred)
            np.save(os.path.join(OUTPUT_DIR, f"mu1_{mode}.npy"), mu1_pred)
        else:
            metrics.update({"ate": None, "pehe": None, "llm_calls": llm_calls})
    return W, metrics

def main():
    df = load_sachs("data/sachs.csv")
    # run three modes; for demo we run synthetic=True only for one mode to demonstrate PEHE
    all_results = []
    for idx, mode in enumerate(["baseline", "hard", "soft"]):
        use_synth = (mode == "soft")  # demo: compute PEHE on soft mode
        W_est, metrics = run_mode(df, mode, use_synthetic=use_synth)
        # placeholder W_true (if you have true adjacency, load and compare)
        W_true = np.zeros_like(W_est)  # replace with actual groundtruth adjacency if available
        struct_metrics = {"SHD": None, "F1": None}
        # join metrics
        res = {"mode": mode, "llm_calls": metrics.get("llm_calls", 0),
               "ate": metrics.get("ate", None), "pehe": metrics.get("pehe", None)}
        all_results.append(res)
        np.save(os.path.join(OUTPUT_DIR, f"W_{mode}.npy"), W_est)
    dfres = pd.DataFrame(all_results)
    dfres.to_csv(os.path.join(OUTPUT_DIR, "experiment_summary.csv"), index=False)
    logger.info("Experiment summary:\n%s", dfres)
    # Quick plot
    plt.figure(figsize=(6,4))
    plt.bar(dfres["mode"], [0 if v is None else v for v in dfres["pehe"].fillna(0)])
    plt.title("PEHE (where available)")
    plt.savefig(os.path.join(OUTPUT_DIR, "pehe_bar.png"))

if __name__ == "__main__":
    main()
