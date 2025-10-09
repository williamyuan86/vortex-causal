# main.py
import logging
import io
from src.utils.data_loader import load_sachs
from src.selector.algorithm_selector import run_selector
from src.da_agent.retriever import SimpleRetriever
from src.ensemble.ensemble_cc_agent import ensemble_decide
from src.constraint_fusion.fusion import apply_hard_constraints, apply_soft_priors_loss_term
from src.graph_generation.notears_al import get_weighted_adjacency
from src.effect_estimation.estimator import ols_ate, ipw_ate
import numpy as np
from src.effect_estimation.dragonnet import DragonNet, dragonnet_train, predict_dragonnet, ate_from_preds, pehe
from src.ensemble.ensemble_cc_agent_openai import ensemble_decide
from src.graph_generation.notears_torch import train_notears_torch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vortex")

def main():
    # 1) load data
    df = load_sachs("data/sachs.csv")
    cols = df.columns.tolist()
    logger.info(f"Loaded Sachs with shape {df.shape}")

    # 2) adaptive selector -> hard / ambiguous / low
    sel = run_selector(df, thresh_hard=0.9, thresh_ambiguous=0.4)
    logger.info(f"Selector produced hard {len(sel['hard'])} ambiguous {len(sel['ambiguous'])} low {len(sel['low'])}")

    # 3) apply hard edges as initial constraints
    # W_init = get_weighted_adjacency(df, lambda1=0.05)
    W_refined = train_notears_torch(
        df.values,
        priors_dict=priors,
        mask_fixed=mask_fixed,
        lambda1=0.02,
        lam_soft=1.0,
        max_iter=200
    )
    logger.info(f"Trained NOTEARS with soft priors, resulting W shape={W_refined.shape}")
    mask_fixed, W_fixed = apply_hard_constraints(W_refined, sel['hard'], df.shape[1])
    # 4) Build retriever (if doc index exists)
    retriever = SimpleRetriever()
    # if not built yet: retriever.build(docs_list)

    # 5) For ambiguous edges, call ensemble (LLM) only for those edges — THIS IS COST SAVING
    var_names = cols
    ensemble_results = []
    for (i,j,w,score) in sel['ambiguous']:
        var_i = var_names[i]
        var_j = var_names[j]
        query = f"Does {var_i} cause {var_j}?"
        evidence = retriever.retrieve(query, k=3) if retriever.index is not None else [f"Placeholder evidence for {var_i}-{var_j}"]
        model_list = ["gpt-4o-mini", "qwen2-72b-chat", "claude-3-opus"]
        res = ensemble_decide((var_i, var_j), evidence, model_list)
        ensemble_results.append((i,j,res))
        logger.info(f"Edge {var_i}->{var_j} -> {res['direction']} conf={res['confidence']:.2f}")

    # 6) convert ensemble results to priors dict for soft constraints
    priors = {}
    hard_from_ensemble = []
    for (i,j,res) in ensemble_results:
        conf = res['confidence']
        if res['direction'] == 'i->j':
            priors[(i,j)] = conf
            if conf > 0.95:
                hard_from_ensemble.append((i,j,1.0,conf))
        elif res['direction'] == 'j->i':
            priors[(j,i)] = conf
            if conf > 0.95:
                hard_from_ensemble.append((j,i,1.0,conf))
        else:
            # no link judged
            priors[(i,j)] = 1.0 - res['confidence']

    # 7) merge hard constraints and re-run graph learner (here we call same get_weighted_adjacency as placeholder)
    # In a real NOTEARS integration you'd pass mask_fixed & priors into the optimizer
    W_refined = get_weighted_adjacency(df, lambda1=0.02)

    # 8) For demo: pick a treatment/outcome pair (needs domain knowledge) - here use two first cols

# ------------- DragonNet usage example --------------
    # assume df is pandas DataFrame with columns
    # choose treatment and outcome columns (业务上需要由你指定)
    treatment_col = 'treatment'  # 替换为真实列名
    outcome_col = 'outcome'      # 替换为真实列名
    covariates = [c for c in df.columns if c not in [treatment_col, outcome_col]]

    # prepare X, T, Y arrays
    X = df[covariates].values.astype(np.float32)
    # make sure treatment is binary 0/1 - here as an example binarizing by median
    T = (df[treatment_col] > df[treatment_col].median()).astype(int).values
    Y = df[outcome_col].values.astype(np.float32)

    # train dragonnet
    model = DragonNet(X.shape[1])
    dragonnet_train(model, X, T, Y, epochs=200, batch_size=128, lr=1e-3)

    # predict potential outcomes
    p, mu0, mu1 = predict_dragonnet(model, X)
    ate_est = ate_from_preds(mu0, mu1)
    print("Estimated ATE:", ate_est)




    treatment = cols[0]
    outcome = cols[1]
    covs = cols[2:6] if len(cols) > 6 else cols[2:]
    try:
        ate_ipw, ps = ipw_ate(df.assign(**{treatment: (df[treatment] > df[treatment].median()).astype(int)}),
                              treatment, outcome, covs)
        logger.info(f"IPW ATE estimate: {ate_ipw}")
    except Exception as e:
        logger.warning("IPW ATE failed (demo): " + str(e))

    try:
        ate_ols = ols_ate(df.assign(**{treatment: (df[treatment] > df[treatment].median()).astype(int)}),
                          treatment, outcome, covs)
        logger.info(f"OLS estimate (treatment coef): {ate_ols}")
    except Exception as e:
        logger.warning("OLS ATE failed: " + str(e))

if __name__ == "__main__":
    main()
