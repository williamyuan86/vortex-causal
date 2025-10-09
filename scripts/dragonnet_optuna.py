# scripts/dragonnet_optuna.py
import optuna
import json
import numpy as np
from src.effect_estimation.dragonnet import DragonNet, dragonnet_train, predict_dragonnet, ate_from_preds, pehe
from scripts.run_experiments import make_synthetic_pehe_dataset
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def objective(trial):
    # sample hyperparams
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    epochs = trial.suggest_int("epochs", 50, 200)
    hidden1 = trial.suggest_int("hidden1", 64, 256)
    hidden2 = trial.suggest_int("hidden2", 32, 128)
    alpha = trial.suggest_float("alpha", 0.1, 5.0)
    beta = trial.suggest_float("beta", 0.1, 5.0)
    # generate synthetic
    X, T, Y, Y0_true, Y1_true = make_synthetic_pehe_dataset(n=2000, d=10, seed=trial.number+1)
    model = DragonNet(X.shape[1], repr_dims=[hidden1, hidden2], head_dims=[max(32, hidden2//2)])
    # train with those params
    dragonnet_train(model, X, T, Y, epochs=epochs, batch_size=256, lr=lr, alpha=alpha, beta=beta, verbose=False)
    p, mu0, mu1 = predict_dragonnet(model, X)
    pehe_val = pehe(mu0, mu1, Y0_true, Y1_true)
    # we minimize PEHE
    return float(pehe_val)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    best = study.best_trial.params
    with open(os.path.join(OUTPUT_DIR, "optuna_result.json"), "w") as f:
        json.dump({"best_params": best, "value": study.best_value}, f, indent=2)
    print("Best params:", best)
