# src/effect_estimation/estimator.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

def ipw_ate(df, treatment_col, outcome_col, covariates):
    """
    Simple IPW ATE estimate:
    - Estimate propensity with logistic regression
    - Compute weighted difference in outcomes
    """
    X = df[covariates].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    model = LogisticRegression(max_iter=200)
    model.fit(X, T)
    ps = model.predict_proba(X)[:,1]
    eps=1e-6
    w = T/ (ps+eps) - (1-T)/(1-ps+eps)
    ate = np.mean(w * Y)
    return ate, ps

def ols_ate(df, treatment_col, outcome_col, covariates):
    X = df[[treatment_col] + covariates]
    y = df[outcome_col]
    model = LinearRegression().fit(X, y)
    coef = model.coef_[0]  # coefficient of treatment
    return coef
