# src/effect_estimation/estimator_fixed.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

def ols_ate(df, treatment_col, outcome_col, covariates=None):
    """
    Estimate Average Treatment Effect using Ordinary Least Squares.

    Args:
        df: DataFrame containing treatment, outcome, and covariates
        treatment_col: Name of treatment column (binary)
        outcome_col: Name of outcome column
        covariates: List of covariate column names (optional)

    Returns:
        ate: Estimated average treatment effect
    """
    try:
        # Validate inputs
        if treatment_col not in df.columns:
            raise KeyError(f"Treatment column '{treatment_col}' not found")
        if outcome_col not in df.columns:
            raise KeyError(f"Outcome column '{outcome_col}' not found")

        # Prepare data
        y = df[outcome_col].values
        t = df[treatment_col].values

        # Check for constant treatment
        if np.unique(t).size < 2:
            warnings.warn("Treatment is constant, ATE estimate undefined")
            return np.nan

        # Build design matrix
        X_design = [t]
        X_design_labels = [treatment_col]

        if covariates:
            for cov in covariates:
                if cov in df.columns:
                    X_design.append(df[cov].values)
                    X_design_labels.append(cov)
                else:
                    warnings.warn(f"Covariate '{cov}' not found, skipping")

        X = np.column_stack(X_design)
        X = sm.add_constant(X)  # Add intercept

        # Fit OLS model
        model = sm.OLS(y, X).fit()

        # Return treatment coefficient
        treatment_idx = X_design_labels.index(treatment_col) + 1  # +1 for constant
        ate = model.params[treatment_idx]

        return float(ate)

    except Exception as e:
        warnings.warn(f"OLS ATE estimation failed: {e}")
        return np.nan

def ipw_ate(df, treatment_col, outcome_col, covariates=None):
    """
    Estimate Average Treatment Effect using Inverse Probability Weighting.

    Args:
        df: DataFrame containing treatment, outcome, and covariates
        treatment_col: Name of treatment column (binary)
        outcome_col: Name of outcome column
        covariates: List of covariate column names (optional)

    Returns:
        ate: Estimated average treatment effect
        propensity_scores: Estimated propensity scores
    """
    try:
        # Validate inputs
        if treatment_col not in df.columns:
            raise KeyError(f"Treatment column '{treatment_col}' not found")
        if outcome_col not in df.columns:
            raise KeyError(f"Outcome column '{outcome_col}' not found")

        # Prepare data
        y = df[outcome_col].values
        t = df[treatment_col].values

        # Check for constant treatment
        if np.unique(t).size < 2:
            warnings.warn("Treatment is constant, IPW estimate undefined")
            return np.nan, np.full(len(t), 0.5)

        # Build propensity score model
        X_prop = []
        if covariates:
            for cov in covariates:
                if cov in df.columns:
                    X_prop.append(df[cov].values)
                else:
                    warnings.warn(f"Covariate '{cov}' not found, skipping")

        if X_prop:
            X_design = np.column_stack(X_prop)
        else:
            # No covariates, use intercept only
            X_design = np.ones((len(t), 1))

        # Fit propensity score model
        try:
            prop_model = LogisticRegression(max_iter=1000, random_state=42)
            prop_model.fit(X_design, t)
            propensity_scores = prop_model.predict_proba(X_design)[:, 1]
        except Exception:
            # Fallback: use constant propensity
            propensity_scores = np.full(len(t), np.mean(t))

        # Trim propensity scores to avoid extreme weights
        epsilon = 1e-6
        propensity_scores = np.clip(propensity_scores, epsilon, 1 - epsilon)

        # Calculate IPW weights
        weights = np.where(t == 1, 1 / propensity_scores, 1 / (1 - propensity_scores))

        # Stabilize weights
        weights = weights / np.mean(weights)

        # Check for extreme weights
        if np.max(weights) > 1000:
            warnings.warn("Extreme weights detected, IPW estimate may be unstable")

        # Calculate weighted outcomes
        treated_outcome = np.sum(weights[t == 1] * y[t == 1]) / np.sum(weights[t == 1])
        control_outcome = np.sum(weights[t == 0] * y[t == 0]) / np.sum(weights[t == 0])

        ate = treated_outcome - control_outcome

        return float(ate), propensity_scores

    except Exception as e:
        warnings.warn(f"IPW ATE estimation failed: {e}")
        return np.nan, np.full(len(df), 0.5)

def validate_ate_data(df, treatment_col, outcome_col, covariates=None):
    """
    Validate data for ATE estimation.

    Args:
        df: DataFrame to validate
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        covariates: List of covariate names

    Returns:
        is_valid: Boolean indicating validity
        issues: List of issues found
    """
    issues = []

    # Check DataFrame
    if not isinstance(df, pd.DataFrame):
        issues.append("Input must be pandas DataFrame")
        return False, issues

    # Check required columns
    required_cols = [treatment_col, outcome_col]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Required column '{col}' not found")

    # Check data types
    if treatment_col in df.columns:
        if not np.issubdtype(df[treatment_col].dtype, np.number):
            issues.append(f"Treatment column '{treatment_col}' must be numeric")

        # Check treatment values
        unique_treatments = df[treatment_col].nunique()
        if unique_treatments < 2:
            issues.append("Treatment must have variation")

    if outcome_col in df.columns:
        if not np.issubdtype(df[outcome_col].dtype, np.number):
            issues.append(f"Outcome column '{outcome_col}' must be numeric")

    # Check covariates
    if covariates:
        for cov in covariates:
            if cov not in df.columns:
                issues.append(f"Covariate '{cov}' not found")
            elif not np.issubdtype(df[cov].dtype, np.number):
                issues.append(f"Covariate '{cov}' must be numeric")

    # Check missing values
    cols_to_check = [treatment_col, outcome_col] + (covariates or [])
    for col in cols_to_check:
        if col in df.columns and df[col].isnull().any():
            issues.append(f"Column '{col}' contains missing values")

    # Check sample size
    n_samples = len(df)
    if n_samples < 50:
        issues.append(f"Small sample size ({n_samples}), estimates may be unreliable")

    return len(issues) == 0, issues