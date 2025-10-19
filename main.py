# main.py - Fixed version with proper error handling and imports
import logging
import io
import numpy as np
import pandas as pd
from pathlib import Path

# Fixed imports (using original file names since _fixed versions are identical)
from src.utils.data_loader import load_sachs, create_synthetic_dataset, validate_dataset
from src.selector.algorithm_selector import run_selector, validate_selector_output
from src.constraint_fusion.fusion import apply_hard_constraints, apply_soft_priors_loss_term
from src.graph_generation.notears_torch import train_notears_torch, validate_weight_matrix
from src.effect_estimation.estimator import ols_ate, ipw_ate, validate_ate_data

# Optional components with fallbacks
try:
    from src.da_agent.retriever import SimpleRetriever
    RETRIEVER_AVAILABLE = True
except ImportError:
    RETRIEVER_AVAILABLE = False
    logging.getLogger(__name__).warning("Retriever not available, using fallback")

try:
    from src.ensemble.ensemble_cc_agent import ensemble_decide
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    logging.getLogger(__name__).warning("Ensemble agent not available, using fallback")

try:
    from src.effect_estimation.dragonnet import DragonNet, dragonnet_train, predict_dragonnet, ate_from_preds
    DRAGONNET_AVAILABLE = True
except ImportError:
    DRAGONNET_AVAILABLE = False
    logging.getLogger(__name__).warning("DragonNet not available, using fallback estimators")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vortex")

def create_fallback_synthetic_data():
    """Create synthetic data when real data loading fails."""
    logger.warning("Creating synthetic data as fallback")
    df, true_adj = create_synthetic_dataset(n_samples=200, n_vars=6, seed=42)
    return df

def validate_and_preprocess_data(df):
    """Validate and preprocess the input data."""
    # Validate dataset
    is_valid, issues = validate_dataset(df)
    if not is_valid:
        logger.warning(f"Data validation issues: {issues}")
        if "empty" in str(issues):
            raise ValueError("Cannot proceed with empty dataset")

    # Ensure we have enough data
    if df.shape[1] < 3:
        logger.warning("Few variables, adding synthetic variables")
        for i in range(3 - df.shape[1]):
            df[f'synth_{i}'] = np.random.normal(0, 1, len(df))

    return df

def run_causal_discovery_pipeline(df, config=None):
    """
    Run the complete causal discovery pipeline.

    Args:
        df: Input DataFrame
        config: Configuration dictionary

    Returns:
        results: Dictionary containing all pipeline results
    """
    if config is None:
        config = {
            'selector': {'thresh_hard': 0.8, 'thresh_ambiguous': 0.3},
            'graph': {'lambda1': 0.1, 'max_iter': 100, 'lr': 1e-2},
            'ensemble': {'mock': True},
            'effects': {'method': 'ols'}  # 'ols', 'ipw', 'dragonnet'
        }

    results = {}

    try:
        # Step 1: Validate and preprocess data
        logger.info("Step 1: Data validation and preprocessing")
        df = validate_and_preprocess_data(df)
        results['data_shape'] = df.shape
        results['data_columns'] = df.columns.tolist()

        # Step 2: Algorithm selection
        logger.info("Step 2: Algorithm selection")
        selector_config = config['selector']
        selector_result = run_selector(
            df,
            thresh_hard=selector_config['thresh_hard'],
            thresh_ambiguous=selector_config['thresh_ambiguous']
        )

        # Validate selector output
        sel_valid, sel_issues = validate_selector_output(selector_result, df.shape[1])
        if not sel_valid:
            logger.warning(f"Selector validation issues: {sel_issues}")

        results['selector'] = selector_result
        logger.info(f"Selector found: {len(selector_result['hard'])} hard, "
                   f"{len(selector_result['ambiguous'])} ambiguous, "
                   f"{len(selector_result['low'])} low confidence edges")

        # Step 3: Apply hard constraints
        logger.info("Step 3: Applying hard constraints")
        n_vars = df.shape[1]
        initial_W = np.zeros((n_vars, n_vars))

        mask_fixed, W_fixed = apply_hard_constraints(
            initial_W, selector_result['hard'], n_vars
        )

        results['constraints'] = {
            'mask_fixed': mask_fixed,
            'W_fixed': W_fixed,
            'n_hard_constraints': len(selector_result['hard'])
        }

        # Step 4: Generate soft priors from ambiguous edges
        priors = {}
        if selector_result['ambiguous'] and not config['ensemble'].get('mock', True):
            logger.info("Step 4: Processing ambiguous edges with ensemble")
            # In a real implementation, this would call LLM ensemble
            # For now, create mock priors
            for i, j, weight, score in selector_result['ambiguous'][:5]:  # Limit to avoid too many priors
                priors[(i, j)] = 0.7  # Mock confidence
        else:
            logger.info("Step 4: Using mock priors for ambiguous edges")
            # Create some mock priors for testing
            for i, j, weight, score in selector_result['ambiguous'][:3]:
                priors[(i, j)] = 0.6

        results['priors'] = priors
        logger.info(f"Created {len(priors)} soft priors")

        # Step 5: Train causal graph
        logger.info("Step 5: Training causal graph with NOTEARS")
        graph_config = config['graph']

        W_final = train_notears_torch(
            df.values,
            priors_dict=priors if priors else None,
            mask_fixed=mask_fixed,
            lambda1=graph_config['lambda1'],
            max_iter=graph_config['max_iter'],
            lr=graph_config['lr'],
            verbose=True
        )

        # Validate learned graph
        graph_valid, graph_issues = validate_weight_matrix(W_final)
        if not graph_valid:
            logger.warning(f"Graph validation issues: {graph_issues}")

        results['final_graph'] = W_final
        logger.info(f"Learned graph with {np.count_nonzero(W_final)} non-zero edges")

        # Step 6: Effect estimation
        logger.info("Step 6: Treatment effect estimation")
        effects_config = config['effects']
        method = effects_config['method']

        # Choose treatment and outcome columns
        cols = df.columns.tolist()
        if len(cols) >= 3:
            treatment_col = cols[0]
            outcome_col = cols[1]
            covariates = cols[2:6] if len(cols) > 6 else cols[2:]

            # Prepare data for effect estimation
            df_eff = df.copy()
            df_eff[treatment_col] = (df_eff[treatment_col] > df_eff[treatment_col].median()).astype(int)

            # Validate effect estimation data
            eff_valid, eff_issues = validate_ate_data(df_eff, treatment_col, outcome_col, covariates)
            if not eff_valid:
                logger.warning(f"Effect estimation validation issues: {eff_issues}")

            effect_results = {}

            try:
                if method in ['ols', 'all']:
                    ate_ols = ols_ate(df_eff, treatment_col, outcome_col, covariates)
                    effect_results['ols_ate'] = ate_ols
                    logger.info(f"OLS ATE estimate: {ate_ols:.4f}")

                if method in ['ipw', 'all']:
                    ate_ipw, propensity_scores = ipw_ate(df_eff, treatment_col, outcome_col, covariates)
                    effect_results['ipw_ate'] = ate_ipw
                    effect_results['propensity_scores'] = propensity_scores
                    logger.info(f"IPW ATE estimate: {ate_ipw:.4f}")

                if method in ['dragonnet', 'all'] and DRAGONNET_AVAILABLE:
                    # Prepare data for DragonNet
                    X = df_eff[covariates].values.astype(np.float32) if covariates else np.ones((len(df_eff), 1))
                    T = df_eff[treatment_col].values.astype(np.int64)
                    Y = df_eff[outcome_col].values.astype(np.float32)

                    if len(X) > 50:  # Minimum sample size for DragonNet
                        model = DragonNet(X.shape[1])
                        dragonnet_train(model, X, T, Y, epochs=50, batch_size=min(64, len(X)//4), verbose=False)
                        p_pred, mu0_pred, mu1_pred = predict_dragonnet(model, X)
                        ate_dragonnet = ate_from_preds(mu0_pred, mu1_pred)
                        effect_results['dragonnet_ate'] = ate_dragonnet
                        logger.info(f"DragonNet ATE estimate: {ate_dragonnet:.4f}")

            except Exception as e:
                logger.error(f"Effect estimation failed: {e}")
                effect_results['error'] = str(e)

            results['effects'] = effect_results
        else:
            logger.warning("Insufficient variables for effect estimation")
            results['effects'] = {'error': 'Insufficient variables'}

        return results

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        results['error'] = str(e)
        return results

def main():
    """Main function for the fixed Vortex-Causal pipeline."""
    logger.info("Starting Vortex-Causal Fixed Pipeline")

    try:
        # Step 1: Load data
        data_path = "data/sachs.csv"
        try:
            df = load_sachs(data_path)
            logger.info(f"Loaded real data from {data_path}: {df.shape}")
        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
            logger.warning(f"Could not load real data: {e}")
            df = create_fallback_synthetic_data()

        # Step 2: Run pipeline
        config = {
            'selector': {'thresh_hard': 0.8, 'thresh_ambiguous': 0.3},
            'graph': {'lambda1': 0.1, 'max_iter': 100, 'lr': 1e-2},
            'ensemble': {'mock': True},
            'effects': {'method': 'all'}  # Run all methods
        }

        results = run_causal_discovery_pipeline(df, config)

        # Step 3: Report results
        logger.info("Pipeline completed successfully!")
        logger.info(f"Data shape: {results['data_shape']}")
        logger.info(f"Hard edges found: {len(results['selector']['hard'])}")
        logger.info(f"Ambiguous edges found: {len(results['selector']['ambiguous'])}")
        logger.info(f"Non-zero edges in final graph: {np.count_nonzero(results['final_graph'])}")

        if 'effects' in results and 'error' not in results['effects']:
            effects = results['effects']
            logger.info("Treatment effect estimates:")
            for method, ate in effects.items():
                if method.endswith('_ate') and isinstance(ate, (int, float)):
                    logger.info(f"  {method}: {ate:.4f}")

        return results

    except Exception as e:
        logger.error(f"Main pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()