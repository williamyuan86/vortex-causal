"""Tests for the fixed Vortex-Causal components."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

# Import fixed components
from src.utils.data_loader_fixed import load_sachs, create_synthetic_dataset, validate_dataset
from src.selector.algorithm_selector_fixed import run_selector, validate_selector_output
from src.constraint_fusion.fusion_fixed import apply_hard_constraints, apply_soft_priors_loss_term
from src.graph_generation.notears_torch_fixed import train_notears_torch, validate_weight_matrix
from src.effect_estimation.estimator_fixed import ols_ate, ipw_ate, validate_ate_data


class TestFixedComponents:
    """Test suite for fixed components."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data for testing."""
        df, true_adj = create_synthetic_dataset(n_samples=100, n_vars=5, seed=42)
        return df, true_adj

    def test_data_loader_fixed(self, synthetic_data):
        """Test fixed data loader functionality."""
        df, true_adj = synthetic_data

        # Test data validation
        is_valid, issues = validate_dataset(df)
        assert is_valid
        assert len(issues) == 0

        # Test synthetic dataset creation
        df2, adj2 = create_synthetic_dataset(n_samples=50, n_vars=4, seed=123)
        assert df2.shape == (50, 4)
        assert adj2.shape == (4, 4)

    def test_algorithm_selector_fixed(self, synthetic_data):
        """Test fixed algorithm selector."""
        df, _ = synthetic_data

        # Test selector with valid data
        result = run_selector(df, thresh_hard=0.8, thresh_ambiguous=0.3)

        # Validate output structure
        assert isinstance(result, dict)
        assert 'hard' in result
        assert 'ambiguous' in result
        assert 'low' in result
        assert 'W1' in result
        assert 'W2' in result

        # Test selector validation
        is_valid, issues = validate_selector_output(result, df.shape[1])
        assert is_valid, f"Selector validation failed: {issues}"

        # Test edge categorization
        total_edges = len(result['hard']) + len(result['ambiguous']) + len(result['low'])
        max_edges = df.shape[1] * (df.shape[1] - 1)
        assert total_edges <= max_edges

    def test_constraint_fusion_fixed(self, synthetic_data):
        """Test fixed constraint fusion functionality."""
        df, _ = synthetic_data
        n_vars = df.shape[1]

        # Test hard constraints
        hard_edges = [(0, 1, 0.7, 0.9), (1, 2, -0.5, 0.8), (2, 3, 0.6, 0.95)]
        initial_W = np.zeros((n_vars, n_vars))

        mask_fixed, W_fixed = apply_hard_constraints(initial_W, hard_edges, n_vars)

        # Validate constraints
        assert mask_fixed.shape == (n_vars, n_vars)
        assert W_fixed.shape == (n_vars, n_vars)
        assert np.all(mask_fixed.diagonal())  # Diagonal should be fixed
        assert np.all(W_fixed.diagonal() == 0)  # Diagonal should be zero

        # Check that hard edges are applied
        for i, j, w, conf in hard_edges:
            if i < n_vars and j < n_vars and i != j:
                assert mask_fixed[i, j]
                assert abs(W_fixed[i, j] - w) < 1e-6

        # Test soft priors
        priors = {(0, 1): 0.8, (1, 2): 0.6}
        loss = apply_soft_priors_loss_term(W_fixed, priors, lam=1.0)
        assert isinstance(loss, (float, np.floating))
        assert loss >= 0

        # Test empty priors
        loss_empty = apply_soft_priors_loss_term(W_fixed, {})
        assert loss_empty == 0.0

    def test_graph_generation_fixed(self, synthetic_data):
        """Test fixed graph generation."""
        df, _ = synthetic_data

        # Test NOTEARS training
        W = train_notears_torch(
            df.values,
            lambda1=0.1,
            max_iter=20,  # Short for testing
            lr=1e-2,
            verbose=False
        )

        # Validate output
        assert W.shape == (df.shape[1], df.shape[1])
        assert not np.any(np.isnan(W))
        assert not np.any(np.isinf(W))

        # Test weight matrix validation
        is_valid, issues = validate_weight_matrix(W)
        assert is_valid, f"Weight matrix validation failed: {issues}"

        # Test with priors
        priors = {(0, 1): 0.7, (1, 2): 0.5}
        W_with_priors = train_notears_torch(
            df.values,
            priors_dict=priors,
            lambda1=0.1,
            max_iter=20,
            lr=1e-2,
            verbose=False
        )
        assert W_with_priors.shape == W.shape

        # Test with mask
        mask_fixed = np.zeros((df.shape[1], df.shape[1]), dtype=bool)
        mask_fixed[0, 1] = True

        W_with_mask = train_notears_torch(
            df.values,
            mask_fixed=mask_fixed,
            lambda1=0.1,
            max_iter=20,
            lr=1e-2,
            verbose=False
        )
        assert W_with_mask.shape == W.shape

    def test_effect_estimation_fixed(self, synthetic_data):
        """Test fixed effect estimation."""
        df, _ = synthetic_data

        # Prepare data for effect estimation
        treatment_col = df.columns[0]
        outcome_col = df.columns[1]
        covariates = df.columns[2:4].tolist()

        df_eff = df.copy()
        df_eff[treatment_col] = (df_eff[treatment_col] > df_eff[treatment_col].median()).astype(int)

        # Test effect estimation data validation
        is_valid, issues = validate_ate_data(df_eff, treatment_col, outcome_col, covariates)
        assert is_valid, f"Effect estimation data validation failed: {issues}"

        # Test OLS estimation
        ate_ols = ols_ate(df_eff, treatment_col, outcome_col, covariates)
        assert isinstance(ate_ols, (float, np.floating))
        assert not np.isnan(ate_ols)

        # Test IPW estimation
        ate_ipw, propensity_scores = ipw_ate(df_eff, treatment_col, outcome_col, covariates)
        assert isinstance(ate_ipw, (float, np.floating))
        assert isinstance(propensity_scores, np.ndarray)
        assert len(propensity_scores) == len(df_eff)
        assert np.all((propensity_scores > 0) & (propensity_scores < 1))

        # Test without covariates
        ate_ols_no_cov = ols_ate(df_eff, treatment_col, outcome_col, [])
        assert isinstance(ate_ols_no_cov, (float, np.floating))

        ate_ipw_no_cov, ps_no_cov = ipw_ate(df_eff, treatment_col, outcome_col, [])
        assert isinstance(ate_ipw_no_cov, (float, np.floating))
        assert len(ps_no_cov) == len(df_eff)

    def test_fixed_component_integration(self, synthetic_data):
        """Test integration of fixed components."""
        df, _ = synthetic_data

        # Step 1: Run selector
        selector_result = run_selector(df, thresh_hard=0.8, thresh_ambiguous=0.3)

        # Step 2: Apply constraints
        n_vars = df.shape[1]
        mask_fixed, W_fixed = apply_hard_constraints(
            np.zeros((n_vars, n_vars)), selector_result['hard'], n_vars
        )

        # Step 3: Train graph
        W_final = train_notears_torch(
            df.values,
            mask_fixed=mask_fixed,
            lambda1=0.1,
            max_iter=30,
            lr=1e-2,
            verbose=False
        )

        # Step 4: Estimate effects
        treatment_col = df.columns[0]
        outcome_col = df.columns[1]
        covariates = df.columns[2:4].tolist()

        df_eff = df.copy()
        df_eff[treatment_col] = (df_eff[treatment_col] > df_eff[treatment_col].median()).astype(int)

        ate_ols = ols_ate(df_eff, treatment_col, outcome_col, covariates)
        ate_ipw, _ = ipw_ate(df_eff, treatment_col, outcome_col, covariates)

        # Validate all components worked
        assert W_final.shape == (n_vars, n_vars)
        assert isinstance(ate_ols, (float, np.floating))
        assert isinstance(ate_ipw, (float, np.floating))
        assert not np.isnan(ate_ols)
        assert not np.isnan(ate_ipw)

    def test_error_handling(self, synthetic_data):
        """Test error handling in fixed components."""
        df, _ = synthetic_data

        # Test with invalid thresholds
        with pytest.raises(ValueError):
            run_selector(df, thresh_hard=0.5, thresh_ambiguous=0.7)  # Invalid: thresh_ambiguous > thresh_hard

        # Test with empty data
        empty_df = pd.DataFrame()
        is_valid, issues = validate_dataset(empty_df)
        assert not is_valid
        assert "empty" in str(issues).lower()

        # Test with insufficient data
        tiny_df = pd.DataFrame({'A': [1], 'B': [2]})
        is_valid, issues = validate_dataset(tiny_df)
        assert not is_valid  # Too few samples

        # Test with constant treatment
        const_treatment_df = synthetic_data[0].copy()
        const_treatment_df[const_treatment_df.columns[0]] = 1
        ate_ols = ols_ate(const_treatment_df, const_treatment_df.columns[0], const_treatment_df.columns[1], [])
        assert np.isnan(ate_ols)

    def test_temporary_file_loading(self):
        """Test loading from temporary files."""
        # Create temporary CSV file with enough samples
        np.random.seed(42)
        n_samples = 50
        df_test = pd.DataFrame({
            'A': np.random.normal(0, 1, n_samples),
            'B': np.random.normal(0, 1, n_samples),
            'C': np.random.normal(0, 1, n_samples)
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_test.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            # Test loading
            df_loaded = load_sachs(temp_file)
            assert df_loaded.shape == df_test.shape
            assert list(df_loaded.columns) == list(df_test.columns)

            # Test validation
            is_valid, issues = validate_dataset(df_loaded)
            assert is_valid

        finally:
            os.unlink(temp_file)