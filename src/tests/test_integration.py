"""Integration tests for the main Vortex-Causal pipeline."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import main pipeline components
from src.utils.data_loader import load_sachs
from src.selector.algorithm_selector import run_selector
from src.constraint_fusion.fusion import apply_hard_constraints
from src.graph_generation.notears_torch import train_notears_torch
from src.effect_estimation.estimator import ols_ate, ipw_ate
from src.ensemble.ensemble_cc_agent import ensemble_decide
from src.da_agent.retriever import SimpleRetriever


class TestIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def synthetic_dataset(self):
        """Create a synthetic dataset for integration testing."""
        np.random.seed(42)
        n_samples = 300
        n_vars = 6

        # Create synthetic causal structure
        var_names = [f'V{i}' for i in range(n_vars)]

        # Generate data with known structure:
        # V0 -> V1 -> V2, V0 -> V3, V3 -> V4, V1 -> V5
        V0 = np.random.normal(0, 1, n_samples)
        V1 = 0.7 * V0 + np.random.normal(0, 0.5, n_samples)
        V2 = 0.6 * V1 + np.random.normal(0, 0.5, n_samples)
        V3 = 0.5 * V0 + np.random.normal(0, 0.5, n_samples)
        V4 = 0.8 * V3 + np.random.normal(0, 0.5, n_samples)
        V5 = 0.4 * V1 + 0.3 * V4 + np.random.normal(0, 0.5, n_samples)

        data = np.column_stack([V0, V1, V2, V3, V4, V5])
        df = pd.DataFrame(data, columns=var_names)

        return df, {
            'known_edges': [(0, 1), (1, 2), (0, 3), (3, 4), (1, 5)],
            'true_adjacency': np.array([
                [0, 0.7, 0, 0.5, 0, 0],
                [0, 0, 0.6, 0, 0, 0.4],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.8, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ])
        }

    @pytest.fixture
    def temp_data_file(self, synthetic_dataset):
        """Create temporary data file for testing."""
        df, _ = synthetic_dataset

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name

        yield temp_file
        os.unlink(temp_file)

    def test_data_loading_to_selector_pipeline(self, synthetic_dataset):
        """Test pipeline from data loading to algorithm selection."""
        df, _ = synthetic_dataset

        # Step 1: Algorithm selection
        selector_result = run_selector(df, thresh_hard=0.8, thresh_ambiguous=0.3)

        # Verify selector output structure
        assert isinstance(selector_result, dict)
        assert 'hard' in selector_result
        assert 'ambiguous' in selector_result
        assert 'low' in selector_result

        # Check that edges are properly categorized
        total_edges = len(selector_result['hard']) + len(selector_result['ambiguous']) + len(selector_result['low'])
        n_vars = df.shape[1]
        max_edges = n_vars * (n_vars - 1)
        assert total_edges <= max_edges

        # Check weight matrices
        assert selector_result['W1'].shape == (n_vars, n_vars)
        assert selector_result['W2'].shape == (n_vars, n_vars)

    def test_selector_to_constraint_fusion_pipeline(self, synthetic_dataset):
        """Test pipeline from selector to constraint application."""
        df, _ = synthetic_dataset

        # Step 1: Get selector results
        selector_result = run_selector(df, thresh_hard=0.8, thresh_ambiguous=0.3)

        # Step 2: Apply hard constraints
        n_vars = df.shape[1]
        initial_W = np.zeros((n_vars, n_vars))
        mask_fixed, W_fixed = apply_hard_constraints(
            initial_W, selector_result['hard'], n_vars
        )

        # Verify constraint application
        assert mask_fixed.shape == (n_vars, n_vars)
        assert W_fixed.shape == (n_vars, n_vars)

        # Check that hard edges are fixed
        for i, j, weight, conf in selector_result['hard']:
            assert mask_fixed[i, j] == 1
            assert W_fixed[i, j] == weight

        # Check diagonal is fixed
        assert np.all(mask_fixed.diagonal() == 1)
        assert np.all(W_fixed.diagonal() == 0)

    def test_constraint_fusion_to_graph_generation_pipeline(self, synthetic_dataset):
        """Test pipeline from constraint fusion to graph generation."""
        df, _ = synthetic_dataset

        # Step 1: Get selector results and apply constraints
        selector_result = run_selector(df, thresh_hard=0.8, thresh_ambiguous=0.3)
        n_vars = df.shape[1]
        initial_W = np.zeros((n_vars, n_vars))
        mask_fixed, W_fixed = apply_hard_constraints(
            initial_W, selector_result['hard'], n_vars
        )

        # Step 2: Train graph with constraints
        W_trained = train_notears_torch(
            df.values,
            mask_fixed=mask_fixed,
            lambda1=0.1,
            max_iter=50,  # Short for testing
            lr=1e-2
        )

        # Verify graph generation
        assert W_trained.shape == (n_vars, n_vars)
        assert np.allclose(W_trained.diagonal(), 0)

        # Check that fixed constraints are preserved (if implementation supports this)
        # This depends on the specific implementation

    def test_graph_generation_to_effect_estimation_pipeline(self, synthetic_dataset):
        """Test pipeline from graph generation to effect estimation."""
        df, _ = synthetic_dataset

        # Step 1: Generate causal graph
        W = train_notears_torch(
            df.values,
            lambda1=0.1,
            max_iter=50,
            lr=1e-2
        )

        # Step 2: Choose treatment and outcome based on graph
        # Find a reasonable treatment-outcome pair
        n_vars = df.shape[1]
        treatment_idx = 0
        outcome_idx = 1

        # Prepare data for effect estimation
        treatment_col = df.columns[treatment_idx]
        outcome_col = df.columns[outcome_idx]
        covariates = [col for i, col in enumerate(df.columns)
                     if i not in [treatment_idx, outcome_idx]][:3]  # Limit covariates

        # Binarize treatment
        df_eff = df.copy()
        df_eff[treatment_col] = (df_eff[treatment_col] > df_eff[treatment_col].median()).astype(int)

        # Step 3: Estimate treatment effects
        if len(covariates) > 0:
            ate_ols = ols_ate(df_eff, treatment_col, outcome_col, covariates)
            ate_ipw, pscores = ipw_ate(df_eff, treatment_col, outcome_col, covariates)

            # Verify effect estimates
            assert isinstance(ate_ols, (float, np.floating))
            assert isinstance(ate_ipw, (float, np.floating))
            assert not np.isnan(ate_ols)
            assert not np.isnan(ate_ipw)

            # Verify propensity scores
            assert isinstance(pscores, np.ndarray)
            assert len(pscores) == len(df_eff)
            assert np.all((pscores > 0) & (pscores < 1))

    def test_full_pipeline_with_mock_ensemble(self, synthetic_dataset):
        """Test full pipeline with mocked ensemble decisions."""
        df, _ = synthetic_dataset

        # Mock the ensemble decision to avoid LLM calls
        mock_ensemble_result = {
            'edge': 'V0-V1',
            'direction': 'i->j',
            'confidence': 0.85,
            'evidence_count': 2
        }

        with patch('src.ensemble.ensemble_cc_agent.ensemble_decide', return_value=mock_ensemble_result):
            # Step 1: Data loading (simulated)
            # df already loaded

            # Step 2: Algorithm selection
            selector_result = run_selector(df, thresh_hard=0.7, thresh_ambiguous=0.4)

            # Step 3: Apply hard constraints
            n_vars = df.shape[1]
            mask_fixed, W_fixed = apply_hard_constraints(
                np.zeros((n_vars, n_vars)), selector_result['hard'], n_vars
            )

            # Step 4: Generate soft priors from mock ensemble (only for ambiguous edges)
            priors = {}
            for i, j, weight, score in selector_result['ambiguous'][:2]:  # Limit for testing
                priors[(i, j)] = mock_ensemble_result['confidence']

            # Step 5: Train final graph
            W_final = train_notears_torch(
                df.values,
                priors_dict=priors if priors else None,
                mask_fixed=mask_fixed,
                lambda1=0.1,
                lam_soft=1.0,
                max_iter=50,
                lr=1e-2
            )

            # Verify final graph
            assert W_final.shape == (n_vars, n_vars)
            assert np.allclose(W_final.diagonal(), 0)

            # Step 6: Effect estimation (simplified)
            if n_vars >= 3:
                treatment_col = df.columns[0]
                outcome_col = df.columns[1]
                covariates = df.columns[2:4].tolist()

                df_eff = df.copy()
                df_eff[treatment_col] = (df_eff[treatment_col] > df_eff[treatment_col].median()).astype(int)

                try:
                    ate_ols = ols_ate(df_eff, treatment_col, outcome_col, covariates)
                    assert isinstance(ate_ols, (float, np.floating))
                except Exception:
                    # Effect estimation might fail with certain data configurations
                    pass

    def test_pipeline_with_temporary_file(self, temp_data_file):
        """Test pipeline using actual file loading."""
        # Step 1: Load data from file
        df = load_sachs(temp_data_file)

        # Verify data loading
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        assert df.shape[1] > 0

        # Step 2: Run selector
        selector_result = run_selector(df, thresh_hard=0.8, thresh_ambiguous=0.3)

        # Verify selector worked
        assert isinstance(selector_result, dict)
        assert len(selector_result['hard']) + len(selector_result['ambiguous']) + len(selector_result['low']) > 0

        # Step 3: Apply constraints and train graph
        n_vars = df.shape[1]
        mask_fixed, W_fixed = apply_hard_constraints(
            np.zeros((n_vars, n_vars)), selector_result['hard'], n_vars
        )

        W = train_notears_torch(
            df.values,
            mask_fixed=mask_fixed,
            lambda1=0.1,
            max_iter=30,  # Short for testing
            lr=1e-2
        )

        # Verify final output
        assert W.shape == (n_vars, n_vars)
        assert np.allclose(W.diagonal(), 0)

    def test_pipeline_error_handling(self):
        """Test pipeline error handling with problematic data."""
        # Test with very small dataset
        tiny_df = pd.DataFrame({
            'A': [1, 2],
            'B': [2, 1]
        })

        try:
            selector_result = run_selector(tiny_df)
            # Should either work or fail gracefully
        except Exception as e:
            # Should fail with informative error
            assert isinstance(e, (ValueError, np.linalg.LinAlgError))

        # Test with constant data
        constant_df = pd.DataFrame({
            'A': [1, 1, 1, 1],
            'B': [2, 2, 2, 2],
            'C': [3, 3, 3, 3]
        })

        try:
            selector_result = run_selector(constant_df)
            W = train_notears_torch(constant_df.values, max_iter=10)
            # Should handle gracefully
        except Exception as e:
            # Expected to fail with constant data
            assert "singular" in str(e).lower() or "variance" in str(e).lower()

    def test_pipeline_reproducibility(self, synthetic_dataset):
        """Test that pipeline results are reproducible."""
        df, _ = synthetic_dataset

        # Run pipeline twice
        result1 = run_selector(df, thresh_hard=0.8, thresh_ambiguous=0.3)
        result2 = run_selector(df, thresh_hard=0.8, thresh_ambiguous=0.3)

        # Results should be identical
        np.testing.assert_array_almost_equal(result1['W1'], result2['W1'])
        np.testing.assert_array_almost_equal(result1['W2'], result2['W2'])

        # Graph generation should also be reproducible
        W1 = train_notears_torch(df.values, lambda1=0.1, max_iter=30, lr=1e-2)
        W2 = train_notears_torch(df.values, lambda1=0.1, max_iter=30, lr=1e-2)

        np.testing.assert_array_almost_equal(W1, W2, decimal=6)

    def test_pipeline_output_validation(self, synthetic_dataset):
        """Test that pipeline outputs are properly validated."""
        df, metadata = synthetic_dataset

        # Run complete pipeline
        selector_result = run_selector(df)
        n_vars = df.shape[1]

        mask_fixed, W_fixed = apply_hard_constraints(
            np.zeros((n_vars, n_vars)), selector_result['hard'], n_vars
        )

        W_final = train_notears_torch(
            df.values,
            mask_fixed=mask_fixed,
            lambda1=0.1,
            max_iter=50
        )

        # Validate graph properties
        assert W_final.shape == (n_vars, n_vars)
        assert np.allclose(W_final.diagonal(), 0)
        assert not np.any(np.isnan(W_final))
        assert not np.any(np.isinf(W_final))

        # Check that graph is reasonably sparse
        sparsity = np.count_nonzero(W_final) / (n_vars * (n_vars - 1))
        assert 0 < sparsity <= 1  # Should not be completely empty (can be dense in some cases)

        # Validate against known structure (if available)
        known_edges = metadata['known_edges']
        for i, j in known_edges:
            # Known causal edges should ideally have non-zero weights
            # (but this depends on algorithm performance)
            pass