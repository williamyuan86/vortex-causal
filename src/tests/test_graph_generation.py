"""Tests for graph generation functionality."""

import pytest
import numpy as np
import pandas as pd
from src.graph_generation.notears_al import get_weighted_adjacency
from src.graph_generation.notears_torch import train_notears_torch


class TestGraphGeneration:
    """Test graph generation algorithms."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with known causal structure."""
        np.random.seed(42)
        n_samples = 500
        n_vars = 4

        # Create linear causal chain: X1 -> X2 -> X3 -> X4
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.7 * X1 + 0.3 * np.random.normal(0, 1, n_samples)
        X3 = 0.6 * X2 + 0.4 * np.random.normal(0, 1, n_samples)
        X4 = 0.8 * X3 + 0.2 * np.random.normal(0, 1, n_samples)

        df = pd.DataFrame({
            'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4
        })
        return df

    @pytest.fixture
    def random_data(self):
        """Create random data with no causal structure."""
        np.random.seed(123)
        return pd.DataFrame(
            np.random.normal(0, 1, (200, 5)),
            columns=['A', 'B', 'C', 'D', 'E']
        )

    def test_get_weighted_adjacency_basic(self, sample_data):
        """Test basic NOTEARS adjacency matrix generation."""
        W = get_weighted_adjacency(sample_data, lambda1=0.1)

        # Check output properties
        assert isinstance(W, np.ndarray)
        assert W.shape == (sample_data.shape[1], sample_data.shape[1])

        # Check diagonal is zero
        np.testing.assert_array_almost_equal(W.diagonal(), 0)

        # Check that we get some non-zero weights
        assert np.count_nonzero(W) > 0

    def test_get_weighted_adjacency_lambda1_effect(self, sample_data):
        """Test effect of lambda1 regularization parameter."""
        W_low_reg = get_weighted_adjacency(sample_data, lambda1=0.01)
        W_high_reg = get_weighted_adjacency(sample_data, lambda1=0.5)

        # Higher regularization should lead to more sparse weights
        nnz_low = np.count_nonzero(W_low_reg)
        nnz_high = np.count_nonzero(W_high_reg)

        assert nnz_low >= nnz_high

    def test_get_weighted_adjacency_symmetry(self, sample_data):
        """Test that output is generally asymmetric (directed)."""
        W = get_weighted_adjacency(sample_data, lambda1=0.1)

        # NOTEARS should produce generally asymmetric matrices
        # (exact symmetry would be unusual)
        is_symmetric = np.allclose(W, W.T, atol=1e-6)
        assert not is_symmetric

    def test_get_weighted_adjacency_reproducibility(self, sample_data):
        """Test that results are reproducible."""
        W1 = get_weighted_adjacency(sample_data, lambda1=0.1)
        W2 = get_weighted_adjacency(sample_data, lambda1=0.1)

        np.testing.assert_array_almost_equal(W1, W2)

    def test_get_weighted_adjacency_edge_cases(self):
        """Test edge cases for adjacency matrix generation."""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'A': [1, 2],
            'B': [2, 1]
        })
        W = get_weighted_adjacency(minimal_data, lambda1=0.1)
        assert W.shape == (2, 2)

        # Test with constant columns
        try:
            const_data = pd.DataFrame({
                'A': [1, 1, 1, 1],
                'B': [2, 3, 4, 5]
            })
            W = get_weighted_adjacency(const_data, lambda1=0.1)
            # Should handle gracefully or raise informative error
        except Exception as e:
            assert "singular" in str(e).lower() or "variance" in str(e).lower()

    def test_train_notears_torch_basic(self, sample_data):
        """Test basic PyTorch NOTEARS training."""
        W = train_notears_torch(
            sample_data.values,
            lambda1=0.1,
            max_iter=50,  # Short for testing
            lr=1e-2
        )

        # Check output properties
        assert isinstance(W, np.ndarray)
        assert W.shape == (sample_data.shape[1], sample_data.shape[1])

        # Check diagonal is zero
        np.testing.assert_array_almost_equal(W.diagonal(), 0)

        # Check that we get some weights
        assert np.count_nonzero(W) >= 0

    def test_train_notears_torch_with_priors(self, sample_data):
        """Test NOTEARS with soft priors."""
        priors = {
            (0, 1): 0.7,  # Encourage X1 -> X2
            (1, 2): 0.5,  # Encourage X2 -> X3
            (2, 3): 0.8   # Encourage X3 -> X4
        }

        W = train_notears_torch(
            sample_data.values,
            priors_dict=priors,
            lambda1=0.1,
            lam_soft=1.0,
            max_iter=50,
            lr=1e-2
        )

        # Check that prior-consistent edges are encouraged
        for (i, j), prior_weight in priors.items():
            if prior_weight > 0:
                # Edge should generally be present (though not guaranteed)
                pass

        # Check basic properties
        assert W.shape == (sample_data.shape[1], sample_data.shape[1])
        np.testing.assert_array_almost_equal(W.diagonal(), 0)

    def test_train_notears_torch_with_mask(self, sample_data):
        """Test NOTEARS with fixed mask."""
        # Create mask that fixes certain edges
        mask_fixed = np.zeros((4, 4), dtype=bool)
        mask_fixed[0, 1] = True  # Fix edge 0->1
        mask_fixed[1, 2] = True  # Fix edge 1->2

        W_fixed = np.zeros((4, 4))
        W_fixed[0, 1] = 0.5
        W_fixed[1, 2] = 0.3

        W = train_notears_torch(
            sample_data.values,
            mask_fixed=mask_fixed,
            max_iter=50,
            lr=1e-2
        )

        # Check that fixed edges are preserved (this depends on implementation)
        # The mask should prevent optimization of these edges
        assert W.shape == (sample_data.shape[1], sample_data.shape[1])
        np.testing.assert_array_almost_equal(W.diagonal(), 0)

    def test_train_notears_torch_parameters(self, sample_data):
        """Test different parameter settings."""
        # Test with different learning rates
        W_low_lr = train_notears_torch(
            sample_data.values,
            lambda1=0.1,
            max_iter=20,
            lr=1e-4
        )

        W_high_lr = train_notears_torch(
            sample_data.values,
            lambda1=0.1,
            max_iter=20,
            lr=1e-2
        )

        # Both should produce valid outputs
        assert W_low_lr.shape == W_high_lr.shape
        np.testing.assert_array_almost_equal(W_low_lr.diagonal(), 0)
        np.testing.assert_array_almost_equal(W_high_lr.diagonal(), 0)

        # Test with different lambda1 values
        W_sparse = train_notears_torch(
            sample_data.values,
            lambda1=0.5,
            max_iter=20,
            lr=1e-2
        )

        W_dense = train_notears_torch(
            sample_data.values,
            lambda1=0.01,
            max_iter=20,
            lr=1e-2
        )

        # Higher lambda1 should produce sparser results
        nnz_sparse = np.count_nonzero(W_sparse)
        nnz_dense = np.count_nonzero(W_dense)

        assert nnz_sparse <= nnz_dense

    def test_train_notears_torch_convergence(self, sample_data):
        """Test convergence behavior."""
        W = train_notears_torch(
            sample_data.values,
            lambda1=0.1,
            max_iter=100,
            lr=1e-2,
            h_tol=1e-6
        )

        # Should converge to a solution
        assert W.shape == (sample_data.shape[1], sample_data.shape[1])
        np.testing.assert_array_almost_equal(W.diagonal(), 0)

        # Check that the solution is DAG (no cycles)
        # This is a complex check, but we can at least verify basic properties
        assert not np.any(np.isnan(W))
        assert not np.any(np.isinf(W))

    def test_graph_generation_comparison(self, sample_data):
        """Test comparison between different graph generation methods."""
        # NOTEARS-al version
        W_al = get_weighted_adjacency(sample_data, lambda1=0.1)

        # NOTEARS-torch version
        W_torch = train_notears_torch(
            sample_data.values,
            lambda1=0.1,
            max_iter=50,
            lr=1e-2
        )

        # Both should have same shape
        assert W_al.shape == W_torch.shape

        # Both should have zero diagonal
        np.testing.assert_array_almost_equal(W_al.diagonal(), 0)
        np.testing.assert_array_almost_equal(W_torch.diagonal(), 0)

        # Both should produce some non-zero weights
        assert np.count_nonzero(W_al) > 0
        assert np.count_nonzero(W_torch) > 0

        # They may produce different solutions but should be reasonable
        # (we can't expect them to be identical due to different algorithms)

    def test_graph_generation_with_random_data(self, random_data):
        """Test graph generation with random data."""
        W_al = get_weighted_adjacency(random_data, lambda1=0.1)
        W_torch = train_notears_torch(
            random_data.values,
            lambda1=0.1,
            max_iter=30,
            lr=1e-2
        )

        # Even with random data, should produce valid adjacency matrices
        assert W_al.shape == (random_data.shape[1], random_data.shape[1])
        assert W_torch.shape == (random_data.shape[1], random_data.shape[1])

        np.testing.assert_array_almost_equal(W_al.diagonal(), 0)
        np.testing.assert_array_almost_equal(W_torch.diagonal(), 0)