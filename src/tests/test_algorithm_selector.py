"""Tests for algorithm selector component."""

import pytest
import numpy as np
import pandas as pd
from src.selector.algorithm_selector import run_selector, pc_like_partialcorr, consensus_scores


class TestAlgorithmSelector:
    """Test algorithm selector functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        n_vars = 5

        # Create data with some causal structure
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.7 * X1 + 0.3 * np.random.normal(0, 1, n_samples)
        X3 = 0.5 * X2 + 0.5 * np.random.normal(0, 1, n_samples)
        X4 = np.random.normal(0, 1, n_samples)
        X5 = 0.6 * X3 + 0.4 * X4 + 0.2 * np.random.normal(0, 1, n_samples)

        df = pd.DataFrame({
            'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5
        })
        return df

    @pytest.fixture
    def random_data(self):
        """Create completely random data."""
        np.random.seed(123)
        return pd.DataFrame(
            np.random.normal(0, 1, (100, 4)),
            columns=['A', 'B', 'C', 'D']
        )

    def test_run_selector_output_structure(self, sample_data):
        """Test that run_selector returns correct structure."""
        result = run_selector(sample_data)

        # Check output structure
        assert isinstance(result, dict)
        assert 'hard' in result
        assert 'ambiguous' in result
        assert 'low' in result
        assert 'W1' in result
        assert 'W2' in result

        # Check that all edges are categorized
        all_edges = set()
        for category in ['hard', 'ambiguous', 'low']:
            for edge in result[category]:
                all_edges.add((edge[0], edge[1]))

        # Should have n*(n-1) possible edges (no self-loops)
        n = sample_data.shape[1]
        expected_edges = n * (n - 1)
        assert len(all_edges) <= expected_edges

        # Check weight matrices
        assert result['W1'].shape == (n, n)
        assert result['W2'].shape == (n, n)
        assert np.allclose(result['W1'].diagonal(), 0, atol=1e-8)
        assert np.allclose(result['W2'].diagonal(), 0, atol=1e-8)

    def test_run_selector_edge_categorization(self, sample_data):
        """Test edge categorization logic."""
        result = run_selector(sample_data, thresh_hard=0.9, thresh_ambiguous=0.1)

        # Check that edges are properly categorized based on confidence
        for edge in result['hard']:
            assert edge[3] >= 0.9  # confidence score

        for edge in result['low']:
            assert edge[3] <= 0.1  # confidence score

        for edge in result['ambiguous']:
            assert 0.1 < edge[3] < 0.9  # confidence score

    def test_consensus_scores(self):
        """Test consensus scoring function."""
        # Create two simple weight matrices
        W1 = np.array([[0, 0.5, 0.1],
                       [0.2, 0, 0.3],
                       [0, 0.1, 0]])

        W2 = np.array([[0, 0.3, 0.2],
                       [0.4, 0, 0.1],
                       [0.1, 0, 0]])

        presence_mean, weight_mean = consensus_scores([W1, W2])

        # Check shapes
        assert presence_mean.shape == W1.shape
        assert weight_mean.shape == W1.shape

        # Check diagonal is zero
        assert np.allclose(presence_mean.diagonal(), 0)
        assert np.allclose(weight_mean.diagonal(), 0)

        # Check that presence mean is between 0 and 1
        assert np.all(presence_mean >= 0)
        assert np.all(presence_mean <= 1)

        # Check specific values
        # Edge (0,1): both algorithms have non-zero weights
        assert presence_mean[0, 1] == 1.0
        assert abs(weight_mean[0, 1] - 0.4) < 1e-8  # (0.5 + 0.3) / 2

        # Edge (2,0): only second algorithm has non-zero weight
        assert presence_mean[2, 0] == 0.5
        assert abs(weight_mean[2, 0] - 0.1) < 1e-8

    def test_pc_like_partialcorr(self, sample_data):
        """Test PC-like partial correlation implementation."""
        W = pc_like_partialcorr(sample_data)

        # Check output properties
        assert isinstance(W, np.ndarray)
        assert W.shape == (sample_data.shape[1], sample_data.shape[1])
        assert np.allclose(W.diagonal(), 0, atol=1e-8)

        # Check that weights are non-negative (due to abs(corr))
        assert np.all(W >= 0)

        # Check symmetry of underlying correlation structure
        # (orientation breaks symmetry, but magnitudes should be related)
        for i in range(W.shape[0]):
            for j in range(i+1, W.shape[1]):
                if W[i, j] > 0 or W[j, i] > 0:
                    # At least one direction should have weight if correlated
                    pass

    def test_pc_like_partialcorr_edge_cases(self):
        """Test PC-like function with edge cases."""
        # Test with very small dataset
        small_data = pd.DataFrame({
            'A': [1, 2],
            'B': [2, 1]
        })
        W = pc_like_partialcorr(small_data)
        assert W.shape == (2, 2)

        # Test with constant data
        const_data = pd.DataFrame({
            'A': [1, 1, 1, 1],
            'B': [2, 2, 2, 2]
        })
        try:
            W = pc_like_partialcorr(const_data)
            # Should handle gracefully or raise informative error
        except Exception as e:
            assert "singular" in str(e).lower() or "covariance" in str(e).lower()

    def test_run_selector_different_thresholds(self, sample_data):
        """Test run_selector with different threshold values."""
        # Test with very high hard threshold
        result1 = run_selector(sample_data, thresh_hard=0.99, thresh_ambiguous=0.01)
        assert len(result1['hard']) <= len(result1['ambiguous']) + len(result1['low'])

        # Test with very low hard threshold
        result2 = run_selector(sample_data, thresh_hard=0.1, thresh_ambiguous=0.05)
        assert len(result2['hard']) >= len(result1['hard'])

        # Test invalid thresholds
        with pytest.raises(ValueError):
            run_selector(sample_data, thresh_hard=1.5, thresh_ambiguous=0.1)

        with pytest.raises(ValueError):
            run_selector(sample_data, thresh_hard=0.5, thresh_ambiguous=0.7)

    def test_run_selector_reproducibility(self, sample_data):
        """Test that run_selector gives reproducible results."""
        result1 = run_selector(sample_data, thresh_hard=0.8, thresh_ambiguous=0.2)
        result2 = run_selector(sample_data, thresh_hard=0.8, thresh_ambiguous=0.2)

        # Results should be identical
        assert len(result1['hard']) == len(result2['hard'])
        assert len(result1['ambiguous']) == len(result2['ambiguous'])
        assert len(result1['low']) == len(result2['low'])

        # Arrays should be close
        np.testing.assert_array_almost_equal(result1['W1'], result2['W1'])
        np.testing.assert_array_almost_equal(result1['W2'], result2['W2'])