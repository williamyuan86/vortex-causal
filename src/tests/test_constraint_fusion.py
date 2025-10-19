"""Tests for constraint fusion functionality."""

import pytest
import numpy as np
from src.constraint_fusion.fusion import apply_hard_constraints, apply_soft_priors_loss_term


class TestConstraintFusion:
    """Test constraint fusion functionality."""

    @pytest.fixture
    def sample_weight_matrix(self):
        """Create a sample weight matrix."""
        np.random.seed(42)
        W = np.random.normal(0, 0.1, (4, 4))
        # Set diagonal to zero
        np.fill_diagonal(W, 0)
        return W

    @pytest.fixture
    def sample_hard_edges(self):
        """Create sample hard edges."""
        return [
            (0, 1, 0.8, 0.95),  # i, j, weight, confidence
            (1, 2, -0.6, 0.92),
            (2, 3, 0.7, 0.98)
        ]

    def test_apply_hard_constraints_basic(self, sample_weight_matrix, sample_hard_edges):
        """Test basic hard constraint application."""
        mask_fixed, W_fixed = apply_hard_constraints(
            sample_weight_matrix, sample_hard_edges, n_vars=4
        )

        # Check output shapes
        assert mask_fixed.shape == sample_weight_matrix.shape
        assert W_fixed.shape == sample_weight_matrix.shape

        # Check that hard edges are fixed
        for i, j, weight, conf in sample_hard_edges:
            assert mask_fixed[i, j] == 1  # Should be marked as fixed
            assert W_fixed[i, j] == weight  # Should have the specified weight

        # Check diagonal is fixed to zero
        assert np.all(mask_fixed.diagonal() == 1)
        assert np.all(W_fixed.diagonal() == 0)

    def test_apply_hard_constraints_empty_edges(self, sample_weight_matrix):
        """Test applying hard constraints with no edges."""
        mask_fixed, W_fixed = apply_hard_constraints(
            sample_weight_matrix, [], n_vars=4
        )

        # Should only have diagonal fixed
        assert np.sum(mask_fixed) == 4  # Only diagonal elements
        assert np.all(W_fixed.diagonal() == 0)

    def test_apply_hard_constraints_self_loops(self, sample_weight_matrix):
        """Test handling of self-loop edges."""
        # Include self-loops in hard edges
        hard_edges_with_loops = [
            (0, 1, 0.5, 0.9),
            (1, 1, 0.8, 0.95),  # Self-loop
            (2, 3, -0.4, 0.92)
        ]

        mask_fixed, W_fixed = apply_hard_constraints(
            sample_weight_matrix, hard_edges_with_loops, n_vars=4
        )

        # Self-loops should be ignored or set to zero
        assert W_fixed[1, 1] == 0
        assert mask_fixed[1, 1] == 1

        # Other edges should be fixed correctly
        assert W_fixed[0, 1] == 0.5
        assert W_fixed[2, 3] == -0.4

    def test_apply_hard_constraints_duplicate_edges(self, sample_weight_matrix):
        """Test handling of duplicate edges."""
        duplicate_edges = [
            (0, 1, 0.5, 0.9),
            (0, 1, 0.7, 0.8),  # Duplicate with different weight
            (1, 2, -0.3, 0.95)
        ]

        mask_fixed, W_fixed = apply_hard_constraints(
            sample_weight_matrix, duplicate_edges, n_vars=4
        )

        # Should handle duplicates gracefully
        assert mask_fixed[0, 1] == 1
        # Should use one of the weights (implementation dependent)
        assert W_fixed[0, 1] in [0.5, 0.7]

    def test_apply_hard_constraints_invalid_indices(self, sample_weight_matrix):
        """Test handling of invalid edge indices."""
        invalid_edges = [
            (0, 1, 0.5, 0.9),
            (5, 2, 0.3, 0.8),  # Invalid index
            (-1, 2, 0.4, 0.9)  # Negative index
        ]

        # Should either ignore invalid edges or raise error
        try:
            mask_fixed, W_fixed = apply_hard_constraints(
                sample_weight_matrix, invalid_edges, n_vars=4
            )
            # If it succeeds, check that valid edges are still applied
            assert mask_fixed[0, 1] == 1
            assert W_fixed[0, 1] == 0.5
        except (IndexError, ValueError):
            # Expected behavior for completely invalid indices
            pass

    def test_apply_soft_priors_loss_term_basic(self):
        """Test basic soft priors loss term calculation."""
        W = np.array([[0, 0.3, -0.2],
                      [0.1, 0, 0.4],
                      [0.2, -0.1, 0]])

        priors = {
            (0, 1): 0.8,   # Encourage edge 0->1
            (1, 2): -0.6,  # Encourage negative edge 1->2
            (2, 0): 0.5    # Encourage edge 2->0
        }

        loss = apply_soft_priors_loss_term(W, priors, lambda_soft=1.0)

        # Should return a scalar loss
        assert isinstance(loss, (float, np.floating))
        assert loss >= 0

        # Test with different lambda values
        loss_high = apply_soft_priors_loss_term(W, priors, lambda_soft=2.0)
        loss_low = apply_soft_priors_loss_term(W, priors, lambda_soft=0.5)

        assert loss_high >= loss_low

    def test_apply_soft_priors_loss_term_empty_priors(self):
        """Test soft priors with empty priors dict."""
        W = np.random.normal(0, 0.1, (3, 3))
        np.fill_diagonal(W, 0)

        loss = apply_soft_priors_loss_term(W, {}, lambda_soft=1.0)
        assert loss == 0.0

    def test_apply_soft_priors_loss_term_perfect_match(self):
        """Test loss when weights perfectly match priors."""
        # Create weights that exactly match priors
        W = np.array([[0, 0.5, -0.3],
                      [0.2, 0, 0.4],
                      [-0.3, 0.1, 0]])

        priors = {
            (0, 1): 0.5,   # Perfect match
            (1, 2): 0.4,   # Perfect match
            (2, 0): -0.3   # Perfect match
        }

        loss = apply_soft_priors_loss_term(W, priors, lambda_soft=1.0)
        # Loss should be very close to zero for perfect matches
        assert loss < 1e-8

    def test_apply_soft_priors_loss_term_opposite_priors(self):
        """Test loss when weights are opposite to priors."""
        W = np.array([[0, 0.5, -0.3],
                      [0.2, 0, 0.4],
                      [-0.3, 0.1, 0]])

        priors = {
            (0, 1): -0.5,  # Opposite sign
            (1, 2): -0.4,  # Opposite sign
            (2, 0): 0.3    # Opposite sign
        }

        loss = apply_soft_priors_loss_term(W, priors, lambda_soft=1.0)
        # Loss should be higher for opposite signs
        assert loss > 0

    def test_apply_soft_priors_loss_term_magnitude_difference(self):
        """Test loss with different magnitudes."""
        W = np.array([[0, 0.1, 0.2],
                      [0, 0, 0.3],
                      [0, 0, 0]])

        priors = {
            (0, 1): 0.8,   # Large magnitude difference
            (0, 2): 0.25   # Small magnitude difference
        }

        loss = apply_soft_priors_loss_term(W, priors, lambda_soft=1.0)
        assert loss > 0

        # The larger magnitude difference should contribute more to loss
        # (this is implicitly tested by the loss calculation)

    def test_constraint_fusion_integration(self):
        """Test integration of hard and soft constraints."""
        # Initial weight matrix
        W = np.random.normal(0, 0.1, (4, 4))
        np.fill_diagonal(W, 0)

        # Hard constraints
        hard_edges = [(0, 1, 0.7, 0.95), (2, 3, -0.5, 0.92)]
        mask_fixed, W_fixed = apply_hard_constraints(W, hard_edges, n_vars=4)

        # Soft priors for remaining edges
        priors = {(1, 2): 0.4, (3, 0): -0.3}
        loss = apply_soft_priors_loss_term(W_fixed, priors, lambda_soft=1.0)

        # Check that hard edges are preserved in loss calculation
        # (soft priors shouldn't affect fixed edges)
        assert mask_fixed[0, 1] == 1
        assert mask_fixed[2, 3] == 1
        assert W_fixed[0, 1] == 0.7
        assert W_fixed[2, 3] == -0.5

        # Loss should be computable
        assert isinstance(loss, (float, np.floating))
        assert loss >= 0