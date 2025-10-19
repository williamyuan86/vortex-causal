"""Tests for effect estimation functionality."""

import pytest
import numpy as np
import pandas as pd
from src.effect_estimation.estimator import ols_ate, ipw_ate
from src.effect_estimation.dragonnet import DragonNet, dragonnet_train, predict_dragonnet, ate_from_preds, pehe


class TestEffectEstimation:
    """Test effect estimation methods."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data with known treatment effect."""
        np.random.seed(42)
        n_samples = 1000

        # Generate confounders
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)
        X3 = np.random.binomial(1, 0.5, n_samples)

        # Generate treatment with confounding
        treatment_prob = 1 / (1 + np.exp(-(0.5*X1 - 0.3*X2 + 0.2*X3)))
        T = np.random.binomial(1, treatment_prob, n_samples)

        # Generate outcomes with treatment effect and confounding
        true_ate = 2.5
        Y0 = 1.0 + 0.8*X1 - 0.5*X2 + 0.3*X3 + np.random.normal(0, 0.5, n_samples)
        Y1 = Y0 + true_ate + 0.2*X1  # Heterogeneous effect
        Y = T * Y1 + (1 - T) * Y0

        df = pd.DataFrame({
            'X1': X1, 'X2': X2, 'X3': X3,
            'treatment': T, 'outcome': Y
        })
        return df, true_ate

    @pytest.fixture
    def simple_data(self):
        """Create simple data for basic testing."""
        np.random.seed(123)
        n = 200
        X = np.random.normal(0, 1, (n, 3))
        T = np.random.binomial(1, 0.5, n)
        Y = 2 + X[:, 0] - 0.5*X[:, 1] + 1.5*T + np.random.normal(0, 0.5, n)

        df = pd.DataFrame({
            'X1': X[:, 0], 'X2': X[:, 1], 'X3': X[:, 2],
            'treatment': T, 'outcome': Y
        })
        return df

    def test_ols_ate_basic(self, simple_data):
        """Test basic OLS ATE estimation."""
        treatment_col = 'treatment'
        outcome_col = 'outcome'
        covariates = ['X1', 'X2', 'X3']

        ate_est = ols_ate(simple_data, treatment_col, outcome_col, covariates)

        # Check that we get a reasonable estimate
        assert isinstance(ate_est, (float, np.floating))
        assert not np.isnan(ate_est)
        assert not np.isinf(ate_est)

        # Should be somewhat close to true value (1.5) within reasonable bounds
        assert 0.5 < ate_est < 3.0

    def test_ols_ate_no_covariates(self, simple_data):
        """Test OLS ATE without covariates."""
        ate_est = ols_ate(simple_data, 'treatment', 'outcome', [])

        assert isinstance(ate_est, (float, np.floating))
        assert not np.isnan(ate_est)
        # Should be less accurate without covariates

    def test_ols_ate_invalid_columns(self, simple_data):
        """Test OLS ATE with invalid column names."""
        with pytest.raises(KeyError):
            ols_ate(simple_data, 'nonexistent_treatment', 'outcome', ['X1'])

        with pytest.raises(KeyError):
            ols_ate(simple_data, 'treatment', 'nonexistent_outcome', ['X1'])

        with pytest.raises(KeyError):
            ols_ate(simple_data, 'treatment', 'outcome', ['nonexistent_cov'])

    def test_ols_ate_constant_treatment(self, simple_data):
        """Test OLS ATE with constant treatment."""
        constant_data = simple_data.copy()
        constant_data['treatment'] = 1  # All treated

        ate_est = ols_ate(constant_data, 'treatment', 'outcome', ['X1', 'X2', 'X3'])
        # Should handle gracefully (might be 0 or undefined)

    def test_ipw_ate_basic(self, simple_data):
        """Test basic IPW ATE estimation."""
        treatment_col = 'treatment'
        outcome_col = 'outcome'
        covariates = ['X1', 'X2', 'X3']

        ate_est, pscores = ipw_ate(simple_data, treatment_col, outcome_col, covariates)

        # Check ATE estimate
        assert isinstance(ate_est, (float, np.floating))
        assert not np.isnan(ate_est)
        assert not np.isinf(ate_est)

        # Check propensity scores
        assert isinstance(pscores, np.ndarray)
        assert len(pscores) == len(simple_data)
        assert np.all(pscores >= 0)
        assert np.all(pscores <= 1)
        assert np.all((pscores > 0) & (pscores < 1))  # No extreme probabilities

        # Should be somewhat reasonable
        assert 0.0 < ate_est < 4.0

    def test_ipw_ate_no_covariates(self, simple_data):
        """Test IPW ATE without covariates."""
        ate_est, pscores = ipw_ate(simple_data, 'treatment', 'outcome', [])

        assert isinstance(ate_est, (float, np.floating))
        assert isinstance(pscores, np.ndarray)
        # Without covariates, propensity scores should be constant
        assert np.allclose(pscores, pscores[0])

    def test_dragonnet_basic(self, simple_data):
        """Test basic DragonNet functionality."""
        # Prepare data
        covariates = ['X1', 'X2', 'X3']
        X = simple_data[covariates].values.astype(np.float32)
        T = simple_data['treatment'].values.astype(np.int64)
        Y = simple_data['outcome'].values.astype(np.float32)

        # Create model
        model = DragonNet(X.shape[1])

        # Check model structure
        assert hasattr(model, 'forward')
        assert not hasattr(model, 'fit')  # PyTorch models use training functions

        # Test forward pass
        import torch
        X_tensor = torch.FloatTensor(X[:5])  # Small batch
        with torch.no_grad():
            p, mu0, mu1 = model(X_tensor)
        # Check shapes of individual outputs
        assert p.shape == (5,)  # propensity scores
        assert mu0.shape == (5,)  # potential outcome Y0
        assert mu1.shape == (5,)  # potential outcome Y1

        # Test combined output as tensor
        combined_output = torch.stack([p, mu0, mu1], dim=1)
        assert combined_output.shape == (5, 3)  # [propensity, mu0, mu1]

    def test_dragonnet_training(self, simple_data):
        """Test DragonNet training."""
        covariates = ['X1', 'X2', 'X3']
        X = simple_data[covariates].values.astype(np.float32)
        T = simple_data['treatment'].values.astype(np.int64)
        Y = simple_data['outcome'].values.astype(np.float32)

        model = DragonNet(X.shape[1])

        # Train for a few epochs
        dragonnet_train(model, X, T, Y, epochs=10, batch_size=32, lr=1e-3, verbose=False)

        # Check that model training completed without errors
        assert True  # If we get here, training didn't crash

    def test_dragonnet_prediction(self, simple_data):
        """Test DragonNet prediction."""
        covariates = ['X1', 'X2', 'X3']
        X = simple_data[covariates].values.astype(np.float32)
        T = simple_data['treatment'].values.astype(np.int64)
        Y = simple_data['outcome'].values.astype(np.float32)

        model = DragonNet(X.shape[1])
        dragonnet_train(model, X, T, Y, epochs=20, batch_size=32, lr=1e-3, verbose=False)

        # Test predictions
        p_pred, mu0_pred, mu1_pred = predict_dragonnet(model, X)

        # Check output shapes and properties
        assert p_pred.shape == (len(X),)
        assert mu0_pred.shape == (len(X),)
        assert mu1_pred.shape == (len(X),)

        # Check propensity score properties
        assert np.all(p_pred >= 0)
        assert np.all(p_pred <= 1)

        # Check outcome predictions are reasonable
        assert not np.any(np.isnan(mu0_pred))
        assert not np.any(np.isnan(mu1_pred))
        assert not np.any(np.isinf(mu0_pred))
        assert not np.any(np.isinf(mu1_pred))

    def test_dragonnet_ate_estimation(self, simple_data):
        """Test DragonNet ATE estimation."""
        covariates = ['X1', 'X2', 'X3']
        X = simple_data[covariates].values.astype(np.float32)
        T = simple_data['treatment'].values.astype(np.int64)
        Y = simple_data['outcome'].values.astype(np.float32)

        model = DragonNet(X.shape[1])
        dragonnet_train(model, X, T, Y, epochs=30, batch_size=32, lr=1e-3, verbose=False)

        p_pred, mu0_pred, mu1_pred = predict_dragonnet(model, X)
        ate_est = ate_from_preds(mu0_pred, mu1_pred)

        # Check ATE estimate
        assert isinstance(ate_est, (float, np.floating))
        assert not np.isnan(ate_est)
        assert not np.isinf(ate_est)

        # Should be somewhat close to true value (1.5)
        assert 0.5 < ate_est < 3.0

    def test_dragonnet_pehe(self, synthetic_data):
        """Test DragonNet PEHE calculation with known potential outcomes."""
        df, true_ate = synthetic_data

        # Prepare data
        covariates = ['X1', 'X2', 'X3']
        X = df[covariates].values.astype(np.float32)
        T = df['treatment'].values.astype(np.int64)

        # Create true potential outcomes for testing
        X1, X2, X3 = X[:, 0], X[:, 1], X[:, 2]
        Y0_true = 1.0 + 0.8*X1 - 0.5*X2 + 0.3*X3
        Y1_true = Y0_true + true_ate + 0.2*X1

        # Train model
        model = DragonNet(X.shape[1])
        Y_obs = df['outcome'].values.astype(np.float32)
        dragonnet_train(model, X, T, Y_obs, epochs=50, batch_size=64, lr=1e-3, verbose=False)

        # Get predictions
        _, mu0_pred, mu1_pred = predict_dragonnet(model, X)

        # Calculate PEHE
        pehe_val = pehe(mu0_pred, mu1_pred, Y0_true, Y1_true)

        # Check PEHE properties
        assert isinstance(pehe_val, (float, np.floating))
        assert not np.isnan(pehe_val)
        assert not np.isinf(pehe_val)
        assert pehe_val >= 0  # PEHE is always non-negative

        # PEHE should be reasonably low for a good model
        # (but this depends on data difficulty and model performance)
        assert pehe_val < 10.0  # Very loose upper bound

    def test_effect_estimation_comparison(self, simple_data):
        """Test comparison between different effect estimation methods."""
        treatment_col = 'treatment'
        outcome_col = 'outcome'
        covariates = ['X1', 'X2', 'X3']

        # OLS estimate
        ate_ols = ols_ate(simple_data, treatment_col, outcome_col, covariates)

        # IPW estimate
        ate_ipw, _ = ipw_ate(simple_data, treatment_col, outcome_col, covariates)

        # DragonNet estimate
        X = simple_data[covariates].values.astype(np.float32)
        T = simple_data['treatment'].values.astype(np.int64)
        Y = simple_data['outcome'].values.astype(np.float32)

        model = DragonNet(X.shape[1])
        dragonnet_train(model, X, T, Y, epochs=30, batch_size=32, lr=1e-3, verbose=False)
        _, mu0_pred, mu1_pred = predict_dragonnet(model, X)
        ate_dragonnet = ate_from_preds(mu0_pred, mu1_pred)

        # All estimates should be reasonable
        for ate in [ate_ols, ate_ipw, ate_dragonnet]:
            assert isinstance(ate, (float, np.floating))
            assert not np.isnan(ate)
            assert not np.isinf(ate)
            assert 0.0 < ate < 5.0  # Reasonable range

    def test_effect_estimation_edge_cases(self):
        """Test effect estimation with edge cases."""
        # Very small dataset
        tiny_data = pd.DataFrame({
            'X': [1, 2, 3, 4],
            'T': [0, 1, 0, 1],
            'Y': [1, 3, 2, 4]
        })

        try:
            ate_ols = ols_ate(tiny_data, 'T', 'Y', ['X'])
            assert isinstance(ate_ols, (float, np.floating))
        except Exception:
            # Small datasets might cause issues
            pass

        # All treated or all control
        all_treated = tiny_data.copy()
        all_treated['T'] = 1

        try:
            ate_ols = ols_ate(all_treated, 'T', 'Y', ['X'])
            # Should handle gracefully
        except Exception:
            # Expected behavior
            pass

        # No variation in outcome
        constant_outcome = tiny_data.copy()
        constant_outcome['Y'] = 2

        ate_ols = ols_ate(constant_outcome, 'T', 'Y', ['X'])
        assert ate_ols == 0.0  # Should be exactly zero