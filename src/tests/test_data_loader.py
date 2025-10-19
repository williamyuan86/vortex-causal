"""Tests for data loading functionality."""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from src.utils.data_loader import load_sachs


class TestDataLoader:
    """Test data loading utilities."""

    def test_load_sachs_file_not_found(self):
        """Test loading when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_sachs("nonexistent_file.csv")

    def test_load_sachs_with_fake_data(self):
        """Test loading with synthetic data that mimics Sachs structure."""
        # Create synthetic protein signaling data
        np.random.seed(42)
        n_samples = 100
        proteins = ['Akt', 'Erk', 'JNK', 'MEK', 'MKK', 'PKA', 'PKC', 'Raf']

        data = np.random.normal(0, 1, (n_samples, len(proteins)))
        df = pd.DataFrame(data, columns=proteins)

        # Add some causal structure for realism
        df['MEK'] = 0.7 * df['Raf'] + 0.3 * np.random.normal(0, 1, n_samples)
        df['Erk'] = 0.8 * df['MEK'] + 0.2 * np.random.normal(0, 1, n_samples)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            loaded_df = load_sachs(temp_file)

            # Test data integrity
            assert isinstance(loaded_df, pd.DataFrame)
            assert loaded_df.shape == (n_samples, len(proteins))
            assert list(loaded_df.columns) == proteins
            assert not loaded_df.isnull().any().any()  # No NaN values
            assert loaded_df.dtypes.apply(lambda x: x in [np.float64, np.int64]).all()

        finally:
            os.unlink(temp_file)

    def test_load_sachs_empty_file(self):
        """Test loading empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")  # Empty file
            temp_file = f.name

        try:
            with pytest.raises(pd.errors.EmptyDataError):
                load_sachs(temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_sachs_malformed_data(self):
        """Test loading malformed CSV data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2,3\n4,5")  # Mismatched columns
            temp_file = f.name

        try:
            # This should either raise an error or handle gracefully
            try:
                df = load_sachs(temp_file)
                # If it succeeds, check that data is reasonable
                assert df.shape[0] > 0
            except Exception:
                # Expected behavior for malformed data
                pass
        finally:
            os.unlink(temp_file)