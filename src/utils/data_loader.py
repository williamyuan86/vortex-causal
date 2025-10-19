# src/utils/data_loader.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_sachs(filepath):
    """
    Load Sachs protein signaling dataset.

    Args:
        filepath: Path to CSV file

    Returns:
        df: Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
        ValueError: If data format is invalid
    """
    try:
        # Check file exists
        if not pd.io.common.file_exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Load data
        df = pd.read_csv(filepath)

        # Validate data
        if df.empty:
            raise pd.errors.EmptyDataError("Data file is empty")

        if df.shape[0] < 10:
            logger.warning(f"Very small dataset: {df.shape[0]} samples")

        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in data")

        # Convert all columns to numeric if possible
        for col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    logger.warning(f"Could not convert column '{col}' to numeric")

        # Handle missing values
        if df.isnull().any().any():
            missing_count = df.isnull().sum().sum()
            logger.warning(f"Found {missing_count} missing values, filling with column means")
            df = df.fillna(df.mean())

        # Remove constant columns (no variance)
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            logger.warning(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
            df = df.drop(columns=constant_cols)

        # Standardize column names (remove special characters, convert to lowercase)
        df.columns = [str(col).strip().replace(' ', '_').replace('-', '_') for col in df.columns]

        logger.info(f"Loaded data: {df.shape[0]} samples, {df.shape[1]} features")
        return df

    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise

def validate_dataset(df, min_samples=10, min_features=2):
    """
    Validate loaded dataset.

    Args:
        df: DataFrame to validate
        min_samples: Minimum required samples
        min_features: Minimum required features

    Returns:
        is_valid: Boolean indicating validity
        issues: List of issues found
    """
    issues = []

    # Check basic properties
    if not isinstance(df, pd.DataFrame):
        issues.append("Input must be pandas DataFrame")
        return False, issues

    if df.empty:
        issues.append("DataFrame is empty")
        return False, issues

    # Check dimensions
    if df.shape[0] < min_samples:
        issues.append(f"Insufficient samples: {df.shape[0]} < {min_samples}")

    if df.shape[1] < min_features:
        issues.append(f"Insufficient features: {df.shape[1]} < {min_features}")

    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        issues.append(f"Missing values found: {missing_count}")

    # Check for constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)

    if constant_cols:
        issues.append(f"Constant columns found: {constant_cols}")

    # Check for numeric data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        issues.append("No numeric columns found")

    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.isin([np.inf, -np.inf]).any().any():
        issues.append("Infinite values found in data")

    return len(issues) == 0, issues

def create_synthetic_dataset(n_samples=1000, n_vars=5, seed=42, noise_std=0.1):
    """
    Create a synthetic causal dataset for testing.

    Args:
        n_samples: Number of samples
        n_vars: Number of variables
        seed: Random seed
        noise_std: Standard deviation of noise

    Returns:
        df: Synthetic DataFrame
        true_adjacency: True adjacency matrix
    """
    np.random.seed(seed)

    # Create simple causal structure
    true_adjacency = np.zeros((n_vars, n_vars))

    # Create some causal edges
    edges = [(0, 1, 0.7), (1, 2, 0.6), (0, 3, 0.5), (3, 4, 0.8)]
    for i, j, weight in edges:
        if i < n_vars and j < n_vars:
            true_adjacency[i, j] = weight

    # Generate data
    X = np.random.normal(0, 1, (n_samples, n_vars))

    # Apply causal structure
    for i, j, weight in edges:
        if i < n_vars and j < n_vars:
            X[:, j] += weight * X[:, i]

    # Add noise
    X += np.random.normal(0, noise_std, X.shape)

    # Create DataFrame
    column_names = [f'V{i}' for i in range(n_vars)]
    df = pd.DataFrame(X, columns=column_names)

    return df, true_adjacency