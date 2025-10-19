#!/usr/bin/env python3
"""
Performance test script for Vortex-Causal system on Sachs dataset.

This script evaluates causal discovery performance using:
- Structural Hamming Distance (SHD)
- F1-score, Precision, Recall
- AUC-ROC and AUC-PR
- Additional metrics

Author: Claude Code Assistant
Date: October 19, 2025
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from src.utils.data_loader import load_sachs
from src.selector.algorithm_selector import run_selector
from src.graph_generation.notears_torch import train_notears_torch
from src.constraint_fusion.fusion import apply_hard_constraints, apply_soft_priors_loss_term
from src.effect_estimation.estimator import ols_ate, ipw_ate
from src.utils.metrics_extended import (
    structural_hamming_distance,
    compute_auc_metrics,
    compute_graph_metrics,
    compute_comprehensive_benchmark
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_ground_truth_graph():
    """
    Create a ground truth causal graph for Sachs protein signaling dataset.

    Based on literature, the known causal relationships include:
    - PKC -> Raf
    - PKC -> PIP2
    - PKC -> PIP3
    - PKA -> Raf
    - PKA -> Mek
    - Raf -> Mek
    - Mek -> Erk
    - Akt -> PIP3
    - PIP3 -> PIP2
    - Plcg -> PIP3
    - Plcg -> PIP2
    - Jnk -> c-Jun (not in dataset)
    - P38 -> c-Jun (not in dataset)

    Returns:
        W_true: Ground truth adjacency matrix (11 x 11)
    """
    proteins = ['Erk', 'Akt', 'PKA', 'Mek', 'Jnk', 'PKC', 'Raf', 'P38', 'PIP3', 'PIP2', 'Plcg']
    n = len(proteins)
    W_true = np.zeros((n, n))

    # Create index mapping
    idx = {protein: i for i, protein in enumerate(proteins)}

    # Known causal relationships (based on Sachs et al., 2005)
    edges = [
        ('PKC', 'Raf'),
        ('PKC', 'PIP2'),
        ('PKC', 'PIP3'),
        ('PKA', 'Raf'),
        ('PKA', 'Mek'),
        ('Raf', 'Mek'),
        ('Mek', 'Erk'),
        ('Akt', 'PIP3'),
        ('PIP3', 'PIP2'),
        ('Plcg', 'PIP3'),
        ('Plcg', 'PIP2'),
        ('Jnk', 'P38'),  # Less certain
    ]

    for source, target in edges:
        if source in idx and target in idx:
            W_true[idx[source], idx[target]] = 1.0

    return W_true, proteins

def run_vortex_causal_pipeline(df, config=None):
    """
    Run the complete Vortex-Causal pipeline on the dataset.

    Args:
        df: Input DataFrame
        config: Configuration dictionary

    Returns:
        results: Dictionary with all results
    """
    if config is None:
        config = {
            'selector': {'thresh_hard': 0.8, 'thresh_ambiguous': 0.3},
            'graph': {'lambda1': 0.1, 'max_iter': 100, 'lr': 1e-2},
            'ensemble': {'mock': True},
        }

    results = {}

    logger.info("Running algorithm selector...")
    try:
        selector_result = run_selector(df,
                                     thresh_hard=config['selector']['thresh_hard'],
                                     thresh_ambiguous=config['selector']['thresh_ambiguous'])
        results['selector'] = selector_result
        logger.info(f"Found {len(selector_result['hard'])} hard edges, "
                   f"{len(selector_result['ambiguous'])} ambiguous edges")
    except Exception as e:
        logger.error(f"Selector failed: {e}")
        return None

    logger.info("Training baseline NOTEARS graph...")
    try:
        W_baseline = train_notears_torch(df.values,
                                        lambda1=config['graph']['lambda1'],
                                        max_iter=config['graph']['max_iter'],
                                        lr=config['graph']['lr'])
        results['W_baseline'] = W_baseline
    except Exception as e:
        logger.error(f"Baseline training failed: {e}")
        return None

    logger.info("Training constrained NOTEARS graph...")
    try:
        # Apply hard constraints
        n_vars = df.shape[1]
        W_init = np.zeros((n_vars, n_vars))
        mask_fixed, W_fixed = apply_hard_constraints(W_init, selector_result['hard'], n_vars)

        W_constrained = train_notears_torch(df.values,
                                           lambda1=config['graph']['lambda1'],
                                           max_iter=config['graph']['max_iter'],
                                           lr=config['graph']['lr'],
                                           mask_fixed=mask_fixed)
        results['W_constrained'] = W_constrained
    except Exception as e:
        logger.error(f"Constrained training failed: {e}")
        return None

    return results

def evaluate_performance(W_true, W_est, method_name):
    """
    Evaluate performance of a causal discovery method.

    Args:
        W_true: Ground truth adjacency matrix
        W_est: Estimated adjacency matrix
        method_name: Name of the method

    Returns:
        metrics: Dictionary of performance metrics
    """
    try:
        # Compute basic graph metrics
        graph_metrics = compute_graph_metrics(W_true, W_est, threshold=0.01)

        # Compute AUC metrics
        auc_metrics = compute_auc_metrics(W_true, W_est)

        # Combine all metrics
        metrics = {**graph_metrics, **auc_metrics}
        metrics['method'] = method_name
        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed for {method_name}: {e}")
        return {'method': method_name, 'error': str(e)}

def main():
    """
    Main performance testing function.
    """
    logger.info("Starting Vortex-Causal performance test on Sachs dataset")

    # Load dataset
    logger.info("Loading Sachs dataset...")
    try:
        df = load_sachs('data/sachs_data.csv')
        logger.info(f"Loaded dataset with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Create ground truth
    logger.info("Creating ground truth causal graph...")
    W_true, proteins = create_ground_truth_graph()
    logger.info(f"Ground truth graph has {np.sum(W_true > 0)} edges")

    # Run pipeline
    logger.info("Running Vortex-Causal pipeline...")
    start_time = time.time()
    results = run_vortex_causal_pipeline(df)
    end_time = time.time()

    if results is None:
        logger.error("Pipeline failed!")
        return

    logger.info(f"Pipeline completed in {end_time - start_time:.2f} seconds")

    # Evaluate performance
    logger.info("Evaluating performance...")
    evaluation_results = []

    # Evaluate baseline method
    if 'W_baseline' in results:
        baseline_metrics = evaluate_performance(W_true, results['W_baseline'], "Baseline NOTEARS")
        evaluation_results.append(baseline_metrics)

    # Evaluate constrained method
    if 'W_constrained' in results:
        constrained_metrics = evaluate_performance(W_true, results['W_constrained'], "Constrained NOTEARS")
        evaluation_results.append(constrained_metrics)

    # Print results
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE RESULTS")
    logger.info("="*80)

    for metrics in evaluation_results:
        if 'error' in metrics:
            logger.info(f"\n{metrics['method']}: FAILED - {metrics['error']}")
            continue

        logger.info(f"\n{metrics['method']}:")
        logger.info(f"  SHD: {metrics.get('shd', 'N/A')}")
        logger.info(f"  F1-score: {metrics.get('f1_score', 'N/A'):.4f}")
        logger.info(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
        logger.info(f"  Recall: {metrics.get('recall', 'N/A'):.4f}")
        logger.info(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}")
        logger.info(f"  AUC-PR: {metrics.get('auc_pr', 'N/A'):.4f}")
        logger.info(f"  True Positives: {metrics.get('true_positives', 'N/A')}")
        logger.info(f"  False Positives: {metrics.get('false_positives', 'N/A')}")
        logger.info(f"  False Negatives: {metrics.get('false_negatives', 'N/A')}")

    # Save results
    logger.info("\nSaving results...")
    try:
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)

        # Save detailed results
        results_df = pd.DataFrame(evaluation_results)
        results_df.to_csv('results/sachs_performance_results.csv', index=False)

        # Save learned graphs
        if 'W_baseline' in results:
            np.save('results/sachs_baseline_graph.npy', results['W_baseline'])
        if 'W_constrained' in results:
            np.save('results/sachs_constrained_graph.npy', results['W_constrained'])

        # Save ground truth
        np.save('results/sachs_ground_truth.npy', W_true)

        logger.info("Results saved to 'results/' directory")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")

    logger.info("\nPerformance test completed!")

if __name__ == "__main__":
    main()