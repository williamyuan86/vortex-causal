#!/usr/bin/env python3
"""
Comprehensive performance test for Vortex-Causal system on Sachs dataset.

This script tests multiple configurations and provides detailed analysis.
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
from src.constraint_fusion.fusion import apply_hard_constraints
from src.utils.metrics_extended import compute_graph_metrics, compute_auc_metrics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_ground_truth_graph():
    """
    Create a ground truth causal graph for Sachs protein signaling dataset.
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
        ('Jnk', 'P38'),
    ]

    for source, target in edges:
        if source in idx and target in idx:
            W_true[idx[source], idx[target]] = 1.0

    return W_true, proteins

def test_configuration(df, W_true, config_name, config):
    """
    Test a specific configuration.

    Args:
        df: Input DataFrame
        W_true: Ground truth adjacency matrix
        config_name: Name of the configuration
        config: Configuration dictionary

    Returns:
        results: Dictionary with results
    """
    logger.info(f"\nTesting configuration: {config_name}")
    results = {'config_name': config_name, 'config': config}

    # Run selector
    try:
        selector_result = run_selector(df,
                                     thresh_hard=config['selector']['thresh_hard'],
                                     thresh_ambiguous=config['selector']['thresh_ambiguous'])
        results['selector'] = selector_result
        logger.info(f"  Found {len(selector_result['hard'])} hard edges, {len(selector_result['ambiguous'])} ambiguous edges")
    except Exception as e:
        logger.error(f"  Selector failed: {e}")
        return None

    # Test baseline method
    try:
        start_time = time.time()
        W_baseline = train_notears_torch(df.values,
                                        lambda1=config['graph']['lambda1'],
                                        max_iter=config['graph']['max_iter'],
                                        lr=config['graph']['lr'])
        baseline_time = time.time() - start_time

        # Evaluate baseline
        baseline_metrics = evaluate_graph_performance(W_true, W_baseline, "Baseline")
        baseline_metrics['runtime'] = baseline_time
        results['baseline'] = baseline_metrics

        logger.info(f"  Baseline - SHD: {baseline_metrics.get('shd', 'N/A')}, "
                   f"F1: {baseline_metrics.get('f1_score', 0):.3f}, "
                   f"AUC-ROC: {baseline_metrics.get('auc_roc', 0):.3f}")

    except Exception as e:
        logger.error(f"  Baseline training failed: {e}")
        results['baseline'] = {'error': str(e)}

    # Test constrained method
    try:
        start_time = time.time()
        # Apply hard constraints
        n_vars = df.shape[1]
        W_init = np.zeros((n_vars, n_vars))
        mask_fixed, W_fixed = apply_hard_constraints(W_init, selector_result['hard'], n_vars)

        W_constrained = train_notears_torch(df.values,
                                           lambda1=config['graph']['lambda1'],
                                           max_iter=config['graph']['max_iter'],
                                           lr=config['graph']['lr'],
                                           mask_fixed=mask_fixed)
        constrained_time = time.time() - start_time

        # Evaluate constrained
        constrained_metrics = evaluate_graph_performance(W_true, W_constrained, "Constrained")
        constrained_metrics['runtime'] = constrained_time
        results['constrained'] = constrained_metrics

        logger.info(f"  Constrained - SHD: {constrained_metrics.get('shd', 'N/A')}, "
                   f"F1: {constrained_metrics.get('f1_score', 0):.3f}, "
                   f"AUC-ROC: {constrained_metrics.get('auc_roc', 0):.3f}")

    except Exception as e:
        logger.error(f"  Constrained training failed: {e}")
        results['constrained'] = {'error': str(e)}

    return results

def evaluate_graph_performance(W_true, W_est, method_name):
    """
    Evaluate performance of a method.
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

def create_benchmark_report(all_results):
    """
    Create a comprehensive benchmark report.
    """
    report_lines = []
    report_lines.append("# Vortex-Causal Performance Benchmark Report")
    report_lines.append("## Dataset: Sachs Protein Signaling")
    report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Summary table
    report_lines.append("## Summary Results")
    report_lines.append("")
    report_lines.append("| Configuration | Method | SHD | F1-Score | Precision | Recall | AUC-ROC | AUC-PR | Runtime (s) |")
    report_lines.append("|---------------|--------|-----|----------|----------|--------|---------|--------|--------------|")

    for result in all_results:
        if result is None:
            continue

        config_name = result['config_name']

        for method in ['baseline', 'constrained']:
            if method in result and 'error' not in result[method]:
                metrics = result[method]
                runtime = metrics.get('runtime', 0)
                report_lines.append(f"| {config_name} | {method.capitalize()} | "
                                  f"{metrics.get('shd', 'N/A')} | "
                                  f"{metrics.get('f1_score', 0):.4f} | "
                                  f"{metrics.get('precision', 0):.4f} | "
                                  f"{metrics.get('recall', 0):.4f} | "
                                  f"{metrics.get('auc_roc', 0):.4f} | "
                                  f"{metrics.get('auc_pr', 0):.4f} | "
                                  f"{runtime:.3f} |")

    report_lines.append("")

    # Detailed analysis
    report_lines.append("## Detailed Analysis")
    report_lines.append("")

    # Find best performing configuration
    best_f1 = 0
    best_config = None
    best_method = None

    for result in all_results:
        if result is None:
            continue
        for method in ['baseline', 'constrained']:
            if method in result and 'error' not in result[method]:
                f1 = result[method].get('f1_score', 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_config = result['config_name']
                    best_method = method

    if best_config:
        report_lines.append(f"**Best Performing Configuration:** {best_config} - {best_method.capitalize()}")
        report_lines.append(f"**Best F1-Score:** {best_f1:.4f}")
        report_lines.append("")

    # Comparison of baseline vs constrained
    report_lines.append("### Baseline vs Constrained Comparison")
    report_lines.append("")

    baseline_shds = []
    constrained_shds = []
    baseline_f1s = []
    constrained_f1s = []

    for result in all_results:
        if result is None:
            continue

        if 'baseline' in result and 'error' not in result['baseline']:
            baseline_shds.append(result['baseline'].get('shd', 0))
            baseline_f1s.append(result['baseline'].get('f1_score', 0))

        if 'constrained' in result and 'error' not in result['constrained']:
            constrained_shds.append(result['constrained'].get('shd', 0))
            constrained_f1s.append(result['constrained'].get('f1_score', 0))

    if baseline_shds and constrained_shds:
        avg_baseline_shd = np.mean(baseline_shds)
        avg_constrained_shd = np.mean(constrained_shds)
        avg_baseline_f1 = np.mean(baseline_f1s)
        avg_constrained_f1 = np.mean(constrained_f1s)

        report_lines.append(f"**Average SHD:**")
        report_lines.append(f"- Baseline: {avg_baseline_shd:.2f}")
        report_lines.append(f"- Constrained: {avg_constrained_shd:.2f}")
        report_lines.append("")
        report_lines.append(f"**Average F1-Score:**")
        report_lines.append(f"- Baseline: {avg_baseline_f1:.4f}")
        report_lines.append(f"- Constrained: {avg_constrained_f1:.4f}")
        report_lines.append("")

        if avg_constrained_f1 > avg_baseline_f1:
            report_lines.append("✅ **Constrained method outperforms baseline on average**")
        else:
            report_lines.append("❌ **Baseline method outperforms constrained on average**")
        report_lines.append("")

    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("")

    if best_f1 < 0.1:
        report_lines.append("⚠️ **Warning:** Low performance detected. Consider:")
        report_lines.append("- Reducing regularization strength (lambda1)")
        report_lines.append("- Increasing max_iter for better convergence")
        report_lines.append("- Pre-processing data (normalization, outlier removal)")
        report_lines.append("- Using different threshold settings")
    else:
        report_lines.append("✅ **Good performance achieved!**")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Generated by Vortex-Causal Performance Testing Suite*")

    return "\n".join(report_lines)

def main():
    """
    Main function to run comprehensive performance tests.
    """
    logger.info("Starting comprehensive Vortex-Causal performance test on Sachs dataset")

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

    # Define test configurations
    configurations = [
        ("Default", {
            'selector': {'thresh_hard': 0.8, 'thresh_ambiguous': 0.3},
            'graph': {'lambda1': 0.01, 'max_iter': 100, 'lr': 1e-2}
        }),
        ("Low Regularization", {
            'selector': {'thresh_hard': 0.8, 'thresh_ambiguous': 0.3},
            'graph': {'lambda1': 0.001, 'max_iter': 100, 'lr': 1e-2}
        }),
        ("Very Low Regularization", {
            'selector': {'thresh_hard': 0.8, 'thresh_ambiguous': 0.3},
            'graph': {'lambda1': 0.0001, 'max_iter': 100, 'lr': 1e-2}
        }),
        ("High Iterations", {
            'selector': {'thresh_hard': 0.8, 'thresh_ambiguous': 0.3},
            'graph': {'lambda1': 0.01, 'max_iter': 500, 'lr': 1e-2}
        }),
        ("Relaxed Thresholds", {
            'selector': {'thresh_hard': 0.6, 'thresh_ambiguous': 0.2},
            'graph': {'lambda1': 0.01, 'max_iter': 100, 'lr': 1e-2}
        }),
    ]

    # Run tests
    all_results = []
    total_start_time = time.time()

    for config_name, config in configurations:
        result = test_configuration(df, W_true, config_name, config)
        all_results.append(result)

    total_time = time.time() - total_start_time
    logger.info(f"\nAll tests completed in {total_time:.2f} seconds")

    # Create and save report
    logger.info("\nGenerating comprehensive report...")
    report = create_benchmark_report(all_results)

    try:
        # Create results directory
        os.makedirs('results', exist_ok=True)

        # Save report
        with open('results/comprehensive_performance_report.md', 'w') as f:
            f.write(report)

        # Save detailed results
        detailed_results = []
        for result in all_results:
            if result:
                config_name = result['config_name']
                for method in ['baseline', 'constrained']:
                    if method in result and 'error' not in result[method]:
                        metrics = result[method].copy()
                        metrics['config_name'] = config_name
                        metrics['method'] = method
                        detailed_results.append(metrics)

        if detailed_results:
            results_df = pd.DataFrame(detailed_results)
            results_df.to_csv('results/detailed_performance_results.csv', index=False)

        logger.info("Reports saved to 'results/' directory")

    except Exception as e:
        logger.error(f"Failed to save reports: {e}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*80)
    logger.info(report)
    logger.info("="*80)

if __name__ == "__main__":
    main()