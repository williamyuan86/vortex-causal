#!/usr/bin/env python3
# scripts/run_auto_eval.py
"""
Auto-Evaluation Script for Vortex-Causal

This script runs the complete 10-iteration auto-evaluation loop for Vortex-Causal,
implementing the autonomous optimization system described in CLAUDE.md.

Usage:
    python scripts/run_auto_eval.py [--max-iterations 10] [--output-dir output] [--dataset synthetic]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.autoeval.controller import AutoEvalController, run_auto_eval
from src.utils.data_loader import load_sachs, create_synthetic_dataset
from src.utils.metrics_extended import compute_comprehensive_benchmark

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_eval.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("auto_eval_script")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AutoEval optimization for Vortex-Causal"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Maximum number of optimization iterations (default: 10)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Output directory for results (default: output)"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["synthetic", "sachs"], default="synthetic",
        help="Dataset to use for optimization (default: synthetic)"
    )
    parser.add_argument(
        "--ground-truth", type=str, default=None,
        help="Path to ground truth adjacency matrix (.npy file)"
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Run baseline experiment only (no optimization)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def load_dataset(dataset_name: str, ground_truth_path: str = None):
    """
    Load the specified dataset.

    Args:
        dataset_name: Name of dataset ("synthetic" or "sachs")
        ground_truth_path: Path to ground truth file

    Returns:
        Tuple of (df, ground_truth_adj)
    """
    if dataset_name == "synthetic":
        logger.info("Creating synthetic dataset with known ground truth")
        df, true_adj = create_synthetic_dataset(n_samples=1000, n_vars=8, seed=42)
        return df, true_adj
    elif dataset_name == "sachs":
        logger.info("Loading Sachs dataset")
        df = load_sachs("data/sachs.csv")

        if ground_truth_path and os.path.exists(ground_truth_path):
            true_adj = np.load(ground_truth_path)
            logger.info(f"Loaded ground truth from {ground_truth_path}")
        else:
            logger.warning("No ground truth available for Sachs dataset. Using synthetic for evaluation.")
            _, true_adj = create_synthetic_dataset(n_samples=1000, n_vars=df.shape[1], seed=42)

        return df, true_adj
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_baseline_experiment(df, ground_truth_adj, output_dir):
    """
    Run baseline experiment for comparison.

    Args:
        df: Input dataset
        ground_truth_adj: Ground truth adjacency matrix
        output_dir: Output directory

    Returns:
        Baseline metrics
    """
    logger.info("Running baseline experiment...")

    from src.selector.algorithm_selector import run_selector
    from src.constraint_fusion.fusion import apply_hard_constraints
    from src.graph_generation.notears_torch import train_notears_torch
    from src.utils.metrics_extended import structural_hamming_distance, auc_precision_recall

    # Default baseline configuration
    baseline_config = {
        'selector': {'thresh_hard': 0.9, 'thresh_ambiguous': 0.4},
        'graph': {'lambda1': 0.1, 'max_iter': 100, 'lr': 1e-2}
    }

    # Run baseline pipeline
    selector_result = run_selector(
        df,
        thresh_hard=baseline_config['selector']['thresh_hard'],
        thresh_ambiguous=baseline_config['selector']['thresh_ambiguous']
    )

    n_vars = df.shape[1]
    initial_W = np.zeros((n_vars, n_vars))
    mask_fixed, W_fixed = apply_hard_constraints(initial_W, selector_result['hard'], n_vars)

    priors = {}
    for i, j, weight, score in selector_result['ambiguous'][:5]:
        priors[(i, j)] = 0.6

    W_baseline = train_notears_torch(
        df.values,
        priors_dict=priors,
        mask_fixed=mask_fixed,
        lambda1=baseline_config['graph']['lambda1'],
        max_iter=baseline_config['graph']['max_iter'],
        lr=baseline_config['graph']['lr'],
        verbose=False
    )

    # Compute baseline metrics
    threshold = 0.1
    W_est_bin = (np.abs(W_baseline) > threshold).astype(int)
    W_true_bin = (np.abs(ground_truth_adj) > 1e-8).astype(int)

    baseline_metrics = {
        'SHD': structural_hamming_distance(ground_truth_adj, W_baseline, threshold),
        'AUPR': auc_precision_recall(W_true_bin.flatten(), np.abs(W_baseline).flatten()),
        'edges_pred': int(np.sum(W_est_bin)),
        'hard_edges': len(selector_result['hard']),
        'ambiguous_edges': len(selector_result['ambiguous'])
    }

    # Save baseline results
    np.save(Path(output_dir) / "W_baseline.npy", W_baseline)
    with open(Path(output_dir) / "baseline_metrics.json", 'w') as f:
        json.dump(baseline_metrics, f, indent=2)

    logger.info(f"Baseline metrics: SHD={baseline_metrics['SHD']}, AUPR={baseline_metrics['AUPR']:.3f}")
    return baseline_metrics


def generate_optimization_report(results, baseline_metrics, output_dir):
    """
    Generate comprehensive optimization report.

    Args:
        results: AutoEval optimization results
        baseline_metrics: Baseline experiment metrics
        output_dir: Output directory
    """
    logger.info("Generating optimization report...")

    report = {
        'optimization_summary': {
            'total_iterations': results['total_iterations'],
            'best_iteration': results['best_iteration']['iteration'],
            'final_SHD': results['final_metrics']['SHD'],
            'final_AUPR': results['final_metrics']['AUPR'],
            'best_SHD': results['best_iteration']['metrics']['SHD'],
            'best_AUPR': results['best_iteration']['metrics']['AUPR']
        },
        'baseline_comparison': {
            'baseline_SHD': baseline_metrics['SHD'],
            'baseline_AUPR': baseline_metrics['AUPR'],
            'improvement_SHD': baseline_metrics['SHD'] - results['best_iteration']['metrics']['SHD'],
            'improvement_AUPR': results['best_iteration']['metrics']['AUPR'] - baseline_metrics['AUPR']
        },
        'optimization_trajectory': results['metrics_history'],
        'best_configuration': results['best_iteration']['config']
    }

    # Save detailed report
    with open(Path(output_dir) / "optimization_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    # Create summary table
    summary_data = {
        'Metric': ['SHD', 'AUPR', 'Edges Predicted', 'Hard Edges', 'Ambiguous Edges'],
        'Baseline': [
            baseline_metrics['SHD'],
            f"{baseline_metrics['AUPR']:.3f}",
            baseline_metrics['edges_pred'],
            baseline_metrics['hard_edges'],
            baseline_metrics['ambiguous_edges']
        ],
        'Best Optimized': [
            results['best_iteration']['metrics']['SHD'],
            f"{results['best_iteration']['metrics']['AUPR']:.3f}",
            results['best_iteration']['metrics']['edges_pred'],
            results['best_iteration']['metrics']['hard_edges'],
            results['best_iteration']['metrics']['ambiguous_edges']
        ],
        'Improvement': [
            f"{baseline_metrics['SHD'] - results['best_iteration']['metrics']['SHD']:+d}",
            f"{results['best_iteration']['metrics']['AUPR'] - baseline_metrics['AUPR']:+.3f}",
            f"{results['best_iteration']['metrics']['edges_pred'] - baseline_metrics['edges_pred']:+d}",
            "N/A",
            "N/A"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(Path(output_dir) / "summary_table.csv", index=False)

    logger.info("Optimization report generated successfully")
    return report


def main():
    """Main execution function."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=== Vortex-Causal Auto-Evaluation System ===")
    logger.info(f"Configuration: {args.max_iterations} iterations, dataset={args.dataset}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    try:
        # Load dataset
        df, ground_truth_adj = load_dataset(args.dataset, args.ground_truth)
        logger.info(f"Dataset loaded: shape={df.shape}, ground_truth_shape={ground_truth_adj.shape}")

        # Save ground truth for reference
        np.save(output_dir / "ground_truth.npy", ground_truth_adj)

        if args.baseline_only:
            # Run baseline experiment only
            baseline_metrics = run_baseline_experiment(df, ground_truth_adj, str(output_dir))
            logger.info("Baseline experiment completed")
            return

        # Run baseline experiment first
        baseline_metrics = run_baseline_experiment(df, ground_truth_adj, str(output_dir))

        # Run AutoEval optimization
        logger.info("Starting AutoEval optimization loop...")
        results = run_auto_eval(max_iterations=args.max_iterations, output_dir=str(output_dir))

        # Generate comprehensive report
        report = generate_optimization_report(results, baseline_metrics, str(output_dir))

        # Print final summary
        logger.info("=== OPTIMIZATION COMPLETED ===")
        logger.info(f"Total iterations: {results['total_iterations']}")
        logger.info(f"Best iteration: {results['best_iteration']['iteration']}")
        logger.info(f"Baseline SHD: {baseline_metrics['SHD']}")
        logger.info(f"Best SHD: {results['best_iteration']['metrics']['SHD']}")
        logger.info(f"SHD improvement: {baseline_metrics['SHD'] - results['best_iteration']['metrics']['SHD']}")
        logger.info(f"Baseline AUPR: {baseline_metrics['AUPR']:.3f}")
        logger.info(f"Best AUPR: {results['best_iteration']['metrics']['AUPR']:.3f}")
        logger.info(f"AUPR improvement: {results['best_iteration']['metrics']['AUPR'] - baseline_metrics['AUPR']:.3f}")

        # Check if targets were met
        if results['best_iteration']['metrics']['SHD'] < 10:
            logger.info("✅ SUCCESS: Target SHD < 10 achieved!")
        elif results['best_iteration']['metrics']['AUPR'] > 0.8:
            logger.info("✅ SUCCESS: High precision AUPR > 0.8 achieved!")
        else:
            logger.info("⚠️  Partial success: Consider increasing iterations or adjusting parameters")

        logger.info(f"All results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"AutoEval execution failed: {e}")
        raise


if __name__ == "__main__":
    main()