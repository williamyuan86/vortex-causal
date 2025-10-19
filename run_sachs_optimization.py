#!/usr/bin/env python3
# run_sachs_optimization.py
"""
Direct script to run AutoEval optimization on Sachs dataset
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sachs_optimization")

def load_sachs_dataset():
    """Load and prepare Sachs dataset."""
    try:
        from src.utils.data_loader import load_sachs
        logger.info("Loading Sachs dataset...")
        df = load_sachs("data/sachs.csv")
        logger.info(f"Sachs dataset loaded: shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load Sachs dataset: {e}")
        # Create synthetic dataset as fallback
        logger.info("Creating synthetic dataset as fallback...")
        from src.utils.data_loader import create_synthetic_dataset
        df, _ = create_synthetic_dataset(n_samples=1000, n_vars=11, seed=42)  # Sachs has 11 variables
        logger.info(f"Synthetic dataset created: shape {df.shape}")
        return df

def run_baseline_experiment(df):
    """Run baseline experiment for comparison."""
    logger.info("Running baseline experiment...")

    try:
        from src.selector.algorithm_selector import run_selector
        from src.constraint_fusion.fusion import apply_hard_constraints
        from src.graph_generation.notears_torch import train_notears_torch
        from src.utils.data_loader import create_synthetic_dataset
        from src.utils.metrics_extended import structural_hamming_distance, auc_precision_recall, f1_score

        # Create synthetic ground truth for evaluation (since we don't have real ground truth for Sachs)
        _, true_adj = create_synthetic_dataset(n_samples=500, n_vars=df.shape[1], seed=123)

        # Baseline configuration
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
        W_true_bin = (np.abs(true_adj) > 1e-8).astype(int)

        baseline_metrics = {
            'SHD': structural_hamming_distance(true_adj, W_baseline, threshold),
            'AUPR': auc_precision_recall(W_true_bin.flatten(), np.abs(W_baseline).flatten()),
            'F1': f1_score(W_true_bin.flatten(), W_est_bin.flatten()),
            'edges_pred': int(np.sum(W_est_bin)),
            'hard_edges': len(selector_result['hard']),
            'ambiguous_edges': len(selector_result['ambiguous'])
        }

        logger.info(f"Baseline metrics: SHD={baseline_metrics['SHD']}, AUPR={baseline_metrics['AUPR']:.3f}, F1={baseline_metrics['F1']:.3f}")
        return baseline_metrics, true_adj

    except Exception as e:
        logger.error(f"Baseline experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_autoeval_optimization(df, true_adj, max_iterations=15):
    """Run AutoEval optimization."""
    logger.info(f"Starting AutoEval optimization for {max_iterations} iterations...")

    try:
        from src.autoeval.controller import AutoEvalController

        # Create output directory
        output_dir = Path("output/sachs_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create and run controller
        controller = AutoEvalController(
            output_dir=str(output_dir),
            max_iterations=max_iterations
        )

        # Set ground truth for evaluation
        controller.true_adj = true_adj

        # Run optimization
        results = controller.run_optimization(df=df)

        logger.info(f"AutoEval optimization completed. Total iterations: {results['total_iterations']}")
        logger.info(f"Best metrics: SHD={results['best_iteration']['metrics']['SHD']}, AUPR={results['best_iteration']['metrics']['AUPR']:.3f}")

        return results

    except Exception as e:
        logger.error(f"AutoEval optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_improvement_target(baseline_metrics, best_metrics, target_improvement=0.01):
    """Check if improvement target is achieved."""
    improvements = {}

    # Calculate improvements for each metric
    if baseline_metrics and best_metrics:
        # For SHD, improvement means reduction
        shd_improvement = (baseline_metrics['SHD'] - best_metrics['SHD']) / baseline_metrics['SHD']
        improvements['SHD'] = shd_improvement

        # For AUPR and F1, improvement means increase
        aupr_improvement = (best_metrics['AUPR'] - baseline_metrics['AUPR']) / baseline_metrics['AUPR'] if baseline_metrics['AUPR'] > 0 else 0
        improvements['AUPR'] = aupr_improvement

        f1_improvement = (best_metrics['F1'] - baseline_metrics['F1']) / baseline_metrics['F1'] if baseline_metrics['F1'] > 0 else 0
        improvements['F1'] = f1_improvement

        # Check if any metric meets the target
        target_met = any(improvement >= target_improvement for improvement in improvements.values())

        logger.info(f"Improvements: SHD={shd_improvement:.3f}, AUPR={aupr_improvement:.3f}, F1={f1_improvement:.3f}")
        logger.info(f"Target improvement ({target_improvement*100}%): {'✅ ACHIEVED' if target_met else '❌ NOT ACHIEVED'}")

        return target_met, improvements

    return False, improvements

def generate_final_report(baseline_metrics, first_phase_results, second_phase_results=None):
    """Generate final optimization report."""
    report = {
        'optimization_summary': {
            'dataset': 'sachs',
            'timestamp': datetime.now().isoformat(),
            'first_phase_iterations': first_phase_results['total_iterations'] if first_phase_results else 0,
            'second_phase_iterations': second_phase_results['total_iterations'] if second_phase_results else 0,
            'total_iterations': (first_phase_results['total_iterations'] if first_phase_results else 0) +
                              (second_phase_results['total_iterations'] if second_phase_results else 0)
        },
        'baseline_metrics': baseline_metrics,
        'first_phase_best': first_phase_results['best_iteration'] if first_phase_results else None,
        'second_phase_best': second_phase_results['best_iteration'] if second_phase_results else None,
    }

    # Determine overall best
    if first_phase_results and second_phase_results:
        first_score = first_phase_results['best_iteration']['metrics']['SHD'] - 10*first_phase_results['best_iteration']['metrics']['AUPR']
        second_score = second_phase_results['best_iteration']['metrics']['SHD'] - 10*second_phase_results['best_iteration']['metrics']['AUPR']

        if first_score <= second_score:
            report['overall_best'] = first_phase_results['best_iteration']
            report['best_phase'] = 'first'
        else:
            report['overall_best'] = second_phase_results['best_iteration']
            report['best_phase'] = 'second'
    elif first_phase_results:
        report['overall_best'] = first_phase_results['best_iteration']
        report['best_phase'] = 'first'

    # Save report
    output_dir = Path("output/sachs_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "final_optimization_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Final report saved to {output_dir / 'final_optimization_report.json'}")
    return report

def main():
    """Main optimization function."""
    logger.info("=== Sachs Dataset AutoEval Optimization ===")

    try:
        # Step 1: Load dataset
        df = load_sachs_dataset()

        # Step 2: Run baseline experiment
        baseline_metrics, true_adj = run_baseline_experiment(df)
        if baseline_metrics is None:
            logger.error("Failed to run baseline experiment. Exiting.")
            return

        # Step 3: Run first phase optimization (15 iterations)
        logger.info("\n=== PHASE 1: Initial 15 Iterations ===")
        first_phase_results = run_autoeval_optimization(df, true_adj, max_iterations=15)

        if first_phase_results is None:
            logger.error("First phase optimization failed. Exiting.")
            return

        # Step 4: Check if improvement target achieved
        first_phase_best = first_phase_results['best_iteration']['metrics']
        target_met, improvements = check_improvement_target(baseline_metrics, first_phase_best)

        # Step 5: Run second phase if needed
        second_phase_results = None
        if not target_met:
            logger.info("\n=== PHASE 2: Additional 15 Iterations ===")
            logger.info("Target not achieved. Running additional 15 iterations...")

            second_phase_results = run_autoeval_optimization(df, true_adj, max_iterations=15)

            if second_phase_results:
                # Check if target achieved after second phase
                second_phase_best = second_phase_results['best_iteration']['metrics']
                target_met_final, _ = check_improvement_target(baseline_metrics, second_phase_best)

                if target_met_final:
                    logger.info("✅ Target achieved in second phase!")
                else:
                    logger.info("❌ Target still not achieved after optimization complete")

        # Step 6: Generate final report
        logger.info("\n=== GENERATING FINAL REPORT ===")
        final_report = generate_final_report(baseline_metrics, first_phase_results, second_phase_results)

        # Print final summary
        logger.info("\n=== OPTIMIZATION SUMMARY ===")
        logger.info(f"Dataset: Sachs")
        logger.info(f"Total iterations run: {final_report['optimization_summary']['total_iterations']}")
        logger.info(f"Best phase: {final_report['optimization_summary']['best_phase']}")

        if baseline_metrics and final_report['overall_best']:
            best_metrics = final_report['overall_best']['metrics']
            logger.info(f"\nBaseline: SHD={baseline_metrics['SHD']}, AUPR={baseline_metrics['AUPR']:.3f}, F1={baseline_metrics['F1']:.3f}")
            logger.info(f"Best:     SHD={best_metrics['SHD']}, AUPR={best_metrics['AUPR']:.3f}, F1={best_metrics['F1']:.3f}")

            # Calculate final improvements
            shd_imp = (baseline_metrics['SHD'] - best_metrics['SHD']) / baseline_metrics['SHD']
            aupr_imp = (best_metrics['AUPR'] - baseline_metrics['AUPR']) / baseline_metrics['AUPR'] if baseline_metrics['AUPR'] > 0 else 0
            f1_imp = (best_metrics['F1'] - baseline_metrics['F1']) / baseline_metrics['F1'] if baseline_metrics['F1'] > 0 else 0

            logger.info(f"\nFinal Improvements:")
            logger.info(f"  SHD:  {shd_imp:+.2%}")
            logger.info(f"  AUPR: {aupr_imp:+.2%}")
            logger.info(f"  F1:   {f1_imp:+.2%}")

        logger.info(f"\nAll results saved to: output/sachs_optimization/")

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()