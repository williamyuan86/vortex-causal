# src/autoeval/controller.py
"""
AutoEval Controller for Vortex-Causal Iterative Optimization

This module implements the autonomous evaluation and tuning loop that continuously
refines causal discovery parameters based on performance feedback.
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

from src.utils.data_loader import create_synthetic_dataset, load_sachs
from src.selector.algorithm_selector import run_selector
from src.constraint_fusion.fusion import apply_hard_constraints
from src.graph_generation.notears_torch import train_notears_torch
from src.effect_estimation.estimator import ols_ate, ipw_ate
from src.utils.metrics_extended import structural_hamming_distance, compute_auc_metrics, compute_graph_metrics
from sklearn.metrics import f1_score, average_precision_score

# Optional imports
try:
    from src.effect_estimation.dragonnet import DragonNet, dragonnet_train, predict_dragonnet, ate_from_preds, pehe
    DRAGONNET_AVAILABLE = True
except ImportError:
    DRAGONNET_AVAILABLE = False

try:
    from src.ensemble.ensemble_cc_agent import ensemble_decide
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autoeval")


class AutoEvalController:
    """
    Autonomous evaluation controller for iterative causal discovery optimization.
    """

    def __init__(self, output_dir: str = "output", max_iterations: int = 10):
        """
        Initialize AutoEval controller.

        Args:
            output_dir: Directory for saving results and logs
            max_iterations: Maximum number of optimization iterations
        """
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.iteration_log = []

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)

        # Initialize metrics tracking
        self.metrics_history = {
            'iteration': [],
            'SHD': [],
            'AUPR': [],
            'edges_pred': [],
            'F1': [],
            'config_changes': [],
            'timestamps': []
        }

        # Default configuration (starting point)
        self.default_config = {
            'selector': {
                'thresh_hard': 0.9,
                'thresh_ambiguous': 0.4
            },
            'graph': {
                'lambda1': 0.1,
                'max_iter': 100,
                'lr': 1e-2
            },
            'fusion': {
                'lam_soft': 1.0,
                'conflict_resolution': 'hard_wins'
            },
            'effects': {
                'method': 'all'
            }
        }

        self.current_config = self.default_config.copy()

    def load_ground_truth(self, dataset_path: Optional[str] = None) -> np.ndarray:
        """
        Load ground truth adjacency matrix for evaluation.

        Args:
            dataset_path: Path to ground truth file (.npy)

        Returns:
            Ground truth adjacency matrix
        """
        if dataset_path and os.path.exists(dataset_path):
            return np.load(dataset_path)
        else:
            # Create synthetic dataset with known ground truth
            logger.info("Creating synthetic dataset with known ground truth")
            df, true_adj = create_synthetic_dataset(n_samples=1000, n_vars=8, seed=42)
            # Save ground truth for future use
            np.save(self.output_dir / "ground_truth_synth.npy", true_adj)
            return true_adj

    def evaluate_configuration(self, df: pd.DataFrame, true_adj: np.ndarray,
                             config: Dict) -> Dict[str, Any]:
        """
        Evaluate a configuration on the given dataset.

        Args:
            df: Input dataset
            true_adj: Ground truth adjacency matrix
            config: Configuration dictionary

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Step 1: Algorithm selection
            selector_result = run_selector(
                df,
                thresh_hard=config['selector']['thresh_hard'],
                thresh_ambiguous=config['selector']['thresh_ambiguous']
            )

            # Step 2: Apply hard constraints
            n_vars = df.shape[1]
            initial_W = np.zeros((n_vars, n_vars))
            mask_fixed, W_fixed = apply_hard_constraints(
                initial_W, selector_result['hard'], n_vars
            )

            # Step 3: Create soft priors from ambiguous edges
            priors = {}
            for i, j, weight, score in selector_result['ambiguous'][:5]:
                priors[(i, j)] = 0.6  # Base confidence

            # Step 4: Train causal graph
            W_final = train_notears_torch(
                df.values,
                priors_dict=priors,
                mask_fixed=mask_fixed,
                lambda1=config['graph']['lambda1'],
                max_iter=config['graph']['max_iter'],
                lr=config['graph']['lr'],
                verbose=False
            )

            # Step 5: Compute metrics
            threshold = 0.1
            W_est_bin = (np.abs(W_final) > threshold).astype(int)
            W_true_bin = (np.abs(true_adj) > 1e-8).astype(int)

            # Structural Hamming Distance
            shd = structural_hamming_distance(true_adj, W_final, threshold)

            # AUPR
            try:
                aupr = average_precision_score(W_true_bin.flatten(), np.abs(W_final).flatten())
            except:
                aupr = 0.0

            # F1-score
            try:
                f1 = f1_score(W_true_bin.flatten(), W_est_bin.flatten())
            except:
                f1 = 0.0

            # Edge count
            edges_pred = np.sum(W_est_bin)

            return {
                'SHD': shd,
                'AUPR': aupr,
                'F1': f1,
                'edges_pred': int(edges_pred),
                'hard_edges': len(selector_result['hard']),
                'ambiguous_edges': len(selector_result['ambiguous'])
            }

        except Exception as e:
            logger.error(f"Configuration evaluation failed: {e}")
            return {
                'SHD': float('inf'),
                'AUPR': 0.0,
                'F1': 0.0,
                'edges_pred': 0,
                'hard_edges': 0,
                'ambiguous_edges': 0,
                'error': str(e)
            }

    def adaptive_refinement(self, iteration: int, current_metrics: Dict,
                          prev_metrics: Optional[Dict]) -> Tuple[Dict, str]:
        """
        Adaptively refine configuration based on performance feedback.

        Args:
            iteration: Current iteration number
            current_metrics: Current performance metrics
            prev_metrics: Previous iteration metrics

        Returns:
            Tuple of (updated_config, change_description)
        """
        new_config = self.current_config.copy()
        changes = []

        if prev_metrics is None:
            # First iteration - no previous comparison
            return new_config, "Initial configuration"

        # Compute performance deltas
        delta_shd = prev_metrics['SHD'] - current_metrics['SHD']
        delta_aupr = current_metrics['AUPR'] - prev_metrics['AUPR']
        delta_f1 = current_metrics['F1'] - prev_metrics['F1']

        # Determine if improvement occurred
        improvement = (delta_shd > 0) or (delta_aupr > 0.01) or (delta_f1 > 0.01)

        if not improvement:
            # No improvement - apply adaptive tuning
            logger.info(f"No improvement detected (�SHD={delta_shd}, �AUPR={delta_aupr:.3f})")

            # Choose tuning strategy based on current performance
            if current_metrics['SHD'] > 20:
                # High SHD - need better structural learning
                if iteration % 3 == 0:
                    # Adjust selector thresholds
                    new_config['selector']['thresh_hard'] = max(0.7, self.current_config['selector']['thresh_hard'] - 0.05)
                    changes.append(f"Reduced thresh_hard to {new_config['selector']['thresh_hard']:.2f}")
                else:
                    # Adjust NOTEARS regularization
                    new_config['graph']['lambda1'] = min(0.5, self.current_config['graph']['lambda1'] * 1.2)
                    changes.append(f"Increased lambda1 to {new_config['graph']['lambda1']:.3f}")

            elif current_metrics['edges_pred'] > 20:  # Assuming around 8-10 variables typical
                # Too many edges - increase sparsity
                new_config['graph']['lambda1'] = min(0.5, self.current_config['graph']['lambda1'] * 1.3)
                changes.append(f"Increased sparsity (lambda1={new_config['graph']['lambda1']:.3f})")

            elif current_metrics['AUPR'] < 0.3:
                # Low precision - adjust selector to be more selective
                new_config['selector']['thresh_hard'] = min(0.95, self.current_config['selector']['thresh_hard'] + 0.03)
                changes.append(f"Increased selector precision (thresh_hard={new_config['selector']['thresh_hard']:.2f})")

            else:
                # Fine-tuning phase
                if iteration % 2 == 0:
                    new_config['graph']['lr'] = max(1e-4, self.current_config['graph']['lr'] * 0.9)
                    changes.append(f"Reduced learning rate to {new_config['graph']['lr']:.4f}")
                else:
                    new_config['fusion']['lam_soft'] = min(2.0, self.current_config['fusion']['lam_soft'] * 1.1)
                    changes.append(f"Adjusted soft prior weight to {new_config['fusion']['lam_soft']:.2f}")
        else:
            changes.append("No changes - improvement detected")

        return new_config, "; ".join(changes)

    def run_iteration(self, df: pd.DataFrame, true_adj: np.ndarray,
                     iteration: int) -> Tuple[Dict, Dict, str]:
        """
        Run a single optimization iteration.

        Args:
            df: Input dataset
            true_adj: Ground truth adjacency matrix
            iteration: Current iteration number

        Returns:
            Tuple of (metrics, config, change_description)
        """
        logger.info(f"=== Iteration {iteration + 1}/{self.max_iterations} ===")
        logger.info(f"Config: {self.current_config}")

        # Evaluate current configuration
        metrics = self.evaluate_configuration(df, true_adj, self.current_config)

        if 'error' in metrics:
            logger.error(f"Iteration {iteration + 1} failed: {metrics['error']}")
            return metrics, self.current_config, f"Evaluation failed: {metrics['error']}"

        # Log metrics
        logger.info(f"Metrics: SHD={metrics['SHD']}, AUPR={metrics['AUPR']:.3f}, "
                   f"F1={metrics['F1']:.3f}, edges={metrics['edges_pred']}")

        # Adaptive refinement for next iteration
        prev_metrics = self.metrics_history['SHD'][-1] if self.metrics_history['SHD'] else None
        prev_metrics_dict = None
        if prev_metrics is not None:
            prev_metrics_dict = {
                'SHD': self.metrics_history['SHD'][-1],
                'AUPR': self.metrics_history['AUPR'][-1],
                'F1': self.metrics_history['F1'][-1]
            }

        if iteration < self.max_iterations - 1:  # Don't refine after last iteration
            new_config, change_desc = self.adaptive_refinement(iteration, metrics, prev_metrics_dict)
        else:
            new_config, change_desc = self.current_config, "Final iteration - no refinement"

        return metrics, new_config, change_desc

    def check_stopping_conditions(self, iteration: int, metrics: Dict) -> Tuple[bool, str]:
        """
        Check if stopping conditions are met.

        Args:
            iteration: Current iteration number
            metrics: Current metrics

        Returns:
            Tuple of (should_stop, reason)
        """
        # Success condition: SHD < 10
        if metrics['SHD'] < 10:
            return True, f"Target SHD achieved: {metrics['SHD']}"

        # Success condition: High AUPR and reasonable SHD
        if metrics['AUPR'] > 0.8 and metrics['SHD'] < 15:
            return True, f"High precision achieved: AUPR={metrics['AUPR']:.3f}, SHD={metrics['SHD']}"

        # Check for stagnation
        if len(self.metrics_history['AUPR']) >= 3:
            recent_aupr = self.metrics_history['AUPR'][-3:]
            aupr_improvement = max(recent_aupr) - min(recent_aupr)
            if aupr_improvement < 0.01:
                return True, f"Stagnation detected: AUPR improvement < 0.01 over 3 iterations"

        # Check for consistent degradation
        if len(self.metrics_history['SHD']) >= 5:
            recent_shd = self.metrics_history['SHD'][-5:]
            if all(recent_shd[i] <= recent_shd[i+1] for i in range(len(recent_shd)-1)):
                return True, "Consistent performance degradation detected"

        return False, ""

    def save_iteration_log(self, iteration: int, metrics: Dict, config: Dict, change_desc: str):
        """
        Save iteration results to log.

        Args:
            iteration: Iteration number
            metrics: Performance metrics
            config: Configuration used
            change_desc: Description of changes made
        """
        log_entry = {
            'iteration': iteration + 1,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'config': config,
            'config_changes': change_desc,
            'delta': {}
        }

        # Compute deltas from previous iteration
        if len(self.iteration_log) > 0:
            prev_entry = self.iteration_log[-1]
            log_entry['delta'] = {
                '�SHD': prev_entry['metrics']['SHD'] - metrics['SHD'],
                '�AUPR': metrics['AUPR'] - prev_entry['metrics']['AUPR'],
                '�F1': metrics['F1'] - prev_entry['metrics']['F1'],
                '�edges': metrics['edges_pred'] - prev_entry['metrics']['edges_pred']
            }

        self.iteration_log.append(log_entry)

        # Update metrics history
        self.metrics_history['iteration'].append(iteration + 1)
        self.metrics_history['SHD'].append(metrics['SHD'])
        self.metrics_history['AUPR'].append(metrics['AUPR'])
        self.metrics_history['F1'].append(metrics['F1'])
        self.metrics_history['edges_pred'].append(metrics['edges_pred'])
        self.metrics_history['config_changes'].append(change_desc)
        self.metrics_history['timestamps'].append(datetime.now().isoformat())

    def run_optimization(self, df: Optional[pd.DataFrame] = None,
                        ground_truth_path: Optional[str] = None) -> Dict:
        """
        Run the complete optimization loop.

        Args:
            df: Input dataset (if None, uses synthetic data)
            ground_truth_path: Path to ground truth adjacency matrix

        Returns:
            Dictionary with optimization results and logs
        """
        logger.info("Starting AutoEval optimization loop")

        # Load data and ground truth
        if df is None:
            df, _ = create_synthetic_dataset(n_samples=1000, n_vars=8, seed=42)
            logger.info("Using synthetic dataset")

        true_adj = self.load_ground_truth(ground_truth_path)
        logger.info(f"Ground truth shape: {true_adj.shape}")

        # Store ground truth for adaptive refinement
        self.true_adj = true_adj

        # Run optimization iterations
        metrics = {}
        for iteration in range(self.max_iterations):
            metrics, new_config, change_desc = self.run_iteration(df, true_adj, iteration)

            # Save results
            self.save_iteration_log(iteration, metrics, self.current_config, change_desc)

            # Check stopping conditions
            should_stop, reason = self.check_stopping_conditions(iteration, metrics)
            if should_stop:
                logger.info(f"Stopping early: {reason}")
                break

            # Update configuration for next iteration
            self.current_config = new_config

        # Save final results
        self.save_results()

        return {
            'final_metrics': metrics,
            'best_iteration': self.get_best_iteration(),
            'total_iterations': len(self.iteration_log),
            'optimization_log': self.iteration_log,
            'metrics_history': self.metrics_history
        }

    def get_best_iteration(self) -> Dict:
        """
        Get the best iteration based on combined metrics.

        Returns:
            Dictionary with best iteration info
        """
        if not self.iteration_log:
            return {}

        # Calculate composite score (lower SHD and higher AUPR/F1 are better)
        best_score = float('inf')
        best_idx = 0

        for i, entry in enumerate(self.iteration_log):
            if 'error' in entry['metrics']:
                continue

            # Composite score: SHD - 10*AUPR - 10*F1 (lower is better)
            score = entry['metrics']['SHD'] - 10*entry['metrics']['AUPR'] - 10*entry['metrics']['F1']

            if score < best_score:
                best_score = score
                best_idx = i

        return {
            'iteration': self.iteration_log[best_idx]['iteration'],
            'metrics': self.iteration_log[best_idx]['metrics'],
            'config': self.iteration_log[best_idx]['config'],
            'composite_score': best_score
        }

    def save_results(self):
        """Save optimization results to files."""
        # Save detailed log
        with open(self.output_dir / "auto_eval_log.json", 'w') as f:
            json.dump(self.iteration_log, f, indent=2)

        # Save metrics summary
        summary_df = pd.DataFrame(self.metrics_history)
        summary_df.to_csv(self.output_dir / "optimization_summary.csv", index=False)

        # Save final configuration
        best_iter = self.get_best_iteration()
        with open(self.output_dir / "best_config.json", 'w') as f:
            json.dump(best_iter['config'], f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")


def run_auto_eval(max_iterations: int = 10, output_dir: str = "output") -> Dict:
    """
    Convenience function to run AutoEval optimization.

    Args:
        max_iterations: Maximum number of iterations
        output_dir: Output directory

    Returns:
        Optimization results
    """
    controller = AutoEvalController(output_dir=output_dir, max_iterations=max_iterations)
    return controller.run_optimization()


if __name__ == "__main__":
    # Example usage
    results = run_auto_eval(max_iterations=10, output_dir="output")
    print(f"Optimization completed. Best iteration: {results['best_iteration']['iteration']}")
    print(f"Final SHD: {results['final_metrics']['SHD']}")
    print(f"Final AUPR: {results['final_metrics']['AUPR']:.3f}")