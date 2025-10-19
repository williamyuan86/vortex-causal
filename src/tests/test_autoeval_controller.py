#!/usr/bin/env python3
"""
Test script for AutoEval Controller class
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.autoeval.controller import AutoEvalController
from src.utils.data_loader import create_synthetic_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_controller_initialization():
    """Test basic controller initialization."""
    print("=== Testing Controller Initialization ===")

    controller = AutoEvalController(output_dir="test_output", max_iterations=5)

    # Check basic attributes
    assert controller.max_iterations == 5
    assert controller.output_dir == Path("test_output")
    assert len(controller.default_config) == 4  # selector, graph, fusion, effects
    assert len(controller.metrics_history) == 7  # iteration, SHD, AUPR, edges_pred, F1, config_changes, timestamps

    # Check default config values
    assert controller.default_config['selector']['thresh_hard'] == 0.9
    assert controller.default_config['graph']['lambda1'] == 0.1
    assert controller.default_config['fusion']['lam_soft'] == 1.0

    print("PASS: Controller initialization test passed")
    return controller

def test_ground_truth_loading(controller):
    """Test ground truth loading functionality."""
    print("\n=== Testing Ground Truth Loading ===")

    # Test with synthetic data
    true_adj = controller.load_ground_truth()

    assert true_adj is not None
    assert isinstance(true_adj, np.ndarray)
    assert true_adj.shape[0] == true_adj.shape[1]  # Square matrix

    print(f"PASS: Ground truth loaded successfully, shape: {true_adj.shape}")
    return true_adj

def test_configuration_evaluation(controller, df, true_adj):
    """Test single configuration evaluation."""
    print("\n=== Testing Configuration Evaluation ===")

    # Test with default configuration
    config = controller.default_config.copy()
    metrics = controller.evaluate_configuration(df, true_adj, config)

    # Check required metrics
    required_keys = ['SHD', 'AUPR', 'F1', 'edges_pred', 'hard_edges', 'ambiguous_edges']
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"

    # Check metric types and ranges
    assert isinstance(metrics['SHD'], (int, float)) and metrics['SHD'] >= 0
    assert isinstance(metrics['AUPR'], (int, float)) and 0 <= metrics['AUPR'] <= 1
    assert isinstance(metrics['F1'], (int, float)) and 0 <= metrics['F1'] <= 1
    assert isinstance(metrics['edges_pred'], int) and metrics['edges_pred'] >= 0

    print(f"PASS: Configuration evaluation completed:")
    print(f"  SHD: {metrics['SHD']}")
    print(f"  AUPR: {metrics['AUPR']:.3f}")
    print(f"  F1: {metrics['F1']:.3f}")
    print(f"  Edges predicted: {metrics['edges_pred']}")
    print(f"  Hard edges: {metrics['hard_edges']}")
    print(f"  Ambiguous edges: {metrics['ambiguous_edges']}")

    return metrics

def test_adaptive_refinement(controller):
    """Test adaptive refinement logic."""
    print("\n=== Testing Adaptive Refinement ===")

    # Test with no previous metrics
    current_metrics = {'SHD': 15, 'AUPR': 0.4, 'F1': 0.3}
    new_config, change_desc = controller.adaptive_refinement(0, current_metrics, None)
    assert change_desc == "Initial configuration"

    # Test with improvement
    prev_metrics = {'SHD': 20, 'AUPR': 0.3, 'F1': 0.2}
    new_config, change_desc = controller.adaptive_refinement(1, current_metrics, prev_metrics)
    assert "No changes - improvement detected" in change_desc

    # Test with no improvement (high SHD)
    bad_metrics = {'SHD': 25, 'AUPR': 0.2, 'F1': 0.15}
    new_config, change_desc = controller.adaptive_refinement(2, bad_metrics, current_metrics)
    assert "Reduced thresh_hard" in change_desc or "Increased lambda1" in change_desc

    print(f"PASS: Adaptive refinement test passed:")
    print(f"  No prev metrics: {change_desc}")
    print(f"  With improvement: {change_desc}")
    print(f"  No improvement: {change_desc}")

def test_single_iteration(controller, df, true_adj):
    """Test a single optimization iteration."""
    print("\n=== Testing Single Iteration ===")

    iteration = 0
    metrics, new_config, change_desc = controller.run_iteration(df, true_adj, iteration)

    # Check returned values
    assert isinstance(metrics, dict)
    assert isinstance(new_config, dict)
    assert isinstance(change_desc, str)

    # Check metrics contain expected keys
    required_keys = ['SHD', 'AUPR', 'F1', 'edges_pred']
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"

    print(f"PASS: Single iteration completed:")
    print(f"  {change_desc}")
    print(f"  SHD: {metrics['SHD']}, AUPR: {metrics['AUPR']:.3f}")

    return metrics

def test_stopping_conditions(controller):
    """Test stopping condition logic."""
    print("\n=== Testing Stopping Conditions ===")

    # Add some fake metrics history
    controller.metrics_history['SHD'] = [20, 15, 12]
    controller.metrics_history['AUPR'] = [0.3, 0.4, 0.5]
    controller.metrics_history['F1'] = [0.2, 0.3, 0.4]

    # Test success condition (low SHD)
    good_metrics = {'SHD': 8, 'AUPR': 0.7, 'F1': 0.6}
    should_stop, reason = controller.check_stopping_conditions(2, good_metrics)
    assert should_stop and "Target SHD achieved" in reason

    # Test stagnation condition
    controller.metrics_history['AUPR'] = [0.5, 0.505, 0.502, 0.501]
    stagnant_metrics = {'SHD': 15, 'AUPR': 0.503, 'F1': 0.4}
    should_stop, reason = controller.check_stopping_conditions(3, stagnant_metrics)
    assert should_stop and "Stagnation detected" in reason

    print(f"PASS: Stopping conditions test passed:")
    print(f"  Low SHD: {reason}")
    print(f"  Stagnation: {reason}")

def test_full_optimization():
    """Test a shortened full optimization loop."""
    print("\n=== Testing Full Optimization Loop (3 iterations) ===")

    # Create new controller for this test
    controller = AutoEvalController(output_dir="test_output_full", max_iterations=3)

    # Create synthetic dataset
    df, true_adj = create_synthetic_dataset(n_samples=500, n_vars=6, seed=42)

    # Run optimization
    results = controller.run_optimization(df=df)

    # Check results structure
    required_keys = ['final_metrics', 'best_iteration', 'total_iterations', 'optimization_log', 'metrics_history']
    for key in required_keys:
        assert key in results, f"Missing result key: {key}"

    # Check that we ran at least one iteration
    assert results['total_iterations'] >= 1
    assert len(results['optimization_log']) >= 1

    # Check that best iteration has required fields
    best_iter = results['best_iteration']
    assert 'iteration' in best_iter
    assert 'metrics' in best_iter
    assert 'composite_score' in best_iter

    print(f"PASS: Full optimization completed:")
    print(f"  Total iterations: {results['total_iterations']}")
    print(f"  Best iteration: {best_iter['iteration']}")
    print(f"  Final SHD: {results['final_metrics']['SHD']}")
    print(f"  Final AUPR: {results['final_metrics']['AUPR']:.3f}")
    print(f"  Files saved to: {controller.output_dir}")

def test_error_handling(controller, df, true_adj):
    """Test error handling in configuration evaluation."""
    print("\n=== Testing Error Handling ===")

    # Test with invalid configuration
    invalid_config = {
        'selector': {'thresh_hard': 1.5, 'thresh_ambiguous': 0.8},  # Invalid thresholds
        'graph': {'lambda1': -0.1, 'max_iter': 0, 'lr': 1e-2},
        'fusion': {'lam_soft': 1.0, 'conflict_resolution': 'hard_wins'},
        'effects': {'method': 'all'}
    }

    metrics = controller.evaluate_configuration(df, true_adj, invalid_config)

    # Should handle error gracefully and return default values
    assert 'error' in metrics or metrics['SHD'] == float('inf')

    print("PASS: Error handling test passed - invalid configuration handled gracefully")

def run_all_tests():
    """Run all tests."""
    print("Starting AutoEval Controller Tests\n")

    try:
        # Test 1: Initialization
        controller = test_controller_initialization()

        # Create test data
        df, true_adj = create_synthetic_dataset(n_samples=500, n_vars=6, seed=42)

        # Test 2: Ground truth loading
        true_adj = test_ground_truth_loading(controller)

        # Test 3: Configuration evaluation
        test_configuration_evaluation(controller, df, true_adj)

        # Test 4: Adaptive refinement
        test_adaptive_refinement(controller)

        # Test 5: Single iteration
        test_single_iteration(controller, df, true_adj)

        # Test 6: Stopping conditions
        test_stopping_conditions(controller)

        # Test 7: Error handling
        test_error_handling(controller, df, true_adj)

        # Test 8: Full optimization (separate controller)
        test_full_optimization()

        print("\n" + "="*50)
        print("ALL TESTS PASSED!")
        print("AutoEval Controller is working correctly.")
        print("="*50)

    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)