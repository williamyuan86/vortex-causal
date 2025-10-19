# src/utils/metrics_extended.py
"""
Extended performance metrics for causal discovery evaluation.

This module implements industry-standard metrics for evaluating causal graph learning:
- Structural Hamming Distance (SHD)
- Area Under ROC Curve (AUC-ROC)
- Area Under Precision-Recall Curve (AUC-PR)
- F1-score, Precision, Recall
- Total Variation Distance (TVD)
- Causal Effect Estimation Metrics
- Graph Topology Metrics
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import networkx as nx
import warnings


def structural_hamming_distance(W_true: np.ndarray, W_est: np.ndarray,
                              threshold: float = 0.01) -> int:
    """
    Compute Structural Hamming Distance (SHD) between true and estimated graphs.

    SHD counts the number of edge additions, deletions, and reversions needed
    to transform the estimated graph into the true graph.

    Args:
        W_true: True weighted adjacency matrix (n x n)
        W_est: Estimated weighted adjacency matrix (n x n)
        threshold: Threshold to binarize weighted matrices

    Returns:
        SHD: Structural Hamming Distance (lower is better)
    """
    n = W_true.shape[0]

    # Binarize matrices
    G_true = (np.abs(W_true) > threshold).astype(int)
    G_est = (np.abs(W_est) > threshold).astype(int)

    # Set diagonal to zero (no self-loops)
    np.fill_diagonal(G_true, 0)
    np.fill_diagonal(G_est, 0)

    # Count edge differences
    shd = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if G_true[i, j] != G_est[i, j]:
                shd += 1

    return shd


def compute_graph_metrics(W_true: np.ndarray, W_est: np.ndarray,
                         threshold: float = 0.01) -> Dict[str, float]:
    """
    Compute comprehensive graph evaluation metrics.

    Args:
        W_true: True weighted adjacency matrix
        W_est: Estimated weighted adjacency matrix
        threshold: Threshold for binarization

    Returns:
        Dictionary of metrics including precision, recall, F1, SHD, etc.
    """
    n = W_true.shape[0]

    # Binarize matrices
    G_true = (np.abs(W_true) > threshold).astype(int)
    G_est = (np.abs(W_est) > threshold).astype(int)

    # Set diagonal to zero
    np.fill_diagonal(G_true, 0)
    np.fill_diagonal(G_est, 0)

    # Flatten for sklearn metrics
    y_true = G_true.flatten()
    y_pred = G_est.flatten()
    y_scores = np.abs(W_est).flatten()

    # Basic metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # SHD
    shd = structural_hamming_distance(W_true, W_est, threshold)

    # Edge count metrics
    true_edges = np.sum(G_true)
    est_edges = np.sum(G_est)

    # False positives and false negatives
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Additional topology metrics
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'shd': shd,
        'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'true_edges': true_edges,
        'estimated_edges': est_edges,
        'edge_ratio': est_edges / true_edges if true_edges > 0 else float('inf'),
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'true_negatives': tn
    }

    return metrics


def compute_auc_metrics(W_true: np.ndarray, W_est: np.ndarray) -> Dict[str, float]:
    """
    Compute AUC-based metrics for weighted graph evaluation.

    Args:
        W_true: True weighted adjacency matrix
        W_est: Estimated weighted adjacency matrix

    Returns:
        Dictionary with AUC-ROC and AUC-PR scores
    """
    n = W_true.shape[0]

    # Prepare data
    y_true = (np.abs(W_true) > 1e-8).astype(int).flatten()
    y_scores = np.abs(W_est).flatten()

    # Remove diagonal elements
    mask = ~np.eye(n, dtype=bool).flatten()
    y_true = y_true[mask]
    y_scores = y_scores[mask]

    # Check if we have both positive and negative examples
    if len(np.unique(y_true)) < 2:
        warnings.warn("Only one class present in y_true. AUC score undefined.")
        return {'auc_roc': 0.5, 'auc_pr': 0.0}

    # Compute AUC metrics
    try:
        auc_roc = roc_auc_score(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)
    except ValueError as e:
        warnings.warn(f"AUC computation failed: {e}")
        auc_roc = 0.5
        auc_pr = 0.0

    return {'auc_roc': auc_roc, 'auc_pr': auc_pr}


def compute_causal_effect_metrics(true_ate: float, estimated_ate: float,
                                true_mu0: Optional[np.ndarray] = None,
                                true_mu1: Optional[np.ndarray] = None,
                                est_mu0: Optional[np.ndarray] = None,
                                est_mu1: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute causal effect estimation metrics.

    Args:
        true_ate: True Average Treatment Effect
        estimated_ate: Estimated Average Treatment Effect
        true_mu0: True potential outcomes under control (optional)
        true_mu1: True potential outcomes under treatment (optional)
        est_mu0: Estimated potential outcomes under control (optional)
        est_mu1: Estimated potential outcomes under treatment (optional)

    Returns:
        Dictionary of effect estimation metrics
    """
    metrics = {}

    # ATE error metrics
    ate_error = abs(estimated_ate - true_ate)
    ate_relative_error = ate_error / (abs(true_ate) + 1e-8)

    metrics['ate_abs_error'] = ate_error
    metrics['ate_rel_error'] = ate_relative_error
    metrics['ate_estimate'] = estimated_ate
    metrics['ate_true'] = true_ate

    # PEHE (Precision in Estimating Heterogeneous Effects) if available
    if all(x is not None for x in [true_mu0, true_mu1, est_mu0, est_mu1]):
        # PEHE = sqrt(mean((mu1_hat - mu0_hat - (mu1 - mu0))^2))
        true_cate = true_mu1 - true_mu0
        est_cate = est_mu1 - est_mu0
        pehe = np.sqrt(np.mean((est_cate - true_cate) ** 2))

        metrics['pehe'] = pehe
        metrics['cate_rmse'] = np.sqrt(np.mean((est_cate - true_cate) ** 2))

    return metrics


def compute_graph_topology_metrics(W: np.ndarray, threshold: float = 0.01) -> Dict[str, float]:
    """
    Compute graph topology metrics for a single graph.

    Args:
        W: Weighted adjacency matrix
        threshold: Threshold for binarization

    Returns:
        Dictionary of topology metrics
    """
    # Binarize and create NetworkX graph
    G_bin = (np.abs(W) > threshold).astype(int)
    np.fill_diagonal(G_bin, 0)

    # Create directed graph
    G = nx.from_numpy_array(G_bin, create_using=nx.DiGraph())

    metrics = {}

    # Basic connectivity
    metrics['n_nodes'] = G.number_of_nodes()
    metrics['n_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)

    # Node degree statistics
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]

    if in_degrees:
        metrics['avg_in_degree'] = np.mean(in_degrees)
        metrics['max_in_degree'] = np.max(in_degrees)
        metrics['min_in_degree'] = np.min(in_degrees)
    else:
        metrics['avg_in_degree'] = 0
        metrics['max_in_degree'] = 0
        metrics['min_in_degree'] = 0

    if out_degrees:
        metrics['avg_out_degree'] = np.mean(out_degrees)
        metrics['max_out_degree'] = np.max(out_degrees)
        metrics['min_out_degree'] = np.min(out_degrees)
    else:
        metrics['avg_out_degree'] = 0
        metrics['max_out_degree'] = 0
        metrics['min_out_degree'] = 0

    # Path-based metrics (if graph is not too large)
    if G.number_of_nodes() <= 50:  # Avoid expensive computations for large graphs
        try:
            # Check if graph is a DAG
            if nx.is_directed_acyclic_graph(G):
                metrics['is_dag'] = 1
                # Longest path length
                try:
                    metrics['longest_path'] = nx.dag_longest_path_length(G)
                except nx.NetworkXError:
                    metrics['longest_path'] = 0
            else:
                metrics['is_dag'] = 0
                metrics['longest_path'] = -1  # Not applicable for cyclic graphs

            # Connected components
            metrics['weakly_connected_components'] = nx.number_weakly_connected_components(G)
            metrics['strongly_connected_components'] = nx.number_strongly_connected_components(G)

        except Exception as e:
            warnings.warn(f"Graph topology computation failed: {e}")
            metrics.update({'is_dag': -1, 'longest_path': -1,
                          'weakly_connected_components': -1, 'strongly_connected_components': -1})

    return metrics


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Total Variation Distance between two probability distributions.

    Args:
        p: First probability distribution
        q: Second probability distribution

    Returns:
        TV distance (0 to 1)
    """
    return 0.5 * np.sum(np.abs(p - q))


def compute_comprehensive_benchmark(W_true: np.ndarray, W_est: np.ndarray,
                                  true_ate: Optional[float] = None,
                                  estimated_ate: Optional[float] = None,
                                  threshold: float = 0.01) -> Dict[str, float]:
    """
    Compute comprehensive benchmark metrics for causal discovery.

    Args:
        W_true: True weighted adjacency matrix
        W_est: Estimated weighted adjacency matrix
        true_ate: True ATE (optional)
        estimated_ate: Estimated ATE (optional)
        threshold: Threshold for binarization

    Returns:
        Comprehensive dictionary of all benchmark metrics
    """
    all_metrics = {}

    # Graph structure metrics
    graph_metrics = compute_graph_metrics(W_true, W_est, threshold)
    all_metrics.update({f'graph_{k}': v for k, v in graph_metrics.items()})

    # AUC metrics
    auc_metrics = compute_auc_metrics(W_true, W_est)
    all_metrics.update({f'auc_{k}': v for k, v in auc_metrics.items()})

    # Topology metrics for estimated graph
    top_metrics = compute_graph_topology_metrics(W_est, threshold)
    all_metrics.update({f'top_{k}': v for k, v in top_metrics.items()})

    # Causal effect metrics (if available)
    if true_ate is not None and estimated_ate is not None:
        effect_metrics = compute_causal_effect_metrics(true_ate, estimated_ate)
        all_metrics.update({f'effect_{k}': v for k, v in effect_metrics.items()})

    return all_metrics


def benchmark_multiple_methods(W_true: np.ndarray, methods_results: Dict[str, np.ndarray],
                             true_ate: Optional[float] = None,
                             ates: Optional[Dict[str, float]] = None,
                             threshold: float = 0.01) -> pd.DataFrame:
    """
    Benchmark multiple causal discovery methods.

    Args:
        W_true: True weighted adjacency matrix
        methods_results: Dictionary mapping method names to estimated adjacency matrices
        true_ate: True ATE (optional)
        ates: Dictionary mapping method names to estimated ATEs (optional)
        threshold: Threshold for binarization

    Returns:
        DataFrame with benchmark results for all methods
    """
    results = []

    for method_name, W_est in methods_results.items():
        # Get ATE for this method if available
        method_ate = ates.get(method_name) if ates else None

        # Compute comprehensive metrics
        metrics = compute_comprehensive_benchmark(
            W_true, W_est, true_ate, method_ate, threshold
        )

        # Add method name
        metrics['method'] = method_name
        results.append(metrics)

    return pd.DataFrame(results)


def generate_benchmark_report(results_df: pd.DataFrame,
                            save_path: Optional[str] = None) -> str:
    """
    Generate a human-readable benchmark report.

    Args:
        results_df: DataFrame with benchmark results
        save_path: Path to save the report (optional)

    Returns:
        Report as string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CAUSAL DISCOVERY BENCHMARK REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summary statistics
    report_lines.append("## SUMMARY STATISTICS")
    report_lines.append(f"Number of methods evaluated: {len(results_df)}")
    report_lines.append("")

    # Key metrics table
    key_metrics = ['method', 'graph_f1_score', 'graph_shd']

    # Add AUC metrics if available
    if 'auc_roc' in results_df.columns:
        key_metrics.append('auc_roc')
    if 'auc_pr' in results_df.columns:
        key_metrics.append('auc_pr')

    # Add effect estimation metrics if available
    if 'effect_ate_abs_error' in results_df.columns:
        key_metrics.append('effect_ate_abs_error')

    # Filter to only existing columns
    available_metrics = [m for m in key_metrics if m in results_df.columns]
    key_df = results_df[available_metrics].round(4)
    report_lines.append("## KEY PERFORMANCE METRICS")
    report_lines.append(key_df.to_string(index=False))
    report_lines.append("")

    # Best performing methods
    report_lines.append("## BEST PERFORMING METHODS")

    # Best F1 score
    best_f1 = results_df.loc[results_df['graph_f1_score'].idxmax()]
    report_lines.append(f"Best F1 Score: {best_f1['method']} ({best_f1['graph_f1_score']:.4f})")

    # Lowest SHD
    best_shd = results_df.loc[results_df['graph_shd'].idxmin()]
    report_lines.append(f"Lowest SHD: {best_shd['method']} ({best_shd['graph_shd']})")

    # Best AUC-ROC
    if 'auc_roc' in results_df.columns and len(results_df) > 0:
        best_auc = results_df.loc[results_df['auc_roc'].idxmax()]
        report_lines.append(f"Best AUC-ROC: {best_auc['method']} ({best_auc['auc_roc']:.4f})")

    # Best ATE estimation (if available)
    if 'effect_ate_abs_error' in results_df.columns and len(results_df) > 0:
        best_ate = results_df.loc[results_df['effect_ate_abs_error'].idxmin()]
        report_lines.append(f"Best ATE Estimation: {best_ate['method']} (error: {best_ate['effect_ate_abs_error']:.4f})")

    report_lines.append("")

    # Detailed metrics
    report_lines.append("## DETAILED METRICS")
    for _, row in results_df.iterrows():
        method = row['method']
        report_lines.append(f"\n### {method}")
        report_lines.append(f"F1 Score: {row['graph_f1_score']:.4f}")
        report_lines.append(f"Precision: {row['graph_precision']:.4f}")
        report_lines.append(f"Recall: {row['graph_recall']:.4f}")
        report_lines.append(f"SHD: {row['graph_shd']}")
        report_lines.append(f"True Edges: {row['graph_true_edges']}")
        report_lines.append(f"Estimated Edges: {row['graph_estimated_edges']}")

        if 'auc_roc' in row:
            report_lines.append(f"AUC-ROC: {row['auc_roc']:.4f}")
            report_lines.append(f"AUC-PR: {row['auc_pr']:.4f}")

        if 'effect_ate_abs_error' in row:
            report_lines.append(f"ATE Error: {row['effect_ate_abs_error']:.4f}")

    report_text = "\n".join(report_lines)

    # Save if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)

    return report_text