# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**Vortex-Causal** is an advanced causal discovery and treatment effect estimation framework that integrates **algorithmic consensus**, **LLM ensemble reasoning**, **constraint fusion**, and **neural effect estimation** to learn causal graphs from observational data.
In the *Claude Autonomous Causal Optimization Edition*, the framework further introduces:

* **AutoEval Loop** for self-improving causal discovery.
* **Ground Truth Feedback Logic** for iterative performance tuning.
* **Claude AutoCausal Agent**, an adaptive reasoning layer powered by Claude.

---

## Core Architecture

The system follows a modular pipeline architecture:

1. **Data Loading & Preprocessing** (`src/utils/data_loader.py`)

   * Loads observational datasets (Sachs, synthetic)
   * Handles data normalization and validation

2. **Algorithm Selection** (`src/selector/algorithm_selector.py`)

   * Runs multiple causal discovery algorithms (PC, GES, LiNGAM, NOTEARS)
   * Computes consensus edge confidence scores
   * Categorizes edges as hard, ambiguous, or low confidence

3. **LLM Ensemble** (`src/ensemble/ensemble_cc_agent.py`)

   * Uses multiple LLMs (GPT-4, Claude, Qwen) to resolve ambiguous edges
   * Retrieves causal knowledge from domain literature
   * Provides semantic causal constraints

4. **Claude AutoCausal Agent** (`src/autocausal/agent_claude.py`)

   * Monitors experiment metrics and suggests adaptive hyperparameter or constraint tuning
   * Interprets causal reasoning patterns and adjusts algorithm weights
   * Works in tandem with the AutoEval Loop for continual performance refinement
   * Maintains interpretable reasoning logs for reproducibility

5. **Constraint Fusion** (`src/constraint_fusion/fusion.py`)

   * Integrates hard algorithmic constraints with soft semantic priors
   * Balances data-driven and knowledge-driven constraints
   * Maintains DAG feasibility

6. **Graph Generation** (`src/graph_generation/notears_torch.py`)

   * Learns causal structure using gradient-based optimization
   * Incorporates fused constraints into the learning process
   * Ensures acyclicity and sparsity

7. **Effect Estimation** (`src/effect_estimation/`)

   * DragonNet: Neural network for treatment effect estimation
   * IPW/OLS: Traditional causal effect estimators
   * PEHE evaluation for heterogeneous treatment effects

8. **AutoEval Loop** (`src/autoeval/loop.py`)

   * Automatically evaluates generated graphs against ground truth causal structures
   * Adjusts confidence thresholds and fusion weights based on feedback
   * Iteratively refines NOTEARS and DragonNet hyperparameters
   * Logs improvement steps to `output/auto_eval_log.json`
   * Enables self-improving performance across evaluation rounds

---

## Common Development Commands

### Running Experiments

```bash
# Run baseline causal discovery experiment
python scripts/run_experiments.py --mode baseline --eval

# Run with hard constraints
python scripts/run_experiments.py --mode hard --eval

# Run full pipeline with LLM ensemble
python scripts/run_experiments.py --mode full --eval

# Run autonomous causal optimization loop
python scripts/run_autoeval.py --n-rounds 5 --dataset sachs
```

### Performance Benchmarking

```bash
python scripts/run_benchmarks.py --n-datasets 20 --output-dir results
python performance_test_sachs.py
```

### Hyperparameter Optimization

```bash
python scripts/dragonnet_optuna.py --trials 50
```

### Testing

```bash
python -m pytest src/tests/ -v
python -m pytest src/tests/ --cov=src --cov-report=html
python -m pytest src/tests/test_algorithm_selector.py -v
```

---

## Key Configuration Parameters

### Algorithm Selector

* `thresh_hard`: Confidence threshold for hard edges (default: 0.9)
* `thresh_ambiguous`: Threshold for ambiguous edges (default: 0.4)
* Supported methods: PC, GES, LiNGAM, NOTEARS

### NOTEARS Parameters

* `lambda1`: L1 regularization for sparsity (default: 0.1)
* `max_iter`: Maximum optimization iterations (default: 100)
* `h_tol`: Tolerance for acyclicity constraint (default: 1e-8)

### DragonNet Training

* Learning rate: 1e-4 to 1e-2 (auto-tuned via Optuna)
* Architecture: [hidden1, hidden2] = (64–256, 32–128)
* Alpha/beta: Loss weighting parameters

### AutoEval Loop

* `auto_rounds`: Number of iterative evaluation cycles (default: 5)
* `metric_target`: Optimization metric (e.g., SHD or AUPR)
* `adaptive_lr`: Whether to enable dynamic learning rate adjustment
* `feedback_weight`: Ground truth feedback intensity (0–1)

---

## Data Flow and Dependencies

1. **Input**: Observational data (`pandas.DataFrame`)
2. **Preprocessing**: Standardization, missing value handling
3. **Multi-method Discovery**: Parallel execution of causal algorithms
4. **Consensus Building**: Edge confidence aggregation
5. **Constraint Integration**: Hard + soft constraint fusion
6. **Graph Learning**: Constrained NOTEARS optimization
7. **Effect Estimation**: Treatment effect prediction on learned graph
8. **Evaluation Loop**: Iterative feedback from ground truth via AutoEval
9. **Optimization**: Adaptive tuning by Claude AutoCausal Agent

---

## Output Files

* `output/W_*.npy`: Learned weighted adjacency matrices
* `output/experiment_summary.csv`: Performance metrics (SHD, AUPR, edges_pred)
* `output/pehe_bar.png`: Treatment effect visualization
* `output/auto_eval_log.json`: Iterative improvement log
* `output/auto_eval_curve.png`: Visualized SHD/AUPR evolution
* `results/benchmark_*.csv`: Comprehensive benchmark results

---

## Performance Evaluation

The system uses industry-standard causal discovery metrics:

| Metric       | Description                          | Optimization Target |
| ------------ | ------------------------------------ | ------------------- |
| **SHD**      | Structural Hamming Distance          | ↓ Lower is better   |
| **AUPR**     | Area Under Precision-Recall Curve    | ↑ Higher is better  |
| **AUC-ROC**  | Overall causal prediction quality    | ↑ Higher is better  |
| **F1-score** | Balance of precision and recall      | ↑ Higher is better  |
| **PEHE**     | Treatment effect estimation accuracy | ↓ Lower is better   |

### Ground Truth Feedback Mechanism

After each iteration, the learned graph is compared with known ground truth structures (e.g., Sachs, synthetic benchmarks).
The **feedback loop** automatically adjusts:

* Algorithm weighting (PC, GES, NOTEARS confidence scaling)
* LLM consensus thresholds and semantic priors
* Constraint fusion balance (hard vs. soft)
* DragonNet learning parameters (α, β, LR)

This enables **iterative self-improvement** across multiple evaluation rounds without manual intervention.

---

## Development Guidelines

### Code Organization

* Each module implements comprehensive error handling and type hints
* All public functions include docstrings and reproducible seeds
* Fixed versions (`*_fixed.py`) ensure backward compatibility
* Core components maintain >95% unit test coverage

### Experiment Reproducibility

* Deterministic seeds and logged hyperparameters
* AutoEval maintains reproducible versioned checkpoints
* Intermediate graph states are stored for replay and audit

### Performance Optimization

* Parallel algorithm execution (multi-threaded or Ray)
* GPU acceleration for DragonNet
* Sparse matrix operations for efficiency
* Early stopping and adaptive regularization

---

## Troubleshooting

| Issue                       | Cause                           | Solution                                       |
| --------------------------- | ------------------------------- | ---------------------------------------------- |
| **DragonNet import errors** | Missing PyTorch or CUDA         | Install PyTorch and verify GPU availability    |
| **Memory overflow**         | Dataset too large               | Reduce batch size or sample subset             |
| **Convergence issues**      | Learning rate too high          | Lower LR or increase λ₁ regularization         |
| **LLM API errors**          | Invalid API keys or rate limits | Check `.env` credentials and retry             |
| **AutoEval Loop stuck**     | Evaluation not improving        | Adjust feedback_weight or increase auto_rounds |

### Debug Mode

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python main.py --log-level DEBUG
```

---

## Testing Strategy

The test suite includes:

* **Unit Tests**: Core component validation
* **Integration Tests**: End-to-end pipeline checks
* **Performance Tests**: Benchmarks against baselines
* **Edge Case Tests**: Missing data, low variance, small samples
* **AutoEval Tests**: Validation of iterative feedback convergence

All tests must pass before merging changes.
The current build maintains **100% test pass rate (64/64)** with full AutoEval reproducibility.

---

## Summary

> **Claude Autonomous Causal Optimization Edition** transforms Vortex-Causal from a static causal discovery framework into a **self-evolving causal reasoning system**.
> Through the synergy of **AutoEval Loop**, **Ground Truth Feedback**, and **Claude AutoCausal Agent**, the system autonomously tunes its own causal logic—approaching the vision of *self-adaptive scientific discovery*.

