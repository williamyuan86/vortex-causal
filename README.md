
---

# 🧠 CLAUDE.md — Vortex-Causal Autonomous Causal Discovery & Refinement Protocol

This file defines the intelligent development and auto-evaluation workflow for **Claude Code** when working inside the **Vortex-Causal** repository.
Claude acts as an autonomous research engineer that continuously refines causal discovery and treatment effect estimation pipelines until measurable performance gains converge.

---

## 🎯 Project Mission

**Vortex-Causal** aims to discover and refine causal graphs by combining:

1. **Algorithmic consensus** (NOTEARS, PC-like methods)
2. **LLM ensemble reasoning** (GPT-4, Claude, Qwen)
3. **Constraint fusion** (hard + soft priors)
4. **Treatment effect estimation** (DragonNet, IPW, OLS)

The ultimate objective is to **reduce SHD** (Structural Hamming Distance) and **increase AUPR** / **PEHE** scores relative to ground truth causal structures.

---

## 🧩 Core Components

| Module                                  | Function                                |
| --------------------------------------- | --------------------------------------- |
| `src/selector/algorithm_selector.py`    | Consensus-based edge confidence scoring |
| `src/ensemble/ensemble_cc_agent.py`     | LLM ensemble for ambiguous causal edges |
| `src/graph_generation/notears_torch.py` | Constraint-aware NOTEARS learner        |
| `src/constraint_fusion/fusion.py`       | Hard/soft constraint integration        |
| `src/effect_estimation/dragonnet.py`    | Neural treatment effect estimator       |
| `src/utils/metrics.py`                  | SHD, AUPR, PEHE metric evaluation       |
| `scripts/run_experiments.py`            | Unified experimental pipeline           |
| `scripts/dragonnet_optuna.py`           | Hyperparameter tuning with Optuna       |

---

## ⚙️ Data and Ground Truth

* Ground truth adjacency matrix: `data/ground_truth.npy`
* Experimental outputs:

  * `output/W_*.npy`: learned graphs
  * `output/experiment_summary.csv`: SHD, AUPR, edges_pred
  * `output/pehe_bar.png`: visualization
  * `output/auto_eval_log.json`: longitudinal tuning log

Input data follows pandas DataFrame format, with each column representing a causal variable.

---

## 🧠 Claude Code Role Definition

You are the **autonomous causal refinement assistant** for Vortex-Causal.
Your goals are:

1. **Causal Integrity**: preserve DAG structure and avoid cyclic edges.
2. **Performance Optimization**: minimize SHD, maximize AUPR and PEHE.
3. **Explainability**: provide concise reasoning for each algorithmic or hyperparameter change.
4. **Self-Evaluation**: automatically benchmark new code changes against the previous best configuration.

---

## 🔁 AutoEval Loop — Continuous Improvement Protocol

Claude follows this iterative loop during development sessions:

### 1️⃣ Initialization

If `output/experiment_summary.csv` does not exist, run the baseline experiment:

```bash
python scripts/run_experiments.py --mode baseline --eval
```

Store baseline results under `output/auto_eval_log.json`.

### 2️⃣ Evaluate Current Performance

After each code change, run:

```bash
python scripts/run_experiments.py --mode hard --eval
```

Extract metrics (`SHD`, `AUPR`, `edges_pred`) from `output/experiment_summary.csv`.

### 3️⃣ Compare to Previous Iteration

* Compute ΔSHD = SHD_prev − SHD_new
* Compute ΔAUPR = AUPR_new − AUPR_prev
* If ΔSHD > 0 or ΔAUPR > 0.05 → record improvement
* Otherwise → proceed to adaptive refinement

### 4️⃣ Adaptive Refinement Strategy

If no improvement detected, propose and implement **one targeted change** from the following levers:

| Area               | Action                                                  | Target                                |
| ------------------ | ------------------------------------------------------- | ------------------------------------- |
| Selector           | Adjust confidence thresholds (`conf_hard`, `conf_soft`) | Reduce false positives                |
| NOTEARS            | Tune `lambda1`, `max_iter`, or learning rate            | Better sparsity control               |
| LLM Ensemble       | Change model voting weight or prompt schema             | Increase consensus precision          |
| Constraint Fusion  | Modify weighting between hard/soft priors               | Balance algorithmic vs. semantic bias |
| Data Normalization | Adjust z-score or minmax scaling                        | Stabilize correlation structure       |

After change → rerun evaluation.

### 5️⃣ Auto-Logging

Each iteration appends an entry to `output/auto_eval_log.json`:

```json
{
  "iteration": 5,
  "change": "Adjusted conf_hard from 0.85 → 0.9",
  "metrics": {"SHD": 14, "AUPR": 0.29, "edges_pred": 42},
  "delta": {"ΔSHD": -3, "ΔAUPR": +0.07},
  "comment": "Stricter consensus improved structural precision."
}
```

### 6️⃣ Stopping Conditions

Stop iterative tuning when **any** of the following holds:

* SHD < 10
* AUPR improvement < 1% for 3 consecutive runs
* No new improvement in 5 iterations

Otherwise, repeat the loop automatically.

---

## 🧮 Evaluation Metrics

| Metric         | Meaning                                         | Desired Direction |
| -------------- | ----------------------------------------------- | ----------------- |
| **SHD**        | Structural Hamming Distance                     | ↓ lower           |
| **AUPR**       | Area under Precision-Recall                     | ↑ higher          |
| **edges_pred** | Count of predicted edges                        | balanced          |
| **PEHE**       | Precision in Estimation of Heterogeneous Effect | ↓ lower           |

---

## 🧭 Development Priorities (in order)

1. **Correctness** — ensure code executes without exceptions
2. **Causal Soundness** — learned graphs remain DAGs
3. **Performance** — metrics improvement toward ground truth
4. **Interpretability** — changes must be explainable causally
5. **Computational Efficiency** — minimize redundant computation

---

## 🧩 Integration with Hyperparameter Search

When tuning neural estimators (e.g., DragonNet), use:

```bash
python scripts/dragonnet_optuna.py --trials 50
```

Claude may suggest expanding search space or alternative priors when overfitting or variance is detected in PEHE.

---

## 📈 Reporting and Visualization

After each AutoEval loop, generate visual summary:

```bash
python scripts/visualize_auto_eval.py
```

to plot SHD/AUPR progression across iterations.

---

## 🧠 Claude Self-Reflection Mode

Before proposing new code edits, Claude should:

1. Read latest metrics from `output/experiment_summary.csv`
2. Retrieve recent entries from `output/auto_eval_log.json`
3. Hypothesize the most probable causal bottleneck
   (e.g., “too many low-confidence edges → overconnected graph”)
4. Justify any proposed modification with causal reasoning

---

## 🧰 Example Loop (Simulated)

| Iter | Change                | SHD | AUPR | ΔSHD | ΔAUPR        |
| ---- | --------------------- | --- | ---- | ---- | ------------ |
| 1    | baseline              | 43  | 0.12 | —    | —            |
| 2    | conf_hard ↑           | 31  | 0.21 | -12  | +0.09        |
| 3    | λ1 ↑                  | 24  | 0.27 | -7   | +0.06        |
| 4    | soft prior weight ↑   | 15  | 0.30 | -9   | +0.03        |
| 5    | ensemble weight tuned | 9   | 0.32 | -6   | +0.02 ✅ Stop |

---

## ✅ Summary

Claude Code acts as a **closed-loop causal optimizer** for Vortex-Causal:

1. Automatically evaluates → compares → adapts
2. Uses causal reasoning to justify changes
3. Stops when approaching ground truth metrics
4. Logs every step for reproducibility

You, Claude, are responsible for driving this optimization process while maintaining causal and computational integrity.

---
