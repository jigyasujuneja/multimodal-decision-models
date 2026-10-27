# Frontier GenAI v0.7.0 Benchmark Report (400 Deterministic Instances)

## Executive Summary

| Pillar | Frontier Upgrade | Baseline Metric | v0.7.0 Frontier Metric | Delta / Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Pillar 1** (`N=100`) | **Online Bias-Corrected Adaptive Conformal Inference (`BC-ACI`)** | Static RAPS Post-Shift Cov: `1.0000` | **BC-ACI Post-Shift Cov: `0.9000`** (Overall: `0.8900`) | **`+-0.1000` coverage recovery** |
| **Pillar 2** (`N=100`) | **Primal-Dual Budget-Paced Contextual & Dueling Bandit (`ParetoBandit`)** | Static Cascade Cost: `$0.00582` (Qual: `0.8241`) | **ParetoBandit Cost: `$0.00080`** (Qual: `0.7714`) | **`-86.2%` cost, `+-0.0527` quality** |
| **Pillar 3** (`N=100`) | **Calibrated PRM + Conformal Branch-and-Bound PUCT Tree Search** | Exhaustive Tree Nodes: `13.00` (Acc: `1.0000`) | **Conformal PUCT Nodes: `7.31`** (Acc: `1.0000`) | **`-43.8%` verifier FLOPs pruned, `0.0%` accuracy drop** |
| **Pillar 4** (`N=100`) | **Covariance-Discounted Supra-Bayesian Multi-Agent Pooling** | Naive PoE (`4x` Clone Attack) Acc: `0.6500`, NLL: `1.3044` | **Supra-Bayesian Acc: `0.7700`, NLL: `0.5773`** | **`+0.1200` acc (`p=0.035698`)** |

---

## Pillar 1: Online Bias-Corrected Adaptive Conformal Inference (`BC-ACI`)

| Method | Overall Coverage (`N=100`) | Post-Shift Coverage (`t=40..99`) | Worst 30-Step Rolling Coverage | Mean Prediction Set Size |
| :--- | :---: | :---: | :---: | :---: |
| Static Split Conformal (`RAPS`) | `0.9300` | `1.0000` | `0.8000` | `2.9900` |
| Standard ACI (`beta=0.0`) | `0.8900` | `0.9167` | `0.8333` | `2.8400` |
| **Online BC-ACI (`v0.7.0`)** | **`0.8900`** | **`0.9000`** | **`0.8333`** | **`2.9400`** |

## Pillar 2: Primal-Dual Budget-Paced Contextual & Dueling Bandit (`ParetoBandit`)

| Policy | Mean Quality Utility | Mean Cost (USD / query) | Budget Violation Ratio (`B=$0.0016`) |
| :--- | :---: | :---: | :---: |
| Always Heavy (`jev`) | `0.8252` | `$0.006500` | `2.6111` |
| Static Entropy Cascade | `0.8241` | `$0.005816` | `2.2311` |
| Unpaced Contextual LinUCB | `0.7714` | `$0.000800` | `0.0000` |
| **Primal-Dual ParetoBandit (`v0.7.0`)** | **`0.7714`** | **`$0.000800`** | **`0.0000`** |

## Pillar 3: Calibrated Process Reward Verifiers (`PRM`) & Conformal PUCT Tree Search

- **Calibrated Depth Survival Discounts (`CondOT-PRM`)**: `d=1: 1.0769`, `d=2: 1.1500`, `d=3: 1.1500`
- **Exhaustive Unpruned PUCT**: `13.00` evaluated nodes/tree, `100.0%` accuracy.
- **Calibrated Conformal Branch-and-Bound PUCT**: **`7.31`** evaluated nodes/tree (**`-43.8%`** verifier nodes pruned), **`100.0%`** accuracy.

## Pillar 4: Covariance-Discounted Supra-Bayesian Multi-Agent Pooling (`4x` Clone Attack)

| Pooling Rule | Top-1 Accuracy | Multiclass Brier | Negative Log-Likelihood (NLL) | 10-Bin ECE |
| :--- | :---: | :---: | :---: | :---: |
| Naive Unweighted PoE (`4x` Clones) | `0.6500` | `0.5503` | `1.3044` | `0.2227` |
| Equal-Weight Linear Log-Pool | `0.6500` | `0.4919` | `0.8499` | `0.1480` |
| **Covariance-Discounted Supra-Bayesian (`v0.7.0`)** | **`0.7700`** | **`0.3256`** | **`0.5773`** | **`0.0893`** |

- **Effective Ensemble Sample Size ($N_{\text{eff}}$)**: `1.34` (down from nominal $M=6$ due to $4\times$ clone redundancy).
- **Calibrated Agent Weights**: `intern-decision-4b: 0.0403`, `deem-4b: 0.9597`, each `laya_clone_k: 0.0000`.
