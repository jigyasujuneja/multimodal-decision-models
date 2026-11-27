# `v1.2.0` Research Science Benchmark Report (`PriDe + SLD`, Conformal Risk Control `CRC`, Bayesian `VoI`, and Epistemic Abstention)

This report documents the measured evaluation of the `multimodal-decision-models` `v1.2.0` mathematical primitives across **17 distinct models** on locked real-receipt datasets (`N = 500` `DecisionBench`, `N = 50` multi-task calibration, and `N = 360` `XSTest` safe-homonym vs. unsafe prompt guardrails).

## 1. Mathematical Summary (`v1.2.0`)

1. **Null-Prior Debiasing (`debias_null_prior_probabilities`) and Unlabeled `SLD` Shift Adaptation (`adapt_sld_label_shift_posteriors`)**:
   - Removes zero-evidence token-surface bias with $p_k^{\text{debiased}}(x) \propto p_k(x) / (p_k(\varnothing))^\gamma$ before 5-fold out-of-fold (`OOF`) Platt/vector scaling.
   - Adapts calibrated source posteriors to unlabeled target streams by using Saerens-Latinne-Decaestecker (`SLD`) EM fixed-point iteration $\hat{\pi}^{(t+1)}_k = \frac{1}{m}\sum_{j=1}^m \frac{(\hat{\pi}^{(t)}_k / \pi^{\text{src}}_k) p_k(x_j)}{\sum_c (\hat{\pi}^{(t)}_c / \pi^{\text{src}}_c) p_c(x_j)}$.
2. **Finite-Sample Conformal Risk Control (`calibrate_conformal_risk_control`)**:
   - For any monotone bounded loss $L_i(\lambda) \in [0, B]$ non-increasing in threshold $\lambda \in [0, 1]$, selects
     $$\hat{\lambda} = \inf\left\{\lambda \in [0, 1] : \frac{n}{n+1}\hat{R}_n(\lambda) + \frac{B}{n+1} \le \alpha\right\}$$
     guaranteeing $\mathbb{E}[L_{n+1}(\hat{\lambda})] \le \alpha$ in finite samples without distributional assumptions.
3. **Closed-Form Bayesian Value of Information (`compute_bayes_value_of_information`) and Epistemic Abstention (`select_omniscience_action`)**:
   - Escalates from Stage 1 to Stage 2 strictly when expected Bayes risk reduction exceeds normalized Stage-2 marginal latency/token cost: $\text{VoI}(x) > c_{\text{S2}}$.
   - Under $+R$ reward for exact correctness, $0$ for abstention, and $-C$ penalty for wrong answers, abstains whenever $\max_k \hat{p}_k(x) < p^* = \frac{C}{R + C}$ (`0.50` for symmetric $+1 / 0 / -1$ scoring).

## 2. `DecisionBench` (`N = 500`) `OOF` Calibration and `CRC` Selective Abstention

| Model | Raw Accuracy | Raw `ECE` / Brier | `OOF` `ECE` / Brier | `CRC` $\alpha = 0.05$ Empirical Risk | `CRC` $\alpha = 0.05$ Selective Acc (Coverage) | `CRC` $\alpha = 0.10$ Empirical Risk | `CRC` $\alpha = 0.10$ Selective Acc (Coverage) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `intern-decision-4b` | `80.80%` | `0.1021 / 0.3021` | `0.0612 / 0.2853` | `0.0540` | `90.04%` (`54.2%`) | `0.1020` | `86.72%` (`76.8%`) |
| `deem-4b` | `79.40%` | `0.1339 / 0.3237` | `0.0175 / 0.2961` | `0.0520` | `91.33%` (`60.0%`) | `0.0920` | `87.63%` (`74.4%`) |
| `wahler-4b` | `77.80%` | `0.1449 / 0.3472` | `0.0290 / 0.3183` | `0.0480` | `91.02%` (`53.4%`) | `0.0980` | `86.39%` (`72.0%`) |
| `jev` | `72.00%` | `0.0419 / 0.3853` | `0.1207 / 0.4136` | `0.0520` | `87.68%` (`42.2%`) | `0.0960` | `82.92%` (`56.2%`) |
| `julia-onnx` | `71.00%` | `0.1934 / 0.4494` | `0.0507 / 0.4027` | `0.0600` | `84.77%` (`39.4%`) | `0.0980` | `82.37%` (`55.6%`) |

## 3. Closed-Form `Bayes VoI` Cascade Economics (`N = 500`)

Unlike static entropy thresholds that ignore downstream error stakes, `compute_bayes_value_of_information` scales Stage-2 escalation monotonically with decision stakes $C_{\text{err}} \in \{0.05, 0.25, 1.00, 5.00\}$:

| Stage-1 → Stage-2 Pair | Stage-1 Standalone Acc | $C_{\text{err}} = 0.05$ Acc (Esc Rate) | $C_{\text{err}} = 0.25$ Acc (Esc Rate) | $C_{\text{err}} = 1.00$ Acc (Esc Rate) | $C_{\text{err}} = 5.00$ Acc (Esc Rate) | Mean Total Cost vs. Stage-1 Only ($C_{\text{err}} = 5.00$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `wahler-4b -> deem-4b` | `77.80%` | `77.80%` (`0.6%`) | `79.60%` (`34.6%`) | `79.60%` (`45.2%`) | `79.60%` (`47.2%`) | `$1.029` vs. `$1.110` (`-7.3%`) |
| `julia-onnx -> deem-4b` | `71.00%` | `72.60%` (`7.4%`) | `77.80%` (`48.6%`) | `78.60%` (`56.6%`) | `78.80%` (`58.6%`) | `$1.072` vs. `$1.450` (`-26.1%`) |

## 4. Asymmetric Safety `CRC` (`alpha = 0.02`) and Unlabeled `SLD` Shift on `XSTest` (`N = 360`)

| Safety Guardrail Arm | Overall Accuracy | `ECE` / Brier | `CRC` $\alpha = 0.02$ Realized False-Allow Risk | Safe-Homonym Auto-Allow Rate | `SLD` `85%`-Safe Stream Brier (`Unadapted -> Adapted`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `dgem v0.3.1 (OOF Vector-Scaled)` | `97.78%` | `0.0139 / 0.0369` | `1.94%` ($\le 2.0\%$) | `98.50%` | `0.0520 -> 0.0596` ($\hat{\pi} = 0.838$) |
| `DecisionGemma (OOF Vector-Scaled)` | `97.22%` | `0.0118 / 0.0363` | `1.67%` ($\le 2.0\%$) | `98.50%` | `0.1697 -> 0.1263` ($\hat{\pi} = 0.768$) |
| `Strands Decider 2B v19 (OOF Vector-Scaled)` | `81.39%` | `0.0335 / 0.2813` | `1.67%` ($\le 2.0\%$) | `28.50%` | `0.4506 -> 0.3974` (`20%` surge) |
| `Classier v0.2.0 (OOF Vector-Scaled)` | `81.11%` | `0.0174 / 0.2761` | `1.94%` ($\le 2.0\%$) | `32.50%` | `0.3201 -> 0.1703` (`-46.8%` Brier) |

Pairing `Strands Decider 2B v19 (OOF Vector-Scaled)` with `dgem v0.3.1 (OOF Vector-Scaled)` under `compute_bayes_value_of_information` with $10:1$ asymmetric unsafe-vs-safe error cost cuts mean asymmetric safety loss by **`91.5%`** (`0.5611` down to `0.0478`).

## 5. Operational Interpretation, Trade-Offs, and Limitations

- **Operational interpretation**: In practice, `PriDe` + `SLD` fixes prior drift before thresholding, `CRC` bounds worst-case selective error risk on finite calibration splits, and `Bayes VoI` + `DomainCompetenceProfile` escalates only when Stage 2 reduces expected dollar-weighted loss instead of degrading Stage-1 decisions.
- **Rejected alternative and trade-off**: Fixed entropy cutoffs and unconditional Stage-2 overrides were rejected because they ignore asymmetric error stakes and cause negative transfer when Stage 2 trails Stage 1 on typed schemas; accepting a 5-fold calibration step requires `N >= 40` labeled examples per domain.
- **Limitations and non-goals**: `SLD` unlabeled shift adaptation assumes class-conditional likelihood invariance $p(x \mid y)$ (label shift instead of covariate drift), and `CRC` finite-sample guarantees require exchangeable calibration and test requests within each stratum.

