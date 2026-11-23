# MDM Five-Rung & 2^4 Factorial Ablation Benchmark Report

- **Evidence Bucket (3-Bucket Claim Discipline)**: **Receipt-Grounded Replay** (real empirical model probability/token receipts in `benchmarks/data/*.jsonl` + real decoded PNG/WAV fixtures in `benchmarks/data/fixtures/`, paired with live deterministic execution of MDM calibration, 5-fold OOF Mondrian/RAPS conformal prediction, dual-buffer visual compilation, submodular keyframe selection, untrusted ASR fencing, context knapsack pruning, polynomial alpha-spending, NEVC loop gating, and primal-dual Pareto contextual bandit routing; zero live cloud GPU traffic claimed).
- **Primary Headline Tier**: `tier2` (`360 D_cal` / `240 D_val` / `600 D_test` episodes = `3600` locked `D_test` turn decisions; manifest SHA-256 `0bcfd508bd737c84...`).
- **Lockfile Path**: `benchmarks/locks/mdm_episodebench_v1.lock.json`

## 1. Five-Rung 1-Factor-Per-Step Cumulative Ladder (`R0`, `R1 -> R2 -> R3 -> R3.5 -> R4`)

| Regime | Bit Code `(C,M,K,P)` | Episode Success | Turn Acc | 4-Stage Pass | Brier | ECE | Conformal Cov | Mean Tokens/Ep | Mean USD/Ep | Escalation | p95 Latency (ms) | Net Utility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R0` | `R0` | 44.67% | 76.39% | 0.00% | 0.2881 | 0.0629 | 100.00% | 22687 | $0.0482 | 100.0% | 2957.9 | -0.4208 |
| `R1_0000` | `0000` | 0.83% | 28.72% | 0.00% | 1.1203 | 0.5029 | 18.11% | 22687 | $0.0091 | 21.1% | 1278.6 | -0.1554 |
| `R2_KI_C_1000` | `1000` | 16.00% | 57.61% | 0.00% | 0.5730 | 0.1325 | 86.36% | 22574 | $0.0276 | 77.8% | 2146.4 | -0.3368 |
| `R3_SYN_CM_1100` | `1100` | 25.67% | 67.14% | 0.00% | 0.4868 | 0.0865 | 74.28% | 19743 | $0.0193 | 59.2% | 1822.0 | -0.0900 |
| `R3_5_LOO_noP_1110` | `1110` | 61.50% | 83.47% | 59.00% | 0.2878 | 0.0920 | 89.72% | 10993 | $0.0105 | 52.8% | 1692.7 | 0.4267 |
| `R4_1111` | `1111` | 62.33% | 83.78% | 58.33% | 0.2830 | 0.0919 | 89.72% | 11225 | $0.0076 | 30.9% | 1443.8 | 0.4870 |

- **`R4 (1111)` vs `R0` Generative Ceiling**: Cost reduced by **84.3%** (`$0.0076` vs `$0.0482`/episode); turn-level Tango Berger-Boos non-inferiority (`margin = 1.0%`): `delta_hat = +7.39%`, `z = 10.47`, `p_BB = 1.0000e-04`, `non_inferior = True`.
- **5-Fold OOF vs In-Sample Resubstitution Diagnostic (`R4`)**: 5-fold OOF marginal coverage = `89.72%` (worst-stratum Mondrian gap `1.05%`) vs in-sample resubstituted coverage = `89.69%`.

## 2. Single-Factor Knock-Ins (`KI_*`) and Leave-One-Out Knockouts (`LOO_*`)

| Slice | Regime | Bit Code | Episode Success | Δ vs Reference | Mean USD/Ep | Net Utility | Δ Net Utility |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Knock-In vs `R1` | `KI_C` | `1000` | 16.00% | +15.17% | $0.0276 | -0.3368 | -0.1814 |
| Knock-In vs `R1` | `KI_M` | `0100` | 2.00% | +1.17% | $0.0068 | -0.1031 | +0.0522 |
| Knock-In vs `R1` | `KI_K` | `0010` | 6.00% | +5.17% | $0.0070 | -0.0661 | +0.0893 |
| Knock-In vs `R1` | `KI_P` | `0001` | 6.83% | +6.00% | $0.0132 | -0.1696 | -0.0142 |
| LOO vs `R4` | `LOO_noC` | `0111` | 47.33% | -15.00% | $0.0076 | 0.3368 | -0.1502 |
| LOO vs `R4` | `LOO_noM` | `1011` | 36.00% | -26.33% | $0.0052 | 0.2662 | -0.2208 |
| LOO vs `R4` | `LOO_noK` | `1101` | 27.67% | -34.67% | $0.0129 | 0.0447 | -0.4423 |
| LOO vs `R4` | `LOO_noP` | `1110` | 61.50% | -0.83% | $0.0105 | 0.4267 | -0.0603 |

## 3. Exact Shapley Attribution & Yates/ANOVA Interaction Effects (`B = 2,000` Cluster Bootstrap)

| Effect / Factor | Net Utility Estimate | 95% Cluster-Bootstrap CI | Holm-Adjusted p | Episode Success Estimate | 95% Success CI | Significant (`FWER <= 0.05`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Main Effect `tau_C` | `+0.0962` | `[+0.0652, +0.1273]` | `1.273e-08` | `+18.00%` | `[+14.69%, +21.29%]` | `True` |
| Main Effect `tau_M` | `+0.2038` | `[+0.1800, +0.2291]` | `9.832e-57` | `+17.38%` | `[+15.21%, +19.58%]` | `True` |
| Main Effect `tau_K` | `+0.3132` | `[+0.2893, +0.3380]` | `3.094e-142` | `+22.96%` | `[+20.67%, +25.42%]` | `True` |
| Main Effect `tau_P` | `+0.1455` | `[+0.1221, +0.1698]` | `6.661e-33` | `+7.83%` | `[+5.52%, +9.96%]` | `True` |
| Interaction `tau_CM` | `+0.0677` | `[+0.0329, +0.1026]` | `2.657e-04` | `+7.50%` | `[+3.88%, +11.08%]` | `True` |
| Interaction `tau_CK` | `+0.1521` | `[+0.1157, +0.1865]` | `4.439e-15` | `+10.17%` | `[+6.46%, +13.88%]` | `True` |
| Interaction `tau_CP` | `+0.1104` | `[+0.0535, +0.1644]` | `2.657e-04` | `-15.08%` | `[-20.21%, -10.00%]` | `True` |
| Interaction `tau_MK` | `+0.1681` | `[+0.1348, +0.2011]` | `1.667e-22` | `+14.92%` | `[+11.75%, +18.00%]` | `True` |
| Interaction `tau_MP` | `-0.0686` | `[-0.0964, -0.0387]` | `1.291e-05` | `+5.50%` | `[+2.71%, +8.50%]` | `True` |
| Interaction `tau_KP` | `+0.0314` | `[+0.0056, +0.0592]` | `2.416e-02` | `+7.17%` | `[+4.50%, +9.92%]` | `True` |
| Exact Shapley `phi_C` | `+0.0589` | `[+0.0286, +0.0891]` | `1.716e-04` | `+17.03%` | `[+13.78%, +20.32%]` | `True` |
| Exact Shapley `phi_M` | `+0.1813` | `[+0.1582, +0.2037]` | `1.079e-54` | `+16.17%` | `[+14.06%, +18.19%]` | `True` |
| Exact Shapley `phi_K` | `+0.2974` | `[+0.2733, +0.3219]` | `5.750e-126` | `+21.94%` | `[+19.61%, +24.35%]` | `True` |
| Exact Shapley `phi_P` | `+0.1047` | `[+0.0835, +0.1272]` | `5.371e-21` | `+6.36%` | `[+4.28%, +8.19%]` | `True` |

- **Shapley Efficiency Identity**: `sum(phi) = +0.642380` vs `U(1111) - U(0000) = +0.642380` (residual `0.00e+00`).

## 4. Mandatory Iso-Escalation & Iso-Budget Controls (`C1..C5` vs `R4`)

| Control Regime | Episode Success | Turn Acc | Escalation Rate | Mean Tokens/Ep | Mean USD/Ep | Net Utility | Exact McNemar p (vs `R4`) | Holm p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `C1_RandomEscalation` | 30.33% | 68.72% | 30.89% | 11749 | $0.0072 | 0.1745 | `7.146e-32` | `4.288e-31` |
| `C2_MatchedRateEntropy` | 25.67% | 65.78% | 30.33% | 11253 | $0.0066 | 0.1377 | `1.135e-40` | `7.942e-40` |
| `C3_RecencyWindow` | 24.50% | 64.61% | 42.72% | 11435 | $0.0095 | 0.0737 | `3.930e-59` | `3.144e-58` |
| `C4_UniformFrameStride` | 50.00% | 73.94% | 18.25% | 11911 | $0.0059 | 0.3936 | `2.547e-15` | `1.019e-14` |
| `C5_PlaceboNEVC` | 58.00% | 82.19% | 30.89% | 11248 | $0.0076 | 0.4436 | `4.095e-04` | `8.191e-04` |

## 5. Complete `2^4 = 16` Factorial Cube (`0000` .. `1111`)

| Bit Code | Canonical Label | Episode Success | Turn Acc | Brier | ECE | Mean Tokens/Ep | Mean USD/Ep | Escalation | Net Utility |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0000` | `R1_0000` | 0.83% | 28.72% | 1.1203 | 0.5029 | 22687 | $0.0091 | 21.1% | -0.1554 |
| `0001` | `KI_P_0001` | 6.83% | 45.81% | 0.8083 | 0.2720 | 22687 | $0.0132 | 28.0% | -0.1696 |
| `0010` | `KI_K_0010` | 6.00% | 46.86% | 0.8422 | 0.3171 | 14417 | $0.0070 | 24.7% | -0.0661 |
| `0011` | `SYN_KP_0011` | 22.00% | 61.56% | 0.6192 | 0.2075 | 14905 | $0.0089 | 27.1% | 0.0596 |
| `0100` | `KI_M_0100` | 2.00% | 40.03% | 0.9466 | 0.3864 | 19900 | $0.0068 | 16.5% | -0.1031 |
| `0101` | `SYN_MP_0101` | 17.50% | 59.69% | 0.5992 | 0.1670 | 19900 | $0.0118 | 28.4% | -0.0378 |
| `0110` | `SYN_MK_0110` | 23.33% | 63.89% | 0.6061 | 0.2513 | 11658 | $0.0045 | 16.2% | 0.1523 |
| `0111` | `LOO_noC_0111` | 47.33% | 74.61% | 0.4448 | 0.1678 | 11881 | $0.0076 | 28.2% | 0.3368 |
| `1000` | `R2_KI_C_1000` | 16.00% | 57.61% | 0.5730 | 0.1325 | 22574 | $0.0276 | 77.8% | -0.3368 |
| `1001` | `SYN_CP_1001` | 9.50% | 49.17% | 0.6188 | 0.0836 | 22687 | $0.0055 | 0.1% | -0.0033 |
| `1010` | `SYN_CK_1010` | 31.17% | 69.00% | 0.4283 | 0.0746 | 13859 | $0.0178 | 77.9% | -0.0082 |
| `1011` | `LOO_noM_1011` | 36.00% | 69.44% | 0.4602 | 0.1033 | 17092 | $0.0052 | 0.9% | 0.2662 |
| `1100` | `R3_SYN_CM_1100` | 25.67% | 67.14% | 0.4868 | 0.0865 | 19743 | $0.0193 | 59.2% | -0.0900 |
| `1101` | `LOO_noK_1101` | 27.67% | 70.44% | 0.4234 | 0.0609 | 19836 | $0.0129 | 32.0% | 0.0447 |
| `1110` | `R3_5_LOO_noP_1110` | 61.50% | 83.47% | 0.2878 | 0.0920 | 10993 | $0.0105 | 52.8% | 0.4267 |
| `1111` | `R4_1111` | 62.33% | 83.78% | 0.2830 | 0.0919 | 11225 | $0.0076 | 30.9% | 0.4870 |

## 6. Tier-1 Fast Screening Summary (`400` Episodes: `120 D_cal / 80 D_val / 200 D_test`)

- **Tier-1 `R1 (0000)` -> `R4 (1111)`**: Episode Success `0.00%` -> `50.00%`, Turn Accuracy `22.83%` -> `78.67%`, Mean Cost `$0.0065` -> `$0.0094`/episode, Net Utility `-0.1178` -> `0.3308` (Shapley sum `+0.4486`).
