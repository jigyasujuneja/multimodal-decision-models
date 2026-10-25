# Multi-Model Head-to-Head Calibration and Conformal Benchmark

This report compares 10 real decision and frontier models plus 5-fold out-of-fold post-hoc calibration and pre-fusion Split Conformal RAPS cascades across two evaluation panels:

1. **Multi-Task Calibration Panel (`N = 50` items, `11` clusters, `K` from 2 to 26)**: evaluates parallel masked diffusion (`/v1/systemone`), autoregressive verbalized confidence, and 3-draw symmetric Dirichlet (`alpha_0 = 0.5`) black-box consensus models.
2. **XSTest Safety Refusal Boundary Panel (`N = 360` items, `12` prompt-type clusters, `K = 2`)**: evaluates embedding probes (`Classy v0.3.3`, `Classier v0.2.0`), compact 2B deciders (`Strands Decider 2B v19`), and masked diffusion verifiers (`dgem v0.3.1`, `DecisionGemma`).

## Panel 1: Multi-Task Calibration Leaderboard (`N = 50`)

| Arm | Acc | Brier (95% Bootstrap CI) | Murphy REL / RES | NLL | ECE | ACE | AURC | Conformal Cov@0.05 / @0.10 | Mean Set@0.10 | McNemar p (vs Ref) | Mean Latency (ms) | Pareto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Gemini 3.8 Flash (Verbalized + TempScaled OOF)` | 0.920 | 0.1419 ([0.0246, 0.2633]) | 0.0555 / 0.8320 | 0.3404 | 0.0231 | 0.0021 | 0.0177 | 0.960 / 0.920 | 1.00 | 1.0000 | 4247.8 | yes |
| `Gemini 3.8 Flash (Verbalized Conf)` | 0.920 | 0.1452 ([0.0102, 0.2774]) | 0.0424 / 0.8124 | 0.3531 | 0.0484 | 0.0018 | 0.0191 | 0.960 / 0.920 | 1.02 | 1.0000 | 4247.8 | no |
| `DecisionGemma` | 0.920 | 0.1796 ([0.0833, 0.2887]) | 0.1326 / 0.8712 | 0.3194 | 0.1364 | 0.0029 | 0.0204 | 0.960 / 0.920 | 1.04 | 1.0000 | 278.3 | yes |
| `dgem v0.3.7 (TempScaled OOF)` | 0.860 | 0.2047 ([0.0638, 0.3326]) | 0.1307 / 0.8477 | 0.4534 | 0.0951 | 0.0035 | 0.1175 | 0.960 / 0.920 | 1.14 | 0.4531 | 389.6 | no |
| `dgem v0.3.7` | 0.860 | 0.2071 ([0.0542, 0.3463]) | 0.0976 / 0.8058 | 0.4778 | 0.0973 | 0.0031 | 0.0996 | 0.960 / 0.920 | 1.14 | 0.4531 | 389.6 | no |
| `Conformal Cascade: dgem v0.3.7 -> Gemini 3.8 Flash (Dirichlet M=3)` | 0.880 | 0.2340 ([0.0714, 0.3823]) | 0.1156 / 0.7962 | 0.5597 | 0.0972 | 0.0032 | 0.0963 | 0.960 / 0.920 | 1.82 | 0.6875 | 8577.8 | no |
| `Conformal Cascade: Strands Decider 2B v19 -> DecisionGemma` | 0.860 | 0.2585 ([0.1511, 0.3592]) | 0.1734 / 0.8321 | 0.4624 | 0.1866 | 0.0039 | 0.0536 | 0.960 / 0.920 | 1.30 | 0.2500 | 390.8 | no |
| `Gemini 3.6 Flash (Dirichlet M=3)` | 0.920 | 0.2624 ([0.1222, 0.4240]) | 0.1518 / 0.8120 | 0.6289 | 0.2625 | 0.0064 | 0.0410 | 0.960 / 0.920 | 1.00 | 1.0000 | 62709.9 | no |
| `Gemini 3.8 Flash (Dirichlet M=3)` | 0.900 | 0.2791 ([0.1373, 0.4371]) | 0.1662 / 0.8120 | 0.6678 | 0.2617 | 0.0064 | 0.0485 | 0.960 / 0.920 | 2.22 | 1.0000 | 56613.1 | no |
| `Gemini 3.5 Flash Lite (Dirichlet M=3)` | 0.840 | 0.3404 ([0.2004, 0.4839]) | 0.1846 / 0.7693 | 0.7474 | 0.2058 | 0.0072 | 0.0798 | 0.940 / 0.920 | 2.92 | 0.2891 | 62790.4 | no |
| `Strands Decider 2B v19` | 0.720 | 0.4249 ([0.3046, 0.5101]) | 0.2613 / 0.7502 | 0.8787 | 0.1524 | 0.0070 | 0.2398 | 0.960 / 0.880 | 2.36 | 0.0213 | 163.0 | yes |

**Panel 1 Brier-Latency Pareto Frontier**: `DecisionGemma`, `Strands Decider 2B v19`, `Gemini 3.8 Flash (Verbalized + TempScaled OOF)`.

## Panel 2: XSTest Safety Boundary Leaderboard (`N = 360`)

| Arm | Acc | Macro-F1 | Brier (95% Bootstrap CI) | Murphy REL / RES | NLL | ECE | ACE | AURC | Conformal Cov@0.05 / @0.10 | Mean Set@0.10 | McNemar p (vs Ref) | Latency (ms) | Pareto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `DecisionGemma` | 0.9694 | 0.9692 | 0.0465 ([0.0106, 0.1005]) | 0.0128 / 0.4604 | 0.0845 | 0.0209 | 0.0328 | 0.0020 | 0.9694 / 0.9694 | 1.00 | 1.0000 | 175.8 | yes |
| `dgem v0.3.1` | 0.9639 | 0.9633 | 0.0565 ([0.0225, 0.1021]) | 0.0229 / 0.4583 | 0.1001 | 0.0204 | 0.0300 | 0.0025 | 0.9639 / 0.9639 | 1.00 | 0.8036 | 215.1 | no |
| `Conformal Cascade: Classy v0.3.3 -> DecisionGemma` | 0.8833 | 0.8793 | 0.1634 ([0.1187, 0.2273]) | 0.0075 / 0.3379 | 0.2453 | 0.0284 | 0.0358 | 0.0245 | 0.9444 / 0.8972 | 1.04 | 3.35e-07 | 866.8 | no |
| `Conformal Cascade: Classier v0.2.0 -> DecisionGemma` | 0.8917 | 0.8913 | 0.1720 ([0.1214, 0.2316]) | 0.0164 / 0.3386 | 0.2860 | 0.0270 | 0.0604 | 0.0326 | 0.9444 / 0.9083 | 1.05 | 2.46e-07 | 537.1 | no |
| `Classier v0.2.0` | 0.7889 | 0.7886 | 0.2986 ([0.2571, 0.3439]) | 0.0303 / 0.2241 | 0.4581 | 0.0273 | 0.1022 | 0.0965 | 0.9417 / 0.9111 | 1.31 | 2.44e-16 | 482.4 | no |
| `Strands Decider 2B v19 (VectorScaled OOF)` | 0.8083 | 0.8027 | 0.3023 ([0.2123, 0.3884]) | 0.0240 / 0.2137 | 0.4761 | 0.0805 | 0.0936 | 0.1181 | 0.9500 / 0.9000 | 1.30 | 4.33e-12 | 153.9 | yes |
| `Strands Decider 2B v19` | 0.7583 | 0.7339 | 0.3623 ([0.2680, 0.4420]) | 0.0834 / 0.2106 | 0.5445 | 0.0907 | 0.1715 | 0.1489 | 0.9500 / 0.9000 | 1.44 | 3.21e-16 | 153.9 | no |
| `Classy v0.3.3 (VectorScaled OOF)` | 0.6444 | 0.6218 | 0.4582 ([0.3865, 0.5176]) | 0.0166 / 0.0508 | 0.6520 | 0.0620 | 0.0729 | 0.3306 | 0.9528 / 0.9000 | 1.69 | 1.54e-27 | 746.2 | no |
| `Classy v0.3.3` | 0.6444 | 0.6107 | 0.4634 ([0.4107, 0.5071]) | 0.0151 / 0.0411 | 0.6563 | 0.0785 | 0.0815 | 0.3297 | 0.9528 / 0.9000 | 1.69 | 1.54e-27 | 746.2 | no |

**Panel 2 Brier-Latency Pareto Frontier**: `DecisionGemma`, `Strands Decider 2B v19 (VectorScaled OOF)`.

## Conformal Cascade Escalation Analysis

- **`Conformal Cascade: dgem v0.3.7 -> Gemini 3.8 Flash (Dirichlet M=3)` (`multimodel_calibration_50`)**: escalates `14.0%` of items where Stage-1 RAPS (`alpha = 0.10`) produces a non-singleton set, reaching accuracy `0.8800`, Brier `0.2340`, AURC `0.0963`, and discordant Stage-1 wins/losses `1/0` (exact two-sided McNemar `p = 1.0000`) at `8577.8 ms` mean latency.
- **`Conformal Cascade: Strands Decider 2B v19 -> DecisionGemma` (`multimodel_calibration_50`)**: escalates `80.0%` of items where Stage-1 RAPS (`alpha = 0.10`) produces a non-singleton set, reaching accuracy `0.8600`, Brier `0.2585`, AURC `0.0536`, and discordant Stage-1 wins/losses `10/3` (exact two-sided McNemar `p = 0.0923`) at `390.8 ms` mean latency.
- **`Conformal Cascade: Classy v0.3.3 -> DecisionGemma` (`safety_xstest_360`)**: escalates `68.6%` of items where Stage-1 RAPS (`alpha = 0.10`) produces a non-singleton set, reaching accuracy `0.8833`, Brier `0.1634`, AURC `0.0245`, and discordant Stage-1 wins/losses `91/5` (exact two-sided McNemar `p = 1.63e-21`) at `866.8 ms` mean latency.
- **`Conformal Cascade: Classier v0.2.0 -> DecisionGemma` (`safety_xstest_360`)**: escalates `31.1%` of items where Stage-1 RAPS (`alpha = 0.10`) produces a non-singleton set, reaching accuracy `0.8917`, Brier `0.1720`, AURC `0.0326`, and discordant Stage-1 wins/losses `39/2` (exact two-sided McNemar `p = 7.84e-10`) at `537.1 ms` mean latency.

## Reproducibility Command

```bash
PYTHONPATH=src python3 benchmarks/run_multimodel_benchmark.py
```
