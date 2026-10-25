# Multi-Model Head-to-Head Calibration and Conformal Benchmark

This report compares 10 real decision and frontier models plus 5-fold out-of-fold post-hoc calibration and pre-fusion Split Conformal RAPS cascades across two evaluation panels:

1. **Multi-Task Calibration Panel (`N = 50` items, `11` clusters, `K` from 2 to 26)**: evaluates parallel masked diffusion (`/v1/systemone`), autoregressive verbalized confidence, and 3-draw symmetric Dirichlet (`alpha_0 = 0.5`) black-box consensus models.
2. **XSTest Safety Refusal Boundary Panel (`N = 360` items, `12` prompt-type clusters, `K = 2`)**: evaluates embedding probes (`Classy v0.3.3`, `Classier v0.2.0`), compact 2B deciders (`Strands Decider 2B v19`), and masked diffusion verifiers (`dgem v0.3.1`, `DecisionGemma`).

## Panel 1: Multi-Task Calibration Leaderboard (`N = 50`)

| Arm | Acc | Brier (95% Bootstrap CI) | Murphy REL / RES | NLL | ECE | ACE | AURC | Conformal Cov@0.05 / @0.10 | Mean Set@0.10 | McNemar p (vs Ref) | Mean Latency (ms) | Pareto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Gemini 3.8 Flash (Verbalized + TempScaled OOF)` | 0.920 | 0.1419 ([0.0246, 0.2633]) | 0.0576 / 0.8365 | 0.3404 | 0.0231 | 0.0369 | 0.0177 | 0.960 / 0.920 | 1.00 | 1.0000 | 4247.8 | yes |
| `Gemini 3.8 Flash (Verbalized Conf)` | 0.920 | 0.1452 ([0.0102, 0.2774]) | 0.0440 / 0.8196 | 0.3531 | 0.0484 | 0.0305 | 0.0191 | 0.960 / 0.920 | 1.02 | 1.0000 | 4247.8 | no |
| `DecisionGemma` | 0.920 | 0.1796 ([0.0833, 0.2887]) | 0.1342 / 0.8753 | 0.3194 | 0.1364 | 0.0545 | 0.0204 | 0.960 / 0.920 | 1.04 | 1.0000 | 278.3 | yes |
| `dgem v0.3.7 (TempScaled OOF)` | 0.860 | 0.2047 ([0.0638, 0.3326]) | 0.1327 / 0.8488 | 0.4534 | 0.0951 | 0.0555 | 0.1175 | 0.960 / 0.920 | 1.14 | 0.4531 | 389.6 | no |
| `dgem v0.3.7` | 0.860 | 0.2071 ([0.0542, 0.3463]) | 0.0988 / 0.8125 | 0.4778 | 0.0973 | 0.0444 | 0.0996 | 0.960 / 0.920 | 1.14 | 0.4531 | 389.6 | no |
| `Conformal Cascade: dgem v0.3.7 -> Gemini 3.8 Flash (Dirichlet M=3)` | 0.880 | 0.2340 ([0.0714, 0.3823]) | 0.1195 / 0.8063 | 0.5597 | 0.0972 | 0.0512 | 0.0963 | 0.960 / 0.920 | 2.30 | 0.6875 | 8577.8 | no |
| `Conformal Cascade: Strands Decider 2B v19 -> DecisionGemma` | 0.860 | 0.2585 ([0.1511, 0.3592]) | 0.1753 / 0.8376 | 0.4624 | 0.1866 | 0.0670 | 0.0536 | 0.960 / 0.920 | 1.30 | 0.2500 | 390.8 | no |
| `Gemini 3.6 Flash (Dirichlet M=3)` | 0.920 | 0.2624 ([0.1222, 0.4240]) | 0.1637 / 0.8220 | 0.6289 | 0.2625 | 0.1036 | 0.0410 | 0.960 / 0.920 | 1.00 | 1.0000 | 62709.9 | no |
| `Gemini 3.8 Flash (Dirichlet M=3)` | 0.900 | 0.2791 ([0.1373, 0.4371]) | 0.1779 / 0.8196 | 0.6678 | 0.2617 | 0.1064 | 0.0485 | 0.960 / 0.920 | 3.94 | 1.0000 | 56613.1 | no |
| `Gemini 3.5 Flash Lite (Dirichlet M=3)` | 0.840 | 0.3404 ([0.2004, 0.4839]) | 0.1965 / 0.7769 | 0.7474 | 0.2058 | 0.1171 | 0.0798 | 0.960 / 0.920 | 5.70 | 0.2891 | 62790.4 | no |
| `Strands Decider 2B v19` | 0.720 | 0.4249 ([0.3046, 0.5101]) | 0.2643 / 0.7603 | 0.8787 | 0.1524 | 0.1194 | 0.2398 | 0.960 / 0.880 | 2.36 | 0.0213 | 163.0 | yes |

**Panel 1 Brier-Latency Pareto Frontier**: `DecisionGemma`, `Strands Decider 2B v19`, `Gemini 3.8 Flash (Verbalized + TempScaled OOF)`.

## Panel 2: XSTest Safety Boundary Leaderboard (`N = 360`)

| Arm | Acc | Macro-F1 | Brier (95% Bootstrap CI) | Murphy REL / RES | NLL | ECE | ACE | AURC | Conformal Cov@0.05 / @0.10 | Mean Set@0.10 | McNemar p (vs Ref) | Latency (ms) | Pareto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `DecisionGemma` | 0.9694 | 0.9692 | 0.0465 ([0.0106, 0.1005]) | 0.0137 / 0.4610 | 0.0845 | 0.0209 | 0.0328 | 0.0020 | 0.9694 / 0.9694 | 1.00 | 1.0000 | 175.8 | yes |
| `dgem v0.3.1` | 0.9639 | 0.9633 | 0.0565 ([0.0225, 0.1021]) | 0.0231 / 0.4604 | 0.1001 | 0.0204 | 0.0300 | 0.0025 | 0.9639 / 0.9639 | 1.00 | 0.8036 | 215.1 | no |
| `Conformal Cascade: Classy v0.3.3 -> DecisionGemma` | 0.8833 | 0.8793 | 0.1634 ([0.1187, 0.2273]) | 0.0084 / 0.3388 | 0.2453 | 0.0284 | 0.0358 | 0.0245 | 0.9444 / 0.8972 | 1.04 | 3.35e-07 | 866.8 | no |
| `Conformal Cascade: Classier v0.2.0 -> DecisionGemma` | 0.8917 | 0.8913 | 0.1720 ([0.1214, 0.2316]) | 0.0180 / 0.3398 | 0.2860 | 0.0270 | 0.0604 | 0.0326 | 0.9444 / 0.9083 | 1.05 | 2.46e-07 | 537.1 | no |
| `Classier v0.2.0` | 0.7889 | 0.7886 | 0.2986 ([0.2571, 0.3439]) | 0.0317 / 0.2269 | 0.4581 | 0.0273 | 0.1022 | 0.0965 | 0.9417 / 0.9111 | 1.31 | 2.44e-16 | 482.4 | no |
| `Strands Decider 2B v19 (VectorScaled OOF)` | 0.8083 | 0.8027 | 0.3031 ([0.2142, 0.3883]) | 0.0257 / 0.2164 | 0.4773 | 0.0834 | 0.0957 | 0.1181 | 0.9500 / 0.9000 | 1.30 | 4.33e-12 | 153.9 | yes |
| `Strands Decider 2B v19` | 0.7583 | 0.7339 | 0.3623 ([0.2680, 0.4420]) | 0.0847 / 0.2162 | 0.5445 | 0.0907 | 0.1715 | 0.1489 | 0.9500 / 0.9000 | 1.44 | 3.21e-16 | 153.9 | no |
| `Classy v0.3.3 (VectorScaled OOF)` | 0.6444 | 0.6218 | 0.4581 ([0.3861, 0.5180]) | 0.0175 / 0.0532 | 0.6520 | 0.0585 | 0.0728 | 0.3308 | 0.9528 / 0.9000 | 1.69 | 1.54e-27 | 746.2 | no |
| `Classy v0.3.3` | 0.6444 | 0.6107 | 0.4634 ([0.4107, 0.5071]) | 0.0164 / 0.0468 | 0.6563 | 0.0785 | 0.0815 | 0.3297 | 0.9528 / 0.9000 | 1.69 | 1.54e-27 | 746.2 | no |

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
