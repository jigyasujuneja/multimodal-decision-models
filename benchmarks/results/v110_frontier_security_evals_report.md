# MDM `v1.1.0` Pre-Registered Security and Evaluation Verification Report

This deterministic verification suite combines committed receipt panels with controlled stress injections (synthetic peer sycophancy, adversarial IPI/CUA payloads, cache isolation probes, and trajectory fault traces) to test every `v1.1.0` statistical and security contract end to end.

- **Total Primary Verification Instances**: `1000` (`10` experiments x `100`)
- **Family-Wise Error Rate Control**: Holm-Bonferroni step-down (`alpha = 0.05`)
- **Binary Acceptance Gates Passed**: `10/10` (`overall_binary_score = 1`)

## Experiment Summary Table

| Exp ID | Wave | Key Baseline Metric | Key `mdm v1.1.0` Metric | Raw `p` | Holm-Adj `p` | Gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-V11-01` | `M5` | nDCG@10: `0.0974` | nDCG@10: `1.0000` | `1.58e-30` | `1.42e-29` | `PASS` |
| `EXP-V11-02` | `M5` | Traj Cov: `0.4800` | Traj Cov: `0.9500` | `<1e-300` | `<1e-300` | `PASS` |
| `EXP-V11-03` | `M6` | Uncert Cov: `0.0000` | Uncert Cov: `0.9571` | `1.36e-20` | `9.49e-20` | `PASS` |
| `EXP-V11-04` | `M6` | Joint Acc: `0.6000` | Joint Acc: `0.8800` | `5.77e-08` | `1.73e-07` | `PASS` |
| `EXP-V11-05` | `M6` | AURC: `0.2371` | AURC: `0.1948` | `5.96e-08` | `1.73e-07` | `PASS` |
| `EXP-V11-06` | `M7` | Escape Rate: `0.8000` | Escape Rate: `0.0000` | `1.82e-12` | `1.09e-11` | `PASS` |
| `EXP-V11-07` | `M7` | Poison Cov: `0.7700` | Poison Cov: `0.9000` | `2.44e-04` | `2.44e-04` | `PASS` |
| `EXP-V11-08` | `M7` | Iso / Reject: `0.0000` | Iso / Reject: `1.0000` | `4.14e-25` | `3.31e-24` | `PASS` |
| `EXP-V11-09` | `M8` | 95% CI Cov: `0.0000` | 95% CI Cov: `0.9440` | `7.45e-09` | `2.98e-08` | `PASS` |
| `EXP-V11-10` | `M8` | Worst-Para Cov: `0.5500` | Worst-Para Cov: `0.9100` | `2.91e-11` | `1.46e-10` | `PASS` |
