# Real-Data Benchmark Report: DecisionBench 1.0, TypedDecisions, and arXiv:2610.03935 Experiments

## 1. Real 7-Model Evaluation on Multi-Slot TypedDecisions (`N = 500` Real Questions, `250` Held-Out Test)

| Model / Paradigm | Held-Out Accuracy | Raw Brier | Calibrated Brier (`TS`) | Raw ECE | Calibrated ECE (`TS`) | Calibrated NLL | KL to Human Soft Consensus | Temp `T` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `deem-4b` | `0.8000` | `0.3130` | `0.2809` | `0.1408` | `0.0643` | `0.4725` | `0.1862` | `0.61` |
| `intern-decision-4b` | `0.8333` | `0.2810` | `0.2625` | `0.1275` | `0.0793` | `0.4491` | `0.2553` | `0.73` |
| `jev` | `0.7000` | `0.3980` | `0.4274` | `0.0942` | `0.1087` | `0.7821` | `0.6221` | `2.40` |
| `julia-onnx` | `0.6958` | `0.5520` | `0.4062` | `0.2682` | `0.0545` | `0.7387` | `0.3700` | `5.74` |
| `karar-4b` | `0.7875` | `0.3152` | `0.3012` | `0.0926` | `0.0556` | `0.5123` | `0.1962` | `0.71` |
| `laya` | `0.3958` | `0.7142` | `0.6694` | `0.1430` | `0.0268` | `1.1825` | `0.4303` | `2.84` |
| `wahler-4b` | `0.7833` | `0.3149` | `0.2985` | `0.0962` | `0.0471` | `0.5090` | `0.2136` | `0.67` |
| **Naive 5-Model Mean** | `0.8042` | `0.3288` | `0.3288` | `0.1655` | `0.1655` | `-` | `-` | `-` |
| **Pre-Gated PoE + Temperature Scaling (Ours)** | **`0.8167`** | `-` | **`0.2775`** | `-` | **`0.0486`** | **`0.4698`** | **`0.2058`** | `Per-Model` |

### Conformal Guarantees on Real Data (Target `1 - alpha = 0.90`)

1. **Primitive-Conditional Mondrian RAPS vs Marginal RAPS**:
   - `choice` (`n = 72`): Marginal RAPS = `0.8611`, Mondrian RAPS = `0.9583`
   - `noul` (`n = 72`): Marginal RAPS = `0.9306`, Mondrian RAPS = `0.9028`
   - `score` (`n = 96`): Marginal RAPS = `0.8958`, Mondrian RAPS = `0.8854`
2. **Simultaneous 5-Slot Workflow Joint Coverage (`N = 50` Held-Out 5-Slot Records)**:
   - Uncorrected Independent Per-Slot RAPS (`alpha = 0.10`): `0.6042` (violates `0.90` target due to 5-slot error compounding)
   - **Simultaneous Max-Score Joint Conformal (Ours)**: **`0.9792`** (mean set size `1.971`)
   - **Simultaneous Bonferroni Joint Conformal (Ours)**: **`1.0000`** (mean set size `2.404`)

## 2. Experiment A: Universal 256-Code Single-Token Bijection & Thinking-Tag Guard (`arXiv:2610.03935` Sec 6.1)

- **Universal Single-Token Code Inventory**: `256` codes (`['A', 'B', 'C', 'D', 'E', 'F'] ... ['PP', 'PR', 'PS', 'PT']`)
- **Contains Rare BPE-Split Two-Letter Pairs (`BQ, CJ, FQ, GZ, ...`)**: `False`
- **Unstripped `<think>` First-Token Leakage Guard Triggered (`LogprobAlignmentError`)**: `True`

## 3. Experiment B: Reasoning-to-Readout Self-Distillation (`arXiv:2610.03935` Sec 6.2 Eq. 12, `N = 300` Real Questions)

| Readout / Distillation Arm | Held-Out Accuracy | Multiclass Brier | ECE | NLL | KL to Human Soft Consensus | Ordinal Score MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. `Qwen3.8-27B` with Unstripped `<think>` Tag Bug (`teacher.json`) | `0.3133` | `0.9745` | `0.4365` | `1.6820` | `1.0410` | `1.4022` |
| 2. `Kev-4B` Base Student Anchor (`kev4b-traincal.json`) | `0.6533` | `0.4611` | `0.0570` | `0.8259` | `0.2844` | `0.4327` |
| 3. `Qwen3.8-27B` Clean Readout (`enable_thinking=False`) | `0.6800` | `0.4421` | `0.0992` | `0.8383` | `0.5822` | `0.4583` |
| 4. **Eq. (12) Distilled Posterior (`27B Teacher + Kev-4B KL Anchor`)** | `0.7000` | `0.4027` | `0.0882` | `0.7336` | `0.2499` | `0.3761` |
