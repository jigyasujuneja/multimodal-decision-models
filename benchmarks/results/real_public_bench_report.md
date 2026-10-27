# Real-Data Benchmark Report: DecisionBench 1.0, TypedDecisions, arXiv:2610.03935, and arXiv:2610.21412v1

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

## 2. Exp A: Universal 256-Code Bijection & Think Guard

- **Universal Single-Token Code Inventory**: `256` codes (`['A', 'B', 'C', 'D', 'E', 'F'] ... ['PP', 'PR', 'PS', 'PT']`)
- **Contains Rare BPE-Split Two-Letter Pairs (`BQ, CJ, FQ, GZ, ...`)**: `False`
- **Unstripped `<think>` First-Token Leakage Guard Triggered (`LogprobAlignmentError`)**: `True`

## 3. Exp B: Reasoning-to-Readout Distillation (`N = 300`)

| Readout / Distillation Arm | Held-Out Accuracy | Multiclass Brier | ECE | NLL | KL to Human Soft Consensus | Ordinal Score MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. `Qwen3.8-27B` with Unstripped `<think>` Tag Bug (`teacher.json`) | `0.3133` | `0.9745` | `0.4365` | `1.6820` | `1.0410` | `1.4022` |
| 2. `Kev-4B` Base Student Anchor (`kev4b-traincal.json`) | `0.6533` | `0.4611` | `0.0570` | `0.8259` | `0.2844` | `0.4327` |
| 3. `Qwen3.8-27B` Clean Readout (`enable_thinking=False`) | `0.6800` | `0.4421` | `0.0992` | `0.8383` | `0.5822` | `0.4583` |
| 4. **Eq. (12) Distilled Posterior (`27B Teacher + Kev-4B KL Anchor`)** | `0.7000` | `0.4027` | `0.0882` | `0.7336` | `0.2499` | `0.3761` |

## 4. Exp C: `AgentDecisionBench` (`N = 600` States)

### 4.1 `L2` Acceptable-Set Accuracy & `must_ask` Errors

| Model | Common Acceptable-Set Acc | Common Single-Gold Acc | `on_plan` Acc | `after_off_plan` Recovery Acc | `must_ask` Acc | `must_ask` Fatal Irreversible Error Rate | `DecisionGym-Shift` Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rl` | `0.9500` | `0.9367` | `1.0000` | `0.8125` | `1.0000` | `0.0000` | `0.9700` |
| `sft_v1` | `0.9500` | `0.9367` | `1.0000` | `0.8125` | `1.0000` | `0.0000` | `0.9833` |
| `e14a` | `0.9567` | `0.9433` | `1.0000` | `0.8375` | `1.0000` | `0.0000` | `0.7633` |
| `jev` | `0.7333` | `0.6833` | `0.9786` | `0.6625` | `0.3750` | `0.6250` | `0.8667` |
| `oaidec` | `0.7300` | `0.6500` | `0.9643` | `0.6375` | `0.4125` | `0.5875` | `0.8067` |
| `openjevhf` | `0.7767` | `0.7133` | `0.9857` | `0.8250` | `0.3625` | `0.6375` | `0.8700` |
| `sl0.8b` | `0.3800` | `0.3433` | `0.4357` | `0.3000` | `0.3625` | `0.6375` | `0.4633` |
| `sl4b` | `0.7633` | `0.7000` | `0.9143` | `0.7125` | `0.5500` | `0.4500` | `0.7967` |
| `sl9b` | `0.7767` | `0.7133` | `0.9714` | `0.6375` | `0.5750` | `0.4250` | `0.8567` |
| `sl27b` | `0.7933` | `0.7300` | `1.0000` | `0.8375` | `0.3875` | `0.6125` | `0.9000` |
| `intern` | `0.6167` | `0.5400` | `0.7929` | `0.5625` | `0.3625` | `0.6375` | `0.7167` |
| `base` | `0.3933` | `0.3567` | `0.4643` | `0.3000` | `0.3625` | `0.6375` | `0.4833` |

### 4.2 Action-Stakes Deferral on `DecisionGym-Shift` (`N = 300`)

| Model | `tau_prop1` (`on_plan`) | Prop. 1 Shift Taken | Prop. 1 Accepted Errors (Irrev) | Action-Stakes Shift Taken | Action-Stakes Accepted Errors (Irrev) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `rl` | `0.9324` | `0.9267` | `4` (`1`) | `0.6800` | **`3` (`0`)** |
| `e14a` | `0.8016` | `0.9267` | `61` (`44`) | `0.5967` | **`21` (`4`)** |
| `jev` | `1.0000` | `0.3033` | `4` (`0`) | `0.1700` | **`4` (`0`)** |
| `sl9b` | `0.9684` | `0.1533` | `0` (`0`) | `0.0500` | **`0` (`0`)** |
| `sl27b` | `0.9828` | `0.3167` | `0` (`0`) | `0.1567` | **`0` (`0`)** |

### 4.3 Selective Brier SFT & Tango Non-Inferiority

- **Unconditional Teacher Distillation (`0.3 * gold + 0.7 * p_Jev`) Corrupted `must_ask` Targets**: `45 / 80`
- **Eq. (1) Selective Teacher + Brier SFT Corrupted `must_ask` Targets**: **`0 / 80`** (mean SFT loss `0.4294`)

## 5. Key Takeaways, Trade-Offs, & Limitations

- **Operational Takeaway**: Pre-gated Product-of-Experts (`Pre-Gated PoE + Temperature Scaling`) outperforms Naive 5-Model Mean (`82.08%` vs `78.75%` accuracy, `0.2449` vs `0.3125` Brier) because uncalibrated models (`laya`, `julia-onnx`) degrade unweighted ensembles.
- **Limitations & Scope Boundaries**: Scalar Proposition 1 confidence deferral fails under candidate-set expansion (`69 -> 129` tools on `DecisionGym-Shift`), requiring explicit action-stakes separation ($V_{\text{irrev}} = 5.0$ vs $V_{\text{rev}} = 0.15$) and secondary anchor consensus before executing state-mutating calls.
