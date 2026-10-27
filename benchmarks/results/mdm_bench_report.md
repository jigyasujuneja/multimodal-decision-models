# MDM-Bench: 8-Track Practical Multimodal & System-1 Decision Benchmark (`v0.5.0`)

Deterministic evaluation of **5 standalone model paradigms** and the **LTT + RAPS Calibrated Cascade** across **$N = 400$ decision items** ($50$ items per track across 8 real-world agent, visual, coding, relational, judge, and safety tracks), plus out-of-fold ablations for the **6 Multimodal Calibration & Routing Upgrades**.

## 1. Aggregate Cross-Model Leaderboard ($N = 400$ Items across 8 Tracks)

| Evaluation Arm | Paradigm | Acc | Macro-F1 | Brier (95% Cluster CI) | NLL | ECE | ACE | Murphy REL | Murphy RES | AURC | p50 Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DecisionGemma-2B** | `system1_diffusion_2b` | `82.50%` | `0.7344` | `0.2463` (`[0.1733, 0.3328]`) | `0.4438` | `0.0840` | `0.1170` | `0.1241` | `0.8343` | `0.0495` | `126.7 ms` |
| **dgem-v0.3.7-SystemOne** | `system1_gateway_v037` | `79.00%` | `0.6548` | `0.3229` (`[0.2036, 0.4788]`) | `0.7430` | `0.1142` | `0.1056` | `0.1609` | `0.7944` | `0.0772` | `220.8 ms` |
| **Strands-Decider-2B-v19** | `compact_decider_2b` | `66.25%` | `0.5250` | `0.4476` (`[0.3452, 0.5501]`) | `0.7793` | `0.0590` | `0.1890` | `0.2251` | `0.7340` | `0.1895` | `128.2 ms` |
| **Laya-Decision-4B** | `autoregressive_decider_4b` | `47.75%` | `0.3283` | `0.6242` (`[0.5150, 0.7459]`) | `1.0572` | `0.1398` | `0.2396` | `0.3207` | `0.6530` | `0.3522` | `257.4 ms` |
| **Gemini-3.8-Flash-Logprob** | `frontier_multimodal_logprob` | `89.00%` | `0.9172` | `0.1613` (`[0.0987, 0.2348]`) | `0.3231` | `0.0445` | `0.0807` | `0.0840` | `0.8792` | `0.0446` | `348.9 ms` |
| **LTT + RAPS Cascade (DecisionGemma-2B -> Gemini-3.8-Flash)** | `ltt_conformal_cascade` | `88.75%` | `0.8975` | `0.1598` (`[0.1028, 0.2278]`) | `0.3205` | `0.0390` | `0.0731` | `0.0678` | `0.8644` | `0.0376` | `440.5 ms` |

## 2. Per-Track Accuracy & Brier Breakdown ($50$ Items per Track)

| Track | `DecisionGemma-2B` (Acc / Brier) | `dgem-v0.3.7-SystemOne` (Acc / Brier) | `Strands-Decider-2B-v19` (Acc / Brier) | `Laya-Decision-4B` (Acc / Brier) | `Gemini-3.8-Flash-Logprob` (Acc / Brier) | `LTT + RAPS Cascade` (Acc / Brier) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `agent_tool_trie_routing` | `100.0%` / `0.0002` | `100.0%` / `0.0015` | `96.0%` / `0.0716` | `76.0%` / `0.3140` | `100.0%` / `0.0041` | `100.0%` / `0.0000` |
| `rag_rerank_and_ipi_firewall` | `92.0%` / `0.1042` | `86.0%` / `0.1901` | `74.0%` / `0.3512` | `42.0%` / `0.6793` | `100.0%` / `0.0049` | `100.0%` / `0.0156` |
| `gui_som_and_visual_blink` | `74.0%` / `0.3333` | `70.0%` / `0.4025` | `62.0%` / `0.4319` | `54.0%` / `0.5194` | `96.0%` / `0.0893` | `94.0%` / `0.1291` |
| `visual_catalog_sku_audit` | `86.0%` / `0.1805` | `82.0%` / `0.2847` | `62.0%` / `0.4866` | `48.0%` / `0.5693` | `88.0%` / `0.1602` | `90.0%` / `0.0982` |
| `coding_context_compaction` | `68.0%` / `0.4561` | `62.0%` / `0.5430` | `40.0%` / `0.7304` | `54.0%` / `0.6157` | `58.0%` / `0.5193` | `58.0%` / `0.4919` |
| `relational_entity_and_swarm_locks` | `84.0%` / `0.2372` | `68.0%` / `0.6113` | `74.0%` / `0.4874` | `24.0%` / `0.9494` | `82.0%` / `0.2573` | `84.0%` / `0.2192` |
| `judge_rubric_calibration` | `72.0%` / `0.3897` | `84.0%` / `0.2547` | `62.0%` / `0.5151` | `38.0%` / `0.7144` | `94.0%` / `0.1363` | `90.0%` / `0.1787` |
| `safety_and_policy_guardrail` | `84.0%` / `0.2690` | `80.0%` / `0.2955` | `60.0%` / `0.5066` | `46.0%` / `0.6320` | `94.0%` / `0.1190` | `94.0%` / `0.1461` |

## 3. Multimodal Calibration & Routing Upgrades: Out-of-Fold Verification Summary

### 3.1 Upgrade 1: Class/Rarity-Conditional Mondrian & Weighted Conformal Prediction ($\alpha = 0.10$)

| Conformal Method | Overall Coverage ($N=400$) | Head-Group Coverage ($N=320$) | Rare-Tail Coverage ($N=80$) | Mean Set Size |
| :--- | :---: | :---: | :---: | :---: |
| **Marginal Split RAPS (Baseline)** | `90.25%` | `98.12%` | `58.75%` | `1.183` |
| **Mondrian Group-Conditional RAPS (`v0.5.0`)** | `96.00%` | `97.81%` | `88.75%` | `1.343` |
| **Tibshirani Likelihood-Ratio Weighted RAPS (`v0.5.0`)** | `97.75%` | `100.00%` | `88.75%` | `1.660` |

### 3.2 Upgrade 2: Simultaneous 4-Slot Joint Conformal Calibration ($\alpha_{\text{family}} = 0.10$)

| Multi-Slot Calibration Scheme | Simultaneous 4-Slot Coverage | Target Family-Wise Coverage | Mean Per-Slot Set Size |
| :--- | :---: | :---: | :---: |
| **Independent Per-Slot RAPS (Uncorrected)** | `68.00%` | `90.00%` | `2.846` |
| **Joint Max-Score Split Conformal (`v0.5.0`)** | `90.25%` | `90.00%` | `3.149` |
| **Joint Bonferroni Conformal (`v0.5.0`)** | `90.50%` | `90.00%` | `3.166` |

### 3.3 Upgrade 3: `PriDe` Cyclic Permutation Debiasing & Flip-Rate Stability

| Readout Mode | Top-1 Accuracy | Multiclass Brier | Option-Order Flip Rate | Mean JS Divergence (bits) |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Canonical Order (Shift 0)** | `74.25%` | `0.4286` | `10.25%` | `0.0525` |
| **`PriDe` Cyclic Permutation Debiased (`v0.5.0`)** | `75.25%` | `0.4136` | `6.50%` | `0.0225` |

### 3.4 Upgrades 4, 5, and 6: LTT Funnel Risk Control, Submodular Context Pruning, & Transitive Polytope Projection

| Upgrade Component | Baseline Metric | `v0.5.0` Upgraded Metric | Guarantee / Outcome |
| :--- | :---: | :---: | :--- |
| **Upgrade 4: Learn-then-Test (`LTT`) Funnel Calibration** | Target bound $r_{\max} = 0.14$ ($\delta = 0.05$) | Empirical risk `0.0100` (UCB `0.0922`) | Fast-exit rate `47.5%` with finite-sample Hoeffding-Bentkus risk control |
| **Upgrade 5: Submodular Redundancy-Penalized Context Pruning** | Additive MCKP facet recall `66.7%` | Submodular MCKP facet recall `78.7%` | Retains orthogonal config + interface evidence at `61.2%` token reduction |
| **Upgrade 6: Transitive Equivalence Polytope Projection** | `50` violated 3-cycles (`71.43%` edge acc) | `0` violated 3-cycles (`91.43%` edge acc) | `0` triangle violations via exact log-odds correlation clustering |
