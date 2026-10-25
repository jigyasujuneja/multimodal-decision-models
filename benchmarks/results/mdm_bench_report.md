# MDM-Bench: 8-Track Practical Multimodal & System-1 Decision Benchmark (`v0.5.0`)

Deterministic evaluation of **5 standalone model paradigms** and the **LTT + RAPS Calibrated Cascade** across **$N = 400$ decision items** ($50$ items per track across 8 real-world agent, visual, coding, relational, judge, and safety tracks), plus out-of-fold ablations for the **6 L8 GDM Research Upgrades**.

## 1. Aggregate Cross-Model Leaderboard ($N = 400$ Items across 8 Tracks)

| Evaluation Arm | Paradigm | Acc | Macro-F1 | Brier (95% Cluster CI) | NLL | ECE | ACE | Murphy REL | Murphy RES | AURC | p50 Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DecisionGemma-2B** | `system1_diffusion_2b` | `84.50%` | `0.8452` | `0.2257` (`[0.1511, 0.2999]`) | `0.4192` | `0.0753` | `0.0893` | `0.1031` | `0.6558` | `0.0466` | `14.2 ms` |
| **Microsoft-Decision-1-Style-9B** | `system1_autoregressive_9b` | `84.75%` | `0.8543` | `0.2121` (`[0.1302, 0.3082]`) | `0.3789` | `0.0397` | `0.0878` | `0.1202` | `0.6865` | `0.0463` | `28.7 ms` |
| **GPT-4o-Mini-Logprob-20Cap** | `frontier_logprob_20cap` | `85.00%` | `0.8547` | `0.2048` (`[0.1175, 0.2992]`) | `0.3790` | `0.0589` | `0.0918` | `0.1136` | `0.6871` | `0.0399` | `82.2 ms` |
| **Claude-3.7-Sonnet-Dirichlet-M5** | `blackbox_dirichlet_m5` | `88.00%` | `0.8849` | `0.1838` (`[0.1201, 0.2544]`) | `0.3859` | `0.1466` | `0.1105` | `0.1374` | `0.7319` | `0.0231` | `214.1 ms` |
| **Gemini-3.8-Flash-Logprob** | `frontier_multimodal_logprob` | `89.00%` | `0.9007` | `0.1490` (`[0.0824, 0.2306]`) | `0.2874` | `0.0784` | `0.0801` | `0.1056` | `0.7349` | `0.0187` | `94.1 ms` |
| **LTT + RAPS Cascade (DecisionGemma-2B -> Gemini-3.8-Flash)** | `ltt_conformal_cascade` | `90.00%` | `0.9080` | `0.1469` (`[0.0682, 0.2309]`) | `0.2567` | `0.0309` | `0.0594` | `0.0895` | `0.7209` | `0.0200` | `14.7 ms` |

## 2. Per-Track Accuracy & Brier Breakdown ($50$ Items per Track)

| Track | `DecisionGemma-2B` (Acc / Brier) | `Microsoft-Decision-1-Style-9B` (Acc / Brier) | `GPT-4o-Mini-Logprob-20Cap` (Acc / Brier) | `Claude-3.7-Sonnet-Dirichlet-M5` (Acc / Brier) | `Gemini-3.8-Flash-Logprob` (Acc / Brier) | `LTT + RAPS Cascade` (Acc / Brier) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `agent_tool_trie_routing` | `84.0%` / `0.2078` | `86.0%` / `0.1633` | `86.0%` / `0.1862` | `90.0%` / `0.1777` | `88.0%` / `0.1616` | `90.0%` / `0.1651` |
| `rag_rerank_and_ipi_firewall` | `84.0%` / `0.1921` | `84.0%` / `0.2286` | `82.0%` / `0.2152` | `90.0%` / `0.1700` | `86.0%` / `0.1677` | `88.0%` / `0.1730` |
| `gui_som_and_visual_blink` | `90.0%` / `0.2237` | `84.0%` / `0.2362` | `84.0%` / `0.2171` | `84.0%` / `0.2099` | `88.0%` / `0.1624` | `88.0%` / `0.1807` |
| `visual_catalog_sku_audit` | `82.0%` / `0.2322` | `86.0%` / `0.1958` | `90.0%` / `0.1514` | `92.0%` / `0.1459` | `94.0%` / `0.0980` | `92.0%` / `0.1217` |
| `coding_context_compaction` | `82.0%` / `0.2814` | `90.0%` / `0.1870` | `88.0%` / `0.1669` | `90.0%` / `0.1737` | `90.0%` / `0.1496` | `92.0%` / `0.1506` |
| `relational_entity_and_swarm_locks` | `86.0%` / `0.2081` | `80.0%` / `0.2483` | `86.0%` / `0.2040` | `88.0%` / `0.1810` | `86.0%` / `0.1741` | `88.0%` / `0.1517` |
| `judge_rubric_calibration` | `84.0%` / `0.2504` | `76.0%` / `0.2953` | `80.0%` / `0.2834` | `90.0%` / `0.1770` | `92.0%` / `0.1213` | `92.0%` / `0.1000` |
| `safety_and_policy_guardrail` | `84.0%` / `0.2096` | `92.0%` / `0.1419` | `84.0%` / `0.2144` | `80.0%` / `0.2354` | `88.0%` / `0.1573` | `90.0%` / `0.1324` |

## 3. L8 GDM Research Upgrades: Out-of-Fold Verification Summary

### 3.1 Upgrade 1: Class/Rarity-Conditional Mondrian & Weighted Conformal Prediction ($\alpha = 0.10$)

| Conformal Method | Overall Coverage ($N=400$) | Head-Group Coverage ($N=320$) | Rare-Tail Coverage ($N=80$) | Mean Set Size |
| :--- | :---: | :---: | :---: | :---: |
| **Marginal Split RAPS (Baseline)** | `90.75%` | `97.81%` | `62.50%` | `1.147` |
| **Mondrian Group-Conditional RAPS (`v0.5.0`)** | `95.25%` | `95.94%` | `92.50%` | `1.242` |
| **Tibshirani Likelihood-Ratio Weighted RAPS (`v0.5.0`)** | `98.00%` | `100.00%` | `90.00%` | `1.502` |

### 3.2 Upgrade 2: Simultaneous 4-Slot Joint Conformal Calibration ($\alpha_{\text{family}} = 0.10$)

| Multi-Slot Calibration Scheme | Simultaneous 4-Slot Coverage | Target Family-Wise Coverage | Mean Per-Slot Set Size |
| :--- | :---: | :---: | :---: |
| **Independent Per-Slot RAPS (Uncorrected)** | `66.25%` | `90.00%` | `1.016` |
| **Joint Max-Score Split Conformal (`v0.5.0`)** | `90.50%` | `90.00%` | `1.386` |
| **Joint Bonferroni Conformal (`v0.5.0`)** | `90.75%` | `90.00%` | `1.404` |

### 3.3 Upgrade 3: `PriDe` Cyclic Permutation Debiasing & Flip-Rate Stability

| Readout Mode | Top-1 Accuracy | Multiclass Brier | Option-Order Flip Rate | Mean JS Divergence (bits) |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Canonical Order (Shift 0)** | `81.00%` | `0.2834` | `16.25%` | `0.0362` |
| **`PriDe` Cyclic Permutation Debiased (`v0.5.0`)** | `92.00%` | `0.1868` | `6.00%` | `0.0037` |

### 3.4 Upgrades 4, 5, and 6: LTT Funnel Risk Control, Submodular Context Pruning, & Transitive Polytope Projection

| Upgrade Component | Baseline Metric | `v0.5.0` Upgraded Metric | Guarantee / Outcome |
| :--- | :---: | :---: | :--- |
| **Upgrade 4: Learn-then-Test (`LTT`) Funnel Calibration** | Target bound $r_{\max} = 0.14$ ($\delta = 0.05$) | Empirical risk `0.0300` (UCB `0.1284`) | Fast-exit rate `67.8%` with finite-sample Hoeffding-Bentkus risk control |
| **Upgrade 5: Submodular Redundancy-Penalized Context Pruning** | Additive MCKP facet recall `66.7%` | Submodular MCKP facet recall `100.0%` | Retains orthogonal git-blame + config evidence at `43.4%` token reduction |
| **Upgrade 6: Transitive Equivalence Polytope Projection** | `50` violated 3-cycles (`71.43%` edge acc) | `0` violated 3-cycles (`100.00%` edge acc) | `0` triangle violations via exact log-odds correlation clustering |
