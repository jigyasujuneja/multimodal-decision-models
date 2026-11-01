# Formal Propositions v0.9.0 Benchmark Report ($N = 400$ Real Decision Receipts)

| Pillar | Proposition / Mechanism | Baseline Metric | Proposed Metric | Exact Guarantee / $p$-Value |
| :--- | :--- | :---: | :---: | :---: |
| **1. Top-$M$ Truncated Hesitation (`N=100`, $K=100$)** | Lemma 1 (`effective_top_m_entropy`) | `0.5800` acc (`0.00%` tie recall) | **`0.7500`** acc (**`100.00%`** tie recall) | $p = 0.003327$ |
| **2. Hint-Conditioned Cascade Risk (`N=100`)** | Proposition 1 (`evaluate_two_stage_cascade_risk_bound`) | `0.3300` Stage-1 risk | **`0.2100`** cascade risk (`-0.1200`) | Coverage `0.9600` $\ge$ `0.9172` |
| **3. Cut-Polytope & Sinkhorn Trie (`N=100`)** | Propositions 2 & 3 (`cut_polytope_triangle_consistency_score`, `sinkhorn_dealiased_trie_decode`) | `0.0000` marginal trie acc | **`1.0000`** Sinkhorn trie acc | Max violated facet `0.424444` $\le 13/27$, $p = 0.000000$ |
| **4. Combined Label + Covariate Shift (`N=80`)** | Proposition 4 (`label_shift_weighted_conformal_prediction_set`) | `0.9000` overall (`0.5750` tail) | **`0.9825`** overall (**`0.9125`** tail) | Target $\ge 0.90$, $p = 0.000000$ |
