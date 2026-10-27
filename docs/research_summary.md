# Research Summary: Calibrated Multimodal Decision Models

This document summarizes the research hypotheses, mathematical mechanisms, and verified empirical results implemented in `multimodal-decision-models` (`v0.4.0`). For full derivations, proofs, and benchmark tables, see the accompanying research paper in [`docs/research_paper.md`](research_paper.md) and the architectural reference in [`docs/architecture.md`](architecture.md).

## 1. Core Research Contributions

| Workstream | Bottleneck Addressed | Mathematical Mechanism | Primary Empirical Result |
| :--- | :--- | :--- | :--- |
| **Four-Paradigm Readouts & Dirichlet Consensus (`backends.py`)** | Masked diffusion, logprob APIs, structured JSON verifiers, and black-box samplers expose incompatible outputs. | Single-token `[A-Z]` bijection, causal `evidence_read` ordering, and symmetric Dirichlet ($\alpha_0 = 0.5$) posterior predictive smoothing with exact epistemic variance $\operatorname{Var}(\theta_k \mid c) = p_k(1 - p_k)/(A_0 + 1)$. | Unifies 10 heterogeneous models under one `DecisionBackend` contract with finite NLL and bounded epistemic variance across $M = 3..5$ draws. |
| **Pre-Fusion Split Conformal & RAPS Cascades (`conformal.py`, `pkg/mdm/`)** | Heuristic confidence thresholds lack finite-sample coverage guarantees and can be masked by external retrieval priors. | Order statistic $m = \lceil(n+1)(1-\alpha)\rceil$ over `lac`, `aps`, and `raps` evaluated on raw Stage-1 probabilities before prior fusion; escalates when $|\mathcal{C}_\alpha(x)| \ne 1$ or abstain $\in \mathcal{C}_\alpha(x)$. | Achieves empirical coverage $\ge 1 - \alpha$ across all 20 benchmark arms; `Classier v0.2.0 -> DecisionGemma` cascade lifts XSTest accuracy from $0.7889$ to $0.8917$ at $31.1\%$ escalation ($p = 7.84 \times 10^{-10}$). |
| **Pre-Gated PoE & Log-Softmax Priors (`calibration.py`, `shortlist.py`)** | Raw retrieval cosines ($\Delta \cos \approx 0.14$) under-actuate log-odds, while unconditional prior fusion degrades confident visual predictions under base-rate shift. | Temperature-scaled log-softmax cosine normalization ($\tau = 0.07$, $\Delta \pi \approx 2.0\text{ nats}$) gated on raw Shannon hesitation $\tilde{H}_{\text{raw}} \ge \tau_H$ (`"hesitant"` mode), plus $L_2$-regularized post-hoc temperature and vector scaling. | Recovers $+80$ top-1 hits on 300-trial hesitant wide-option reads and reduces Brier score by $>15\%$ under $5\%$ and $90\%$ induced base-rate shifts. |
| **Dual-Buffer Visual Compiler (`visual_compiler.py`, `embedding_index.py`)** | Local contrast stretching improves fine-print OCR for verifiers but distorts brand chromaticity for Stage-2A retrieval. | Stage-2A unsharpened Minkowski $p=6$ Shades-of-Gray + inner-80% CIE $L^*a^*b^*$ $\Delta E$ + residual MLP adapter; Stage-2B contact sheets with inter-candidate pixel variance band $V(y)$, $2\times$ bottom-30% weight zoom, and $1.15\times$ cylindrical unwarping. | Enables zero-VLM-call fast-exit when $P_1 - P_2 \ge 0.14$ and $Z \ge 1.65$ while isolating fine-print net-weight and variant cues for Stage-2 verification. |
| **Two-Level Semantic Taxonomy Tries (`hierarchy.py`, `pkg/mdm/`)** | Flat Round-1 tournament brackets arbitrarily split domain siblings ($30.8\%$ of `MASSIVE` bracket errors are sibling confusions). | Two-level `[A-Z]` taxonomy tries (`mtp_trie`, `beam_2pass`, `hesitation_hybrid`) packing up to 8 brackets per pass across $K \le 625$ classes with opaque `option_k` stems. | Compresses $K = 60..150$ routing to 2 passes (`3` passes for $K = 202$) and outperforms marginal argmax decoding by $>15$ hits on ragged taxonomies. |
| **8-Slot MCKP Context Pruning (`context_pruner.py`, `pkg/mdm/`)** | Multi-turn coding and catalog agents exceed context token budgets; naive truncation drops required tool definitions and stack traces. | Single-pass 8-slot `/v1/systemone` scoring over `{"full", "stub", "drop"}` coupled with dominance-pruned Multiple-Choice Knapsack dynamic programming (`mckp_hesitation_stub`) and pinned-block constraints. | Reduces trajectory tokens by $58.85\%$ at $100.00\%$ required-block recall across 8 models ($p \le 7.45 \times 10^{-9}$ vs FIFO, TF-IDF, and greedy drop), matching $3^8$ brute-force DP on $202/202$ trajectories. |
| **Single-Pass RAG Reranking & Injection Quarantine (`contracts.py`)** | Adversarial instructions embedded inside retrieved chunks hijack single-pass rerankers (`28.18%` to `34.55%` unfenced `ASR@1`). | Structural `<untrusted_data>` XML fencing (`sanitize_untrusted_context`) stripping spoofed tags, `SYSTEM:` headers, and role delimiters in an 8-slot relevance + injection-audit pass. | Cuts `ASR@1` to $0.00\%$ across 6 attack families on a 220-query metadata-balanced benchmark while reaching $0.8900$ to $0.9295$ `nDCG@3`. |
| **Expected-Grade Rubric Scoring (`eval_suite.py`)** | Discrete 5-level Likert autoraters suffer massive score ties under argmax decoding. | Continuous expected grade $\mathbb{E}[g \mid p] = \sum_k p_k g_k$, exact four-term within-bin Murphy Brier decomposition ($\operatorname{REL} - \operatorname{RES} + \operatorname{UNC} + \operatorname{WBV}$), and K-fold cascade threshold calibration. | Resolves 5 Likert grades into 100 unique continuous scores ($0.960$ pairwise concordance) and achieves frontier McNemar parity ($p = 0.1250$) at $3.95\%$ escalation. |
| **Mondrian, Weighted, & Multi-Slot Joint Conformal (`conformal.py`, `pkg/mdm/`)** | Marginal conformal under-covers rare tail classes (`58.75%`), and independent per-slot conformal fails family-wise coverage across $S=4$ simultaneous slots (`68.00%`). | Group/class-conditional Mondrian quantiles, Tibshirani likelihood-ratio weighted conformal, and joint max-score $M_i = \max_s s(X_{i,s}, Y_{i,s})$ calibration. | Lifts rare-tail coverage on `MDM-Bench` ($N=400$) to $88.75\%$ (Mondrian & Weighted; $96.00\%$ / $97.75\%$ overall), and restores simultaneous 4-slot coverage to $90.25\%$ (Max-Score) and $90.50\%$ (Bonferroni). |
| **`PriDe` Debiasing, `LTT` Funnel Gates, Submodular MCKP, & Polytope Projection (`calibration.py`, `context_pruner.py`, `consistency.py`)** | `[A-Z]` positional bias flips decisions under option reordering (`10.25%`), additive MCKP retains duplicate stack traces, and pairwise entity scores form 3-cycle violations. | Cyclic permutation prior debiasing (`debiased_cyclic_permutation_probs`), Hoeffding-Bentkus `calibrate_ltt_funnel_gates`, Jaccard-discounted MCKP, and exact log-odds correlation clustering (`project_triangles_to_transitive_consistency`). | Cuts permutation flip rate from $10.25\%$ to $6.50\%$ ($0.4286 \to 0.4136$ Brier), bounds selective risk at $\text{UCB} = 0.0922 \le 0.14$ ($0.0100$ empirical risk), lifts context facet recall on 50 real `EXP-49` trajectories from $66.67\%$ to $78.67\%$, and eliminates $50 \to 0$ 3-cycle contradictions ($91.43\%$ edge acc). |
| **Frontier GenAI Suite: Online `BC-ACI`, `ParetoBandit`, Calibrated `PRM` PUCT, & Supra-Bayesian Pooling (`v0.7.0`)** | Abrupt distribution shift breaks static split conformal; static cascades overspend budget; tree search over-evaluates suboptimal nodes; naive PoE collapses under $4\times$ correlated clones. | Online bias-corrected quantile tracking (`OnlineBCACITracker`), Lagrangian primal-dual contextual/dueling bandit (`PrimalDualParetoBanditRouter`), conformal branch-and-bound PUCT, and inverse-covariance logit pooling (`covariance_discounted_logit_pool`). | Tracks exact $0.9000$ post-shift coverage ($N=100$), cuts routing cost by $-86.2\%$ with $0.0000$ budget violation ($N=100$), prunes $-43.8\%$ verifier nodes with $0.0\%$ accuracy drop ($N=100$), and lifts accuracy from $65.00\%$ to $77.00\%$ ($p=0.0357$) under $4\times$ clone attack. |

## 2. Pareto-Optimal Operating Regimes

Based on the 10-model empirical benchmark (`benchmarks/results/multimodel_benchmark_report.md`), three operating regimes lie on the Brier-latency Pareto frontier:

1. **Ultra-Low-Latency Edge / Probe Regime (`~150-180 ms`)**:
   - Use `Strands Decider 2B v19` with out-of-fold diagonal vector scaling (`fit_vector_scaling`) or `DecisionGemma` on `/v1/systemone`.
   - On XSTest ($N = 360$), `DecisionGemma` achieves $0.9694$ accuracy, $0.0465$ Brier score, and $0.0020$ AURC at $175.8\text{ ms}$.
2. **Selective Conformal Cascade Regime (`~280-540 ms` effective mean latency)**:
   - Pair a fast Stage-1 model (`dgem v0.3.7`, `Classier v0.2.0`, or `Strands Decider 2B v19`) with pre-fusion RAPS ($\alpha = 0.10$, `k_reg=1`, `lambda_reg=0.02`) and escalate only non-singleton or high-hesitation ($\tilde{H}_{\text{raw}} \ge 0.16$) items to a Stage-2 verifier.
   - Escalates only $14.0\%$ to $31.1\%$ of traffic while recovering statistically significant accuracy gains ($p = 7.84 \times 10^{-10}$ on XSTest).
3. **High-Precision Frontier Verifier Regime (`~4.2 s`)**:
   - Use `Gemini 3.8 Flash` with single-token `[A-Z]` / verbalized confidence readout and post-hoc temperature scaling (`fit_temperature_scaling`) plus dual-buffer Stage-2B contact sheets (`compile_dual_buffer_contact_sheet`).
   - Achieves $0.920$ accuracy, $0.1419$ Brier score, $0.0231$ ECE, and $0.0177$ AURC on Panel 1 ($N = 50$).

## 3. Reproducing the Research Artifacts

Run the benchmark and verification scripts from the repository root:

```bash
# 1. Execute all deterministic, probabilistic, and adversarial test gates
PYTHONPATH=src python3 -m unittest discover -s tests -v
go test -race ./pkg/mdm/...

# 2. Regenerate the 10-model / 20-arm calibration and conformal benchmark report
PYTHONPATH=src python3 benchmarks/run_multimodel_benchmark.py

# 3. Regenerate the 400-item 8-track MDM-Bench report and run the 8 practical workflows
PYTHONPATH=src python3 benchmarks/run_mdm_bench.py
PYTHONPATH=src python3 examples/how_i_ai_workflows.py
```

The generated JSON and Markdown artifacts are written to [`benchmarks/results/multimodel_benchmark_results.json`](../benchmarks/results/multimodel_benchmark_results.json), [`benchmarks/results/multimodel_benchmark_report.md`](../benchmarks/results/multimodel_benchmark_report.md), [`benchmarks/results/mdm_bench_results.json`](../benchmarks/results/mdm_bench_results.json), and [`benchmarks/results/mdm_bench_report.md`](../benchmarks/results/mdm_bench_report.md). See [`docs/practical_use_cases.md`](practical_use_cases.md) for the 8 practical workflow recipes.
