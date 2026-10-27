# Architecture and Mathematical Design

Single-pass logprob engines, masked diffusion deciders, and black-box sampled models expose incompatible output surfaces and miscalibrated confidence scales. This document specifies the calibration, set-valued abstention, dual-buffer photometry, retrieval, context-pruning, bandit-routing, and contract primitives in `multimodal-decision-models` (`v0.7.0`). For the full research paper manuscript and empirical evaluations ($N = 400$ `MDM-Bench`, $N = 360$ `XSTest`, $N = 600$ `AgentDecisionBench`, and $N = 400$ `v0.7.0` Frontier Suite), see [`docs/research_paper.md`](research_paper.md) and [`docs/research_summary.md`](research_summary.md).

| Architectural Module | Core Primitives | Mathematical Guarantee / Contract | Verified Empirical Performance |
| :--- | :--- | :--- | :--- |
| **`backends.py` & `shortlist.py`** | `DgemSystemOneBackend`, `OpenAILogprobBackend`, `VertexGeminiBackend`, `BlackBoxSampledBackend`, `UNIVERSAL_256_DECISION_CODES` | Validated simplex $\Delta^{K-1}$; symmetric Dirichlet $\alpha_0 = 0.5$ smoothing; `255/255` single-token BPE bijection | Unifies 12 models with bounded NLL (`0.1419` Brier on $N = 50$, `0.9694` acc at `175.8 ms` on $N = 360$ `XSTest`) |
| **`conformal.py` & `pkg/mdm/`** | `calibrate_conformal_quantile`, `calibrate_mondrian_conformal_quantiles`, `calibrate_multislot_joint_conformal`, `OnlineBCACITracker` | Marginal, group-conditional Mondrian, likelihood-ratio weighted, $S$-slot joint FWER, and online `BC-ACI` coverage $\ge 1 - \alpha$ | Restores rare-tail coverage (`58.75% -> 88.75%`), 4-slot coverage (`68.00% -> 90.25%`), and post-shift online coverage (`0.9000` on $N = 100$) |
| **`calibration.py`** | `fuse_with_prior`, `fit_temperature_scaling`, `fit_vector_scaling`, `debiased_cyclic_permutation_probs`, `calibrate_ltt_funnel_gates` | Pre-gated Bayesian PoE on $\tilde{H}_{\text{raw}}$; `PriDe` cyclic debiasing; Hoeffding-Bentkus `LTT` risk bound $\text{UCB}(\hat{\lambda}) \le r_{\max}$ | `LTT + RAPS Cascade` achieves `0.1598` Brier and `0.0390` ECE on `MDM-Bench` ($N = 400$) at `128.4 ms` p50 latency |
| **`visual_compiler.py` & `som_navigator.py`** | `shades_of_gray_white_balance`, `compile_dual_buffer_contact_sheet`, `safe_expected_mark_center` | Unsharpened Stage-2A Minkowski $p=6$ retrieval buffer vs contrast-stretched Stage-2B saliency sheet; unimodal-safe UI click snapping | Prevents CIE $L^*a^*b^*$ chromaticity drift while isolating $2\times$ fine-print net-weight cues (`12,124` bytes PNG) |
| **`context_pruner.py` & `contracts.py`** | `prune_agent_context_blocks`, `sanitize_untrusted_context`, `validate_pack_quantity_consistency` | Submodular facility-location MCKP under `token_budget` with pinned-block invariants; `<untrusted_data>` XML prompt-injection quarantine | `58.85%` token reduction at `100.0%` required-block recall ($N = 202$, $p \le 7.45 \times 10^{-9}$) and `0.00%` `ASR@1` ($N = 220$) |
| **`bandit_router.py`, `prm_tree_search.py`, `consistency.py`** | `PrimalDualParetoBanditRouter`, `run_calibrated_prm_puct_search`, `covariance_discounted_logit_pool`, `project_triangles_to_transitive_consistency` | Lagrangian primal-dual cost pacing; conformal branch-and-bound PUCT; inverse-covariance logit pooling; transitive polytope projection | `-86.2%` routing cost ($N = 100$), `-43.8%` tree nodes pruned ($N = 100$), `+12.0` pts under $4\times$ clone attack ($p = 0.0357$), `50 -> 0` 3-cycles |

## 1. Four-Paradigm Backends and Dirichlet Smoothing (`backends.py`)

`DecisionBackend` standardizes multi-slot probability readouts across four execution surfaces:

- **`DgemSystemOneBackend`**: Scores up to 8 slots $\times$ 26 options in one forward pass against `/v1/systemone`.
- **`OpenAILogprobBackend`**: Maps option identifiers onto single-token uppercase letters (`A..Z`) and verifies `top_logprobs` alignment on `/v1/chat/completions`.
- **`VertexGeminiBackend`**: Enforces `propertyOrdering` with `evidence_read` decoded ahead of `[A-Z]` slot enums so lower-triangular causal attention conditions choices on explicit visual or numeric observations.
- **`BlackBoxSampledBackend`**: Aggregates $M$ sampled completions or verbalized vote counts $c_k$ without requiring token logprobs. Under a symmetric Dirichlet prior $\alpha_0 > 0$ (default $\alpha_0 = 0.5$ Jeffreys prior) with $A_0 = \sum_{j=1}^K (c_j + \alpha_0)$, posterior predictive probabilities and per-option epistemic variances are:

$$p_k = \frac{c_k + \alpha_0}{A_0}, \qquad \operatorname{Var}(\theta_k \mid c) = \frac{p_k(1 - p_k)}{A_0 + 1}$$

For finite $M$, $p_k \in (0, 1)$ keeps negative log-likelihood finite and normalized entropy strictly positive while $\sum_k \operatorname{Var}(\theta_k \mid c)$ shrinks monotonically as $M$ grows.

## 2. Split Conformal Prediction and Pre-Fusion Gating (`conformal.py`)

Given $n$ exchangeable calibration examples $(X_i, Y_i)$ and target miscoverage $\alpha \in (0, 1)$, `calibrate_conformal_quantile` ranks each slot's normalized probabilities $p_{(1)} \ge \dots \ge p_{(K)}$ and computes nonconformity scores:

- **LAC**: $s_{\text{lac}}(x, y) = 1 - p_y(x)$
- **APS**: $s_{\text{aps}}(x, y) = \sum_{j < r(y)} p_{(j)}(x) + u \cdot p_y(x)$ ($u = 0$ in deterministic mode, $u \sim \operatorname{Uniform}(0, 1)$ when `randomized=True`)
- **RAPS**: $s_{\text{raps}}(x, y) = s_{\text{aps}}(x, y) + \lambda_{\text{reg}} \max(0, r(y) - k_{\text{reg}})$

Selecting the order statistic $m = \lceil (n + 1)(1 - \alpha) \rceil$ (with $\hat{q} = +\infty$ when $m > n$) yields $\mathbb{P}(Y_{n+1} \in \mathcal{C}_\alpha(X_{n+1})) \ge 1 - \alpha$ without distributional assumptions. `CascadeDecisionOrchestrator` evaluates $\mathcal{C}_\alpha(x) = \{y : s(x, y) \le \hat{q}\}$ on raw Stage-1 probabilities before fusing external priors, escalating whenever $|\mathcal{C}_\alpha(x)| \ne 1$ or an abstain option enters $\mathcal{C}_\alpha(x)$.

## 3. Pre-Gated Bayesian PoE and Post-Hoc Scaling (`calibration.py`)

Cardinality-normalized Shannon hesitation is computed on raw Stage-1 probabilities before prior fusion:

$$\tilde{H}_{\text{raw}} = -\frac{1}{\ln K}\sum_{k=1}^K p_{\text{raw},k}\ln p_{\text{raw},k} \in [0, 1]$$

Log-linear prior fusion combines temperature-scaled log-probabilities with prior logits $\pi_k$:

$$\ln p_{\text{fused}}(k) = \frac{1}{T}\ln p_{\text{raw},k} + w_{\text{eff}}(\tilde{H}_{\text{raw}}) \cdot \pi_k$$

Available `gate_mode` schedules are `"hesitant"` ($w \cdot \mathbb{I}[\tilde{H}_{\text{raw}} \ge \tau]$), `"proportional"` ($w \cdot \tilde{H}_{\text{raw}}$), `"confident"` ($w \cdot \mathbb{I}[\tilde{H}_{\text{raw}} < \tau]$), and `"always"` ($w$).

For post-hoc calibration on held-out logits or probabilities:

- **`fit_temperature_scaling`**: Minimizes $L_2$-regularized cross-entropy over $\tau = \ln T$ via an 81-point grid search followed by damped Newton-Raphson refinement.
- **`fit_vector_scaling`**: Fits class-wise scales $a_k$ and biases $b_k$ via L-BFGS with diagonal Gauss-Newton preconditioning under penalty $\frac{\lambda}{2}\sum_k \bigl((a_k - 1/T_0)^2 + b_k^2\bigr)$, shrinking rare classes toward scalar temperature $T_0$.

## 4. Dual-Buffer Photometry and Auto-Saliency Sheets (`visual_compiler.py`)

Applying local contrast stretching before dense retrieval distorts brand packaging colors. `visual_compiler.py` maintains two distinct buffers encoded with pure-stdlib PNG (`IHDR`, `IDAT` filters `0..4`, `IEND`, IEEE CRC32) and PPM (`P6`/`P3`) codecs:

1. **Stage-2A Unsharpened Buffer (`shades_of_gray_white_balance`)**: Normalizes channel illuminants via Minkowski $p=6$ Shades-of-Gray without local sharpening, feeding inner-80% CIE $L^*a^*b^*$ $\Delta E$ similarity (`inner_crop_cie_lab_similarity`) and the residual InfoNCE adapter $e' = \operatorname{norm}(q + \alpha W_2 \operatorname{GELU}(W_1 q))$ (`apply_residual_mlp_adapter`).
2. **Stage-2B Contact Sheet (`compile_dual_buffer_contact_sheet`)**: Applies $3 \times 3$ local contrast stretching on cloned comparison panels, extracts the maximum inter-candidate pixel variance band $V(y) = \operatorname{Mean}_{x,c}(\operatorname{Var}_k R_k(x, y, c))$ (`extract_inter_candidate_saliency_band`), appends $2\times$ saliency and bottom-30% net-weight zoom strips, and optionally unwarps curved containers by $1.15\times$.

## 5. Shortlists, Embedding Index, and Context Pruning (`shortlist.py`, `embedding_index.py`, `context_pruner.py`)

- **Multimodal Embedding Index (`MultimodalEmbeddingIndex`, `VertexMultimodalEmbeddingAdapter`)**: Builds Vertex AI `:predict` payloads for `gemini-embedding-2` and combines cosine or `CSLS` scores, CIE $L^*a^*b^*$ color similarity, and Lidstone-smoothed container aspect priors, triggering `stage2a_fast_exit_gate` when $P_1 - P_2 \ge 0.14$, $Z \ge 1.65$, and top candidates have no sister pack-size collision.
- **Wide-Option Shortlists and Taxonomy Tries (`shortlist_wide_question`, `HierarchicalDecisionRouter`)**: Compresses $K > 26$ options into top-$M$ single-pass shortlists with $\tau = 0.07$ log-softmax cosine priors or two-level `[A-Z]` semantic tries (`mtp_trie`, `beam_2pass`, `hesitation_hybrid`).
- **8-Slot Context Pruning (`prune_agent_context_blocks`)**: Scores up to 8 trajectory blocks per pass over `{"full", "stub", "drop"}` with utilities $u(\text{full}) = p_{\text{full}} + 0.5 p_{\text{stub}}$, $u(\text{stub}) = 0.6 p_{\text{full}} + p_{\text{stub}}$, and $u(\text{drop}) = 0$, solving exact Pareto-frontier dynamic programming under `token_budget` while forbidding `"drop"` on `pinned` blocks.

## 6. UI Grounding, Relational Guards, Contracts, and MCP (`som_navigator.py`, `consistency.py`, `contracts.py`, `eval_suite.py`, `mcp_server.py`)

- **Set-of-Marks Grounding**: Maps detector boxes onto $3 \times 3$ and $4 \times 4$ grids, computes unimodal-safe centroids (`safe_expected_mark_center`), and resolves nested boxes by smallest containing area.
- **Relational and Pack Guards**: Scores probabilistic 3-cycles (`score_probabilistic_triangles`), verifies canonical metric/imperial unit volumes (`validate_pack_quantity_consistency`), and arbitrates concurrent multi-agent locks.
- **Total Contracts and Proper Scoring**: Enforces closed terminal states (`STAGE1_RESOLVED`, `STAGE2_VERIFIED`, `STAGE2A_FAST_EXIT`, `GUARD_OVERRIDDEN`, `ABSTAIN_OOS`, `RETRY_PENDING`, `REJECTED_CONTRACT_VIOLATION`), `<untrusted_data>` isolation, SHA-256 JSONL caching, and evaluation metrics (`brier_murphy_decomposition` into `REL` / `RES` / `UNC`, `ECE`, quantile-binned `ACE`, Macro-F1, `nDCG@k`, `AURC`, exact paired McNemar, Rogan-Gladen prevalence, and cluster bootstrap CIs).
- **MCP `stdio` Server (`dgem-catalog`)**: Exposes `shortlist_candidates`, `evaluate_conformal_decision`, `prune_context_blocks`, and `compile_contact_sheet` over JSON-RPC 2.0 `stdio` without external SDK dependencies.

## 7. Non-Exchangeable & Multi-Slot Conformal, `PriDe`, `LTT`, Submodular MCKP, and Polytope Projection (`v0.5.0`)

- **Mondrian, Weighted, & Simultaneous Multi-Slot Conformal (`conformal.py`, `pkg/mdm/conformal.go`)**:
  - `calibrate_mondrian_conformal_quantiles` and `mondrian_conformal_prediction_set`: Partition calibration scores by class or rarity stratum $g$ so rare tail classes receive stratum-valid finite-sample coverage $\mathbb{P}(Y \in \mathcal{C}_\alpha(X) \mid G = g) \ge 1 - \alpha$.
  - `weighted_conformal_prediction_set`: Implements Tibshirani et al. (2019) likelihood-ratio weighted split conformal prediction under covariate shift $w(x) = dP_{\text{test}}(x)/dP_{\text{calib}}(x)$.
  - `calibrate_multislot_joint_conformal` and `multislot_joint_conformal_sets`: Calibrate the joint max-score statistic $M_i = \max_{s \in \{1..S\}} s(X_{i,s}, Y_{i,s})$ and Bonferroni $\alpha/S$ thresholds across $S$ simultaneous slots so $\mathbb{P}(\forall s: Y_{n+1,s} \in \mathcal{C}_s(X_{n+1})) \ge 1 - \alpha_{\text{family}}$.
- **Cyclic Permutation Debiasing (`PriDe`) & Learn-then-Test (`LTT`) Gates (`calibration.py`, `eval_suite.py`)**:
  - `debiased_cyclic_permutation_probs` and `option_permutation_stability`: Factor out slot-index positional bias across cyclic option shifts and quantify top-1 flip rate and Jensen-Shannon divergence.
  - `calibrate_ltt_funnel_gates`: Jointly selects $(\tau_H, w, T)$ on a calibration fold subject to the finite-sample Hoeffding-Bentkus upper confidence bound $\text{UCB}(\lambda) \le r_{\max}$ at family-wise confidence $1 - \delta$.
- **Submodular Context Pruning & Transitive Polytope Projection (`context_pruner.py`, `consistency.py`)**:
  - `prune_agent_context_blocks(..., redundancy_penalty_weight)`: Discounts marginal block utility by $1 - \gamma \max_{j \in \text{retained}} \operatorname{Jaccard}(S_i, S_j)$ over word-shingle sets so duplicate stack traces do not displace orthogonal root-cause blocks.
  - `project_triangles_to_transitive_consistency`: Solves exact branch-and-bound log-odds correlation clustering on connected components ($\le 11$ nodes, agglomerative refinement for larger components) so projected equivalence graphs satisfy $E_{uv} + E_{vw} - E_{uw} \le 1$ with zero 3-cycle violations.

## 8. Frontier GenAI Extensions: Online `BC-ACI`, Primal-Dual `ParetoBandit`, Calibrated `PRM` PUCT, and Supra-Bayesian Pooling (`v0.7.0`)

- **Online Bias-Corrected Adaptive Conformal Inference (`OnlineBCACITracker`, `evaluate_online_bc_aci_stream` in `conformal.py`, `pkg/mdm/frontier_v070.go`)**:
  Updates $\alpha_{t+1} = \alpha_t + \gamma(\alpha_{\text{target}} - \text{err}_t) + \beta(\bar{\text{err}}_t - \alpha_{\text{target}})$ with online EMA label-prior shift tracking so post-shift coverage tracks $1 - \alpha = 0.9000$ ($N = 100$) without exchangeability assumptions.
- **Primal-Dual Budget-Paced Contextual & Dueling Bandit Router (`PrimalDualParetoBanditRouter`, `evaluate_pareto_bandit_stream` in `bandit_router.py`, `pkg/mdm/frontier_v070.go`)**:
  Maintains discounted ridge regression contexts per arm alongside Bradley-Terry dueling preference log-odds and online dual multipliers $(\lambda_{\text{cost}}, \lambda_{\text{lat}})$, reducing per-query routing cost by $-86.2\%$ (`$0.00582 -> $0.00080`, $N = 100$) with zero budget violations (`0.0000`).
- **Calibrated Process Reward Verifiers & Conformal Branch-and-Bound PUCT (`calibrate_prm_horizon_discount`, `run_calibrated_prm_puct_search` in `prm_tree_search.py`, `pkg/mdm/frontier_v070.go`)**:
  Calibrates depth-discounted rubric step scores $V_d(s) = \gamma_{\text{depth}}^d \mathbb{E}[g \mid p_d(s)]$ and prunes reasoning branches whose conformal upper bound falls below the best verified leaf lower bound, cutting verifier evaluations by $-43.8\%$ (`13.00 -> 7.31` nodes/tree, $N = 100$) with `0.0%` accuracy loss.
- **Covariance-Discounted Supra-Bayesian Pooling (`calibrate_correlated_ensemble_weights`, `covariance_discounted_logit_pool` in `consistency.py`, `pkg/mdm/frontier_v070.go`)**:
  Estimates the $M \times M$ logit-residual covariance matrix $\hat{\Sigma}$ on a calibration split and weights models by $w \propto (\hat{\Sigma} + \lambda I)^{-1}\mathbf{1}$, neutralizing a $4\times$ correlated clone attack ($N_{\text{eff}} = 1.34$, lifting accuracy from `65.00%` to `77.00%`, $p = 0.0357$, and cutting NLL by $-55.7\%$).

## 9. Scope Boundaries, Failure Modes, and Non-Goals

- **Non-Goals**: `multimodal-decision-models` does not train foundation model weights or run GPU tensor kernels; it operates purely on model probability readouts, embeddings, and raw RGB/PNG image buffers using zero external runtime dependencies.
- **Operational Failure Boundaries**:
  - When calibration sample size $n < \lceil(1 - \alpha)/\alpha\rceil$, split conformal order index $m = \lceil(n+1)(1-\alpha)\rceil > n$ returns $\hat{q} = +\infty$ (full option set abstention) by design rather than emitting an invalid finite threshold.
  - When an upstream reasoning endpoint emits unstripped `<think>` or `<thought>` tokens at position 0, `OpenAILogprobBackend.extract_letter_distribution` raises `LogprobAlignmentError` rather than falling back to a flat uniform distribution.
