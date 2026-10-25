# Architecture and Mathematical Design

Single-pass logprob engines, masked diffusion deciders, and black-box sampled models expose incompatible output surfaces and miscalibrated confidence scales. This document specifies the calibration, set-valued abstention, dual-buffer photometry, retrieval, context-pruning, and contract primitives in `multimodal-decision-models` (`v0.4.0`).

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
