# Architecture and Mathematical Design

## 1. Pre-Gated Bayesian Product-of-Experts (`calibration.py`)

Given an uncalibrated $K$-way model readout $p_{\text{raw}}$, we measure cardinality-normalized Shannon hesitation strictly before fusing external priors:

$$\tilde{H}_{\text{raw}} = -\frac{1}{\ln K}\sum_{k=1}^K p_{\text{raw},k}\ln p_{\text{raw},k} \in [0, 1]$$

Downstream cascade routing ($\tilde{H}_{\text{raw}} \ge \tau$, default $\tau = 0.16$) and confidence reporting ($C_2 = 1 - \tilde{H}_{\text{raw}}$) read $\tilde{H}_{\text{raw}}$ directly so an adversarial or miscalibrated stage-1 prior cannot suppress epistemic uncertainty on out-of-distribution inputs.

Log-linear prior fusion combines temperature-scaled log-probabilities with caller-supplied prior logits $\pi_k$:

$$\ln p_{\text{fused}}(k) = \frac{1}{T}\ln p_{\text{raw},k} + w_{\text{eff}}(\tilde{H}_{\text{raw}}) \cdot \pi_k$$

Four gating schedules are available via `gate_mode`:

- **`"hesitant"`**: $w_{\text{eff}} = w \cdot \mathbb{I}[\tilde{H}_{\text{raw}} \ge \tau]$. Leaves confident predictions ($\tilde{H}_{\text{raw}} < \tau$) untouched (zero prior-induced flips on low-entropy items) while letting retrieval or base-rate priors break ties on ambiguous cases ($\tilde{H}_{\text{raw}} \ge \tau$).
- **`"proportional"`**: $w_{\text{eff}} = w \cdot \tilde{H}_{\text{raw}}$. Scales prior influence smoothly from $0$ at model certainty ($\tilde{H}_{\text{raw}} = 0$) to $w$ at uniform uncertainty ($\tilde{H}_{\text{raw}} = 1$).
- **`"confident"`**: $w_{\text{eff}} = w \cdot \mathbb{I}[\tilde{H}_{\text{raw}} < \tau]$. Sharpens low-entropy predictions without altering high-entropy distributions.
- **`"always"`**: $w_{\text{eff}} = w$ across items while preserving pre-fusion $\tilde{H}_{\text{raw}}$ for cascade escalation.

## 2. Stage-2A Fast-Exit, Shortlist Routing, and Taxonomy Tries (`shortlist.py`, `hierarchy.py`)

For wide taxonomies ($K > 26$ up to $10,000+$ SKUs or intents):

1. **Stage-2A Fast-Exit (`stage2a_fast_exit_gate`)**: When dense embedding retrieval produces an unambiguous top-1 match ($P_1 - P_2 \ge 0.14$, shortlist $Z \ge 1.65$, and no sister pack-size collision), the funnel resolves in `0` VLM calls.
2. **Dense Top-$M$ Shortlist (`shortlist_wide_question`)**: Prunes $K$ options to top-$M$ ($M \in [6, 25]$) plus an explicit `none_of_the_above` catch-all in 1 forward pass, compared to $1 + \lceil\lceil K/20 \rceil / 8\rceil$ passes for packed tournament brackets (`packed_bracket_pass_count`).
3. **Log-Softmax Cosine Priors (`normalize_cosine_prior_logits`)**: Converts compressed raw cosines ($[0.25, 0.55]$) into calibrated log-probabilities $\pi_k = (\cos_k / \tau) - \operatorname{logsumexp}_j(\cos_j / \tau)$ at default $\tau = 0.07$ before `pre_gated_bayesian_poe_fusion`.
4. **Opt-In Contrastive Cues (`pairwise_diff`)**: Defaults to `False` so terse intent descriptions remain clean; enable `pairwise_diff=True` when catalog entries share repetitive boilerplate and differ only in compact attributes such as `6x330ml` vs `1x330ml`.
5. **Two-Level Semantic Taxonomy Tries (`build_prefix_taxonomy_trie`, `HierarchicalDecisionRouter`)**: Keeps coarse and fine slots within $\le 26$ `[A-Z]` options, buckets opaque `option_N` schemas by description stems, aligns shared cross-domain suffixes (`role_<suffix>`), and supports 1-pass `mtp_trie`, 2-pass `beam_2pass`, and `hesitation_hybrid`.

## 3. Set-of-Marks Discrete UI Navigation (`som_navigator.py`)

Continuous coordinate regression degrades on compact UI icons. `som_navigator.py` converts spatial UI grounding into a discrete 4-slot single-pass policy over numbered accessibility or detector candidate boxes (`M1..Mn` + `none_visible`):

- `assign_box_to_grid_cell`, `assign_box_to_grid_4x4`, and `grid_4x4_cell_bounds` map `[y_min, x_min, y_max, x_max]` boxes onto $3 \times 3$ and $4 \times 4$ spatial grids.
- `safe_expected_mark_center` computes the probability-weighted centroid when dispersion stays unimodal and inside a candidate box, snapping deterministically to the highest-probability mark when bimodal splits or non-convex layouts place the centroid in dead space.
- `select_containing_or_nearest_mark` resolves nested containers by selecting the smallest-area mark containing a target point before falling back to Euclidean center distance.

## 4. Continuous Triangle Consistency and Multi-Slot Arbiters (`consistency.py`)

- **Continuous Triangle Contradiction (`score_probabilistic_triangles`)**: Scores 3-cycles $(A, B, C)$ by $\max\bigl(p_{ab}p_{bc}(1-p_{ac}),\; p_{ab}p_{ac}(1-p_{bc}),\; p_{bc}p_{ac}(1-p_{ab})\bigr) \in [0, 1]$, catching soft relational contradictions below hard threshold cutoffs (`check_transitive_triangles`).
- **Pack-Quantity and Unit-Measure Guard (`parse_unit_measure`, `validate_pack_quantity_consistency`)**: Normalizes multipack counts (`6x330ml`, `Pack of 12`) and canonical metric volumes or weights (`0.5L` vs `500ml`, `1kg` vs `1000g`, `16 fl oz` vs `473ml`) to block false `exact_duplicate` merges across distinct sizes.
- **Swarm Lock Arbiter (`arbitrate_swarm_locks`)**: Grants, serializes, or escalates concurrent multi-agent file-lock requests in a single pass.

## 5. Total Contracts and Six-Tier Evaluation (`contracts.py`, `eval_suite.py`)

Production inference pipelines fail closed rather than substituting fabricated `0.98` or uniform `1/K` probabilities when logprobs are missing or misaligned:

- **Terminal State Machine (`DecisionTerminalState`)**: Every `SlotReadout` and `DecisionResult` ends in `RESOLVED`, `ABSTAINED`, `RETRY_PENDING` (transient HTTP `429`/`503` or socket timeout), or `REJECTED_CONTRACT_VIOLATION` (unaligned `top_logprobs`, non-finite logits, or malformed schema payloads).
- **Single-Token Enum Bijection**: `VertexGeminiBackend` and `OpenAILogprobBackend` bind multi-token option keys to single-token uppercase letters (`A..Z`) at the wire boundary so token logprobs map 1:1 back to domain choices.
- **Untrusted Input Isolation**: `sanitize_untrusted_text` and `wrap_untrusted_state` strip control characters, neutralize fake closing tags, and fence OCR or user text inside `<untrusted_data>` blocks.
- **Content-Addressed Caching (`ContentAddressedDecisionCache`)**: Hashes canonical policy specifications, sanitized state, image payload digests, and calibration options with SHA-256 while excluding `RETRY_PENDING` and `REJECTED_CONTRACT_VIOLATION` outcomes from persistence.
- **Proper-Scoring and Statistical Metrics (`eval_suite.py`)**: Computes multiclass Brier score, NLL, `ECE`, `AUROC`, selective `AURC`, exact paired McNemar $p$-values, Rogan-Gladen prevalence adjustment, Cohen's $\kappa$, and cluster-level paired bootstrap intervals.
