# Mathematical Architecture & Design Guide

## 1. Pre-Gated Bayesian Product-of-Experts (`calibration.py`)

Given an uncalibrated $K$-way model readout $p_{\text{raw}}$, cardinality-normalized Shannon hesitation is measured strictly **before** touching external priors:

$$\tilde{H}_{\text{raw}} = -\frac{1}{\ln K}\sum_{k=1}^K p_{\text{raw},k}\ln p_{\text{raw},k} \in [0, 1]$$

Downstream cascade routing ($\tilde{H}_{\text{raw}} \ge \tau$, default $\tau = 0.16$) and confidence reporting ($C_2 = 1 - \tilde{H}_{\text{raw}}$) read $\tilde{H}_{\text{raw}}$ directly so an adversarial or miscalibrated stage-1 prior cannot suppress epistemic uncertainty on out-of-distribution inputs.

Log-linear prior fusion then combines temperature-scaled log-probabilities with caller-supplied prior logits $\pi_k$:

$$\ln p_{\text{fused}}(k) = \frac{1}{T}\ln p_{\text{raw},k} + w_{\text{eff}}(\tilde{H}_{\text{raw}}) \cdot \pi_k$$

Four gating schedules are supported via `gate_mode`:
- **`"hesitant"`**: $w_{\text{eff}} = w \cdot \mathbb{I}[\tilde{H}_{\text{raw}} \ge \tau]$. Leaves confident model predictions ($\tilde{H}_{\text{raw}} < \tau$) untouched—guaranteeing zero prior-induced flips on low-entropy predictions—while allowing retrieval or base-rate priors to break ties on ambiguous cases ($\tilde{H}_{\text{raw}} \ge \tau$).
- **`"proportional"`**: $w_{\text{eff}} = w \cdot \tilde{H}_{\text{raw}}$. Smoothly interpolates prior influence from $0$ at complete model certainty ($\tilde{H}_{\text{raw}} = 0$) to $w$ at uniform uncertainty ($\tilde{H}_{\text{raw}} = 1$).
- **`"confident"`**: $w_{\text{eff}} = w \cdot \mathbb{I}[\tilde{H}_{\text{raw}} < \tau]$. Conservatively sharpens low-entropy predictions without altering high-entropy distributions.
- **`"always"`**: $w_{\text{eff}} = w$ across all items while still preserving pre-fusion $\tilde{H}_{\text{raw}}$ for cascade escalation.

---

## 2. Stage-2A Fast-Exit & Dense Shortlist Routing (`shortlist.py`)

For wide taxonomies ($K > 26$ up to $10,000+$ SKUs or intents):
1. **Stage-2A Fast-Exit (`stage2a_fast_exit_gate`)**: When dense embedding retrieval produces an unambiguous top-1 match ($P_1 - P_2 \ge 0.14$, shortlist $Z \ge 1.65$, and no sister pack-size collision), the funnel resolves immediately in `0` VLM calls.
2. **Dense Top-$M$ Shortlist (`shortlist_wide_question`)**: On ambiguous or catch-all-sensitive inputs, caller-supplied dense retrieval scores (`CSLS` or cosine similarity) prune $K$ options to top-$M$ ($M \in [6, 25]$) plus an explicit `none_of_the_above` catch-all in a single forward pass—halving pass count and latency compared to multi-pass bracket tournaments while avoiding the recall loss of lexical or BM25 matching on paraphrased queries.
3. **Contrastive Descriptions (`pairwise_diff`)**: Beneficial when option descriptions share long catalog boilerplate and differ only in compact attributes (e.g., `6x100ml` vs `1x100ml`), and optional (`pairwise_diff=False`) on terse label-only schemas.

---

## 3. Set-of-Marks Discrete UI Navigation (`som_navigator.py`)

Continuous coordinate regression degrades sharply on compact UI icons. `som_navigator.py` converts spatial UI grounding into a discrete 4-slot single-pass policy over numbered accessibility or detector candidate boxes (`M1..Mn` + `none_visible`):
- `assign_box_to_grid_cell` and `filter_marks_by_grid_cell` map `[y_min, x_min, y_max, x_max]` boxes onto a $3 \times 3$ coarse spatial grid.
- `select_containing_or_nearest_mark` resolves nested containers by selecting the **smallest-area** mark containing a target point before falling back to Euclidean center distance.

---

## 4. Transitive Triangle Consistency & Multi-Slot Arbiters (`consistency.py`)

- **Triangle Consistency (`check_transitive_triangles`)**: Audits pairwise equivalence graphs for 3-cycles $(A, B, C)$ where two edges satisfy $P(\text{equiv}) \ge \tau_{\text{high}}$ (`0.80`) and the third satisfies $P(\text{equiv}) \le \tau_{\text{low}}$ (`0.25`), catching confident relational contradictions that single-slot entropy misses.
- **Pack-Quantity Guard (`validate_pack_quantity_consistency`)**: Deterministic regex extraction of multipack multipliers (`6x330ml`, `Pack of 12`, `4-pack`) prevents false `exact_duplicate` merges across single-unit and multipack siblings.
- **Swarm Lock Arbiter (`arbitrate_swarm_locks`)**: Deterministically grants, serializes, or escalates concurrent multi-agent file-lock requests in a single pass.
