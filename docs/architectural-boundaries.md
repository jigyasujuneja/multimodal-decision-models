# Architectural boundaries and hybrid patterns for decision models

To avoid deploying single-pass System-1 decision models where their independence or representation assumptions break down, pair them with cross-encoder margin scorers, sequential alpha-spending budgets, blind-first swarm pre-commitment, constrained joint decoding, and degraded-modality visual contrast checks. Across five pre-registered 100-instance experiments (`EXP-V11-01` through `EXP-V11-05` in [`benchmarks/run_v110_frontier_security_evals_bench.py`](../benchmarks/run_v110_frontier_security_evals_bench.py)), replacing naive standalone calls with these hybrid patterns eliminates `32.0%` catalog-invalid tuples, prevents `100.0%` of unimodal text-prior traps, and restores `95.0%+` finite-sample coverage under sequential and multi-agent shift.

## Summary of the five standalone failure boundaries

The following table maps each structural limitation of a standalone System-1 readout to its failure symptom and the corresponding `v1.1.0` hybrid primitive:

| Boundary | What fails with a standalone decision model | Verified baseline metric (`N = 100`) | Hybrid primitive in `mdm` (`v1.1.0`) | Verified hybrid metric (`N = 100`) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Pointwise reranking saturation** | Independent `[yes, no]` relevance prompts saturate at `0.99+` across top-50 passages and break cross-document ties randomly. | Pointwise `nDCG@10 = 0.0974`; margin-only `no_doc` TNR `= 0.1600` | Stage-1 cross-encoder margin (`HttpCrossEncoderRerankBackend`) → `acurank_trueskill_shortlist` + `conformal_rag_context_filter` | `nDCG@10 = 1.0000`, `no_doc` TNR `= 1.0000`, `1.27` mean MDM passes (`p = 1.58e-30`) |
| **2. Multi-turn trajectory miscoverage and placebo interventions** | Flat per-step conformal $\alpha = 0.10$ compounds error across $T$ turns, while uncalibrated reflection interrupts passing rollouts. | Flat trajectory coverage `= 0.4800`; placebo net value `= 0.7050` | `PolynomialAlphaSpendingTracker` + `NetExpectedValueController` (`NEVC`) | Trajectory coverage `= 0.9500`; `NEVC` net value `= 0.9350` (`-94.83%` wasted turns) |
| **3. Multi-agent peer-conditioning sycophancy** | Exposing peer rationales before initial scoring shifts the nonconformity score mechanism itself, breaking importance-weighted conformal bounds. | Uncertain-panel coverage `= 0.0000` (both unweighted and covariate-weighted) | `evaluate_blind_first_consensus` (Round-1 solo pre-commitment + `JSD` sycophancy veto) | Uncertain-panel coverage `= 0.9571`, overall coverage `= 0.9700` (`p = 1.36e-20`) |
| **4. Coupled multi-slot factorization collapse** | Co-denoising $S$ slots in one pass models marginal $P(Y_s \mid X)$ rather than joint $P(Y_1, \dots, Y_S \mid X)$ when slots share high mutual information. | `32.00%` catalog-invalid tuples, `0.6000` exact joint accuracy | `decode_constrained_joint_tuple` (0 extra RPCs) + selective 2-pass `demask_coupled_slots` | `0.00%` invalid tuples, `0.8800` exact joint accuracy at `1.30` mean RPCs (`p = 5.77e-08`) |
| **5. Unimodal text-prior shortcuts** | Dominant title or OCR priors let a VLM emit `0.92+` confidence without attending to contradicting pixels (damaged seals or sister SKUs). | `25/25` (`100.0%`) prior-trap errors, selective `AURC = 0.2371` | `compile_degraded_modality_twin` + `compute_visual_evidence_gain` (`AVCD` KL guard) | `0/25` (`0.0%`) prior-trap errors, selective `AURC = 0.1948` (`p = 5.96e-08`) |

## Combine cross-encoder margins with `AcuRank` and conformal RAG filtering

Do not use a binary or categorical System-1 decision model as a pointwise top-100 document reranker. Because masked-diffusion and single-token classifiers normalize probabilities over discrete labels (`relevant` versus `irrelevant`) rather than across candidate documents, plausible distractor passages cluster near `1.0` and lose relative margin ordering. Conversely, margin-trained cross-encoders (`Qwen3-Reranker`, `Jina-Reranker-v3`, `BGE-Reranker`) rank top-100 candidates cleanly without calibrated `"none of these documents answer the query"` abstention probabilities.

Use the three-tier pipeline in [`reranking.py`](../src/multimodal_decision_models/reranking.py):

1. **Score top-50 to top-100 candidates with a margin cross-encoder** (`HttpCrossEncoderRerankBackend`).
2. **Resolve top-6 comparative uncertainty and explicit abstention** with `acurank_trueskill_shortlist`, which tracks Gaussian `(mu, sigma)` skill posteriors alongside an explicit `__no_relevant_doc__` option and halts early when top-$k$ credible intervals separate.
3. **Filter retrieved passages before generation** with `conformal_rag_context_filter`, which quarantines adversarial prompt injections and abstains when `__no_relevant_doc__` dominates.

```python
from multimodal_decision_models import (
    MockCallableBackend,
    RerankCandidate,
    acurank_trueskill_shortlist,
    conformal_rag_context_filter,
)


def _offline_comparative_scorer(state, questions, images):
    del state, questions, images
    return {
        "comparative_select": {
            "doc_api_v2": 0.86,
            "doc_legacy_v1": 0.09,
            "doc_unrelated": 0.02,
            "__no_relevant_doc__": 0.03,
        }
    }


candidates = [
    RerankCandidate(doc_id="doc_api_v2", text="Set timeout_seconds=10 in HttpCrossEncoderRerankBackend.", stage1_margin_score=4.2),
    RerankCandidate(doc_id="doc_legacy_v1", text="Deprecated v1 XML configuration.", stage1_margin_score=3.9),
    RerankCandidate(doc_id="doc_unrelated", text="Cafeteria menu for Thursday.", stage1_margin_score=-1.8),
]

# Swap MockCallableBackend for DgemSystemOneBackend("https://stage1.example.com") in production.
shortlist = acurank_trueskill_shortlist(
    query="How do I configure the cross-encoder timeout in v2?",
    candidates=candidates,
    mdm_backend=MockCallableBackend("comparative-judge", _offline_comparative_scorer),
    top_k=1,
    stage1_shortlist_size=3,
)
assert shortlist.selected_doc_ids == ("doc_api_v2",)

filtered = conformal_rag_context_filter(
    candidates=candidates,
    relevance_probabilities={"doc_api_v2": 0.91, "doc_legacy_v1": 0.14, "doc_unrelated": 0.02},
    q_hat=0.25,
)
assert filtered.retained_doc_ids == ("doc_api_v2",)
```

## Control multi-turn trajectory error budgets and intervention value

Applying a fixed conformal error level $\alpha = 0.10$ at every turn of a $T$-step agent rollout allows trajectory-level miscoverage to compound toward $1 - (1 - \alpha)^T$, dropping empirical 5-step coverage to `48.0%` under Markov state drift (`EXP-V11-02`). At the same time, intervening whenever an uncalibrated critic predicts `P(fail) > 0.50` often degrades net task value below a no-intervention baseline, while frequency-matched random interventions (`placebo`) produce apparent lift by accidentally perturbing stalled loops.

[`trajectory_control.py`](../src/multimodal_decision_models/trajectory_control.py) addresses both failure modes:

- **`PolynomialAlphaSpendingTracker`**: Allocates step budget $\alpha_t = \gamma_{\text{Markov}} \cdot \alpha_{\text{total}} \cdot w_t / \sum_j w_j$ with $\gamma_{\text{Markov}} \in (0, 1]$ and $w_t \propto t^{-p}$ (or infinite-horizon Riemann $\zeta(p)$ normalization when `horizon_max_steps=None`), preserving family-wise trajectory coverage $\ge 1 - \alpha_{\text{total}}$.
- **`NetExpectedValueController` (`NEVC`)**: Triggers mid-episode intervention only when Platt-calibrated failure probability $p_{\text{fail}}$ exceeds the positive-gain threshold:

$$p_{\text{fail}} > \frac{C_{\text{int}} + V \cdot p_{\text{reg}}}{V \cdot (p_{\text{fix}} + p_{\text{reg}})}$$

- **`evaluate_placebo_controller_baseline`**: Benchmarks your learned controller against a deterministic `HMAC`-seeded frequency-matched random interrupter so that uncalibrated intervention noise never masquerades as causal steering gain.

```python
from multimodal_decision_models import (
    NetExpectedValueController,
    PolynomialAlphaSpendingTracker,
)

tracker = PolynomialAlphaSpendingTracker(
    calib_probs=[{"proceed": 0.92, "halt": 0.08}] * 40,
    calib_labels=["proceed"] * 40,
    alpha_total=0.10,
    exponent_p=1.6,
    horizon_max_steps=5,
    markov_slack_factor=0.85,
)
step1_set = tracker.prediction_set_for_step(step_index=1, probs={"proceed": 0.88, "halt": 0.12})

controller = NetExpectedValueController(
    task_value=1.0,
    intervention_cost=0.08,
    fix_probability_if_failing=0.45,
    regression_probability_if_passing=0.04,
)
decision = controller.evaluate(raw_fail_probability=0.42)
```

## Enforce blind-first pre-commitment in multi-agent swarms

When multiple judge or router agents view peer outputs before recording their own posteriors, conformity bias collapses inter-agent variance without improving accuracy. Because peer conditioning alters $P(Y \mid S(X))$ rather than $P(X)$ alone, covariate-shift importance weights cannot repair conformal calibration (`0.00%` uncertain-panel coverage in `EXP-V11-03`).

Use `evaluate_blind_first_consensus` in [`consistency.py`](../src/multimodal_decision_models/consistency.py) (and `EvaluateBlindFirstConsensus` in Go `pkg/mdm`):

1. Collect independent Round-1 solo probability maps from every agent before sharing peer messages.
2. Compute conformal prediction sets strictly from the pre-committed Round-1 ensemble distribution.
3. Veto Round-2 posterior shifts whose Jensen-Shannon divergence exceeds `max_unjustified_jsd_bits` unless the shifting agent cites verified new tool or retrieval evidence (`verified_new_evidence_agents`).

```python
from multimodal_decision_models import evaluate_blind_first_consensus

consensus = evaluate_blind_first_consensus(
    solo_agent_probs={
        "agent_a": {"approve": 0.52, "escalate": 0.48},
        "agent_b": {"approve": 0.46, "escalate": 0.54},
    },
    peer_conditioned_agent_probs={
        "agent_a": {"approve": 0.94, "escalate": 0.06},
        "agent_b": {"approve": 0.93, "escalate": 0.07},
    },
    q_hat=0.55,
    verified_new_evidence_agents=(),
    max_unjustified_jsd_bits=0.08,
)
assert bool(consensus.sycophancy_flagged_agents) is True
assert set(consensus.robust_conformal_set) == {"approve", "escalate"}
```

## Decode coupled multi-slot schemas with constrained MAP and `DEMASK`

Parallel 1-pass multi-slot readouts (`MAX_SLOTS_PER_PASS = 8`) assume conditional independence across slots given the prompt. When slots obey hard domain constraints (for example, `(category, brand, subline, pack_tier)`), taking per-slot argmax independently produces `32.0%` invalid combinations (`EXP-V11-04`).

[`hierarchy.py`](../src/multimodal_decision_models/hierarchy.py) provides two escalating remedies:

- **Zero-extra-RPC constrained joint MAP (`decode_constrained_joint_tuple`)**: Scores every schema-valid tuple $\mathbf{y} \in \mathcal{V}$ by $\sum_s \log P_s(y_s \mid x)$ and returns exact joint posteriors restricted to $\mathcal{V}$, cutting invalid tuples to `0.0%` with zero extra model calls.
- **Selective 2-pass `DEMASK` (`demask_coupled_slots`)**: Estimates empirical pairwise mutual information on calibration labels, commits confident anchor slots in Pass 1, and triggers a conditional Pass 2 only when anchor entropy exceeds `anchor_hesitation_threshold` (`1.30` mean RPCs for `+28.0` accuracy points in `EXP-V11-04`).

```python
from multimodal_decision_models import decode_constrained_joint_tuple

joint = decode_constrained_joint_tuple(
    slot_probabilities={
        "brand": {"acme": 0.58, "zenith": 0.42},
        "tier": {"enterprise": 0.55, "starter": 0.45},
    },
    valid_tuples=[
        {"brand": "acme", "tier": "starter"},
        {"brand": "zenith", "tier": "enterprise"},
    ],
)
```

## Detect unimodal text-prior shortcuts with visual contrastive evidence gain

When a catalog listing title says *"Intact Factory Seal - 12 Pack"* while the user's photo shows a punctured blister pack or a 6-pack sister SKU, a VLM can output `0.94` confidence from text priors alone without reading the image pixels.

Use `compile_degraded_modality_twin` and `compute_visual_evidence_gain` in [`visual_compiler.py`](../src/multimodal_decision_models/visual_compiler.py) to run a selective Adversarial Visual Contrastive Decoding (`AVCD`) check on high-stakes visual predictions:

1. Generate a deterministic box-blurred, chroma-desaturated twin of the input `ImageBuffer`.
2. Measure $\mathrm{KL}(p_{\text{clean}} \parallel p_{\text{degraded}})$.
3. Flag `veto_fast_exit = True` and escalate to Stage-2 Dual-Buffer Contact-Sheet verification whenever $\max_k p_{\text{clean}}(k) \ge 0.70$ while $\mathrm{KL}(p_{\text{clean}} \parallel p_{\text{degraded}}) < 0.12\text{ nats}$.

```python
from multimodal_decision_models import (
    ImageBuffer,
    compile_degraded_modality_twin,
    compute_visual_evidence_gain,
)

clean_image = ImageBuffer(16, 16, bytes([180, 40, 40]) * 256)
degraded_twin = compile_degraded_modality_twin(clean_image, blur_radius=6, chroma_desaturate=0.85)

audit = compute_visual_evidence_gain(
    clean_probabilities={"intact_12pk": 0.93, "damaged_or_mismatch": 0.07},
    degraded_probabilities={"intact_12pk": 0.91, "damaged_or_mismatch": 0.09},
    min_kl_nats=0.12,
    high_confidence_floor=0.70,
)
assert audit.veto_fast_exit is True
```

## Pair stateless decision endpoints with stateful MDM orchestration

Managed single-call decision endpoints such as `GeminiDecisionsBackend` (`POST /v1beta/decisions` on `models/gemini-decision-1` for `noul`, `choice` with `K <= 255`, `include_none_of_above`, and `score` with `L <= 255`) score structured slots in a single stateless HTTP round-trip, yet leave multi-turn risk budgeting, wide-catalog reduction, token compaction, and multi-model cost pacing to the caller. `multimodal-decision-models` wraps stateless endpoints and GA Vertex verifiers (`VertexGeminiBackend.from_ga_preset` across `gemini-3.5-flash-lite`, `gemini-3.5-flash`, `gemini-3.8-flash`, and `gemini-3.5-pro`) with four stateful or pre-call layers verified in the `1,200`-episode `MDM-EpisodeBench` `Tier-2` Receipt-Grounded Replay suite ([`docs/security-and-eval-science.md`](security-and-eval-science.md)):

1. **Pre-call context `MCKP` and visual token compression (`context_pruner.py`, `visual_compiler.py`, `audio_video_decision.py`)**: Stateless APIs bill and attend across all submitted prompt tokens. Pruning trajectory blocks with `prune_agent_context_blocks` (`{full, stub, drop}`) while compiling `2x` saliency and pack-weight contact sheets plus submodular video keyframes cuts mean episode tokens from `22,687` to `11,225` (`-50.5%`) and produces a `+0.1681` super-additive net-utility synergy ($\hat{\tau}_{MK}$, Holm `p = 1.67e-22`).
2. **Wide-catalog shortlisting beyond `K = 255` (`shortlist_wide_question`)**: When a SKU catalog or tool registry exceeds `255` options (or `254` explicit options when `include_none_of_above=True`), `GeminiDecisionsBackend` bridges the slot through `shortlist_wide_question` before serialization and normalizes returned log-likelihoods in log-space alongside `abstention_log_likelihood`.
3. **Out-of-fold Mondrian calibration and sequential trajectory $\alpha$-spending (`experiment_protocol.py`, `trajectory_control.py`)**: Raw single-call log-likelihoods neither adjust for class-stratum sharpness nor track cumulative error across 6-turn rollouts. Pairing `fit_oof_calibrated_conformal_bundle` (`C`) with `PolynomialAlphaSpendingTracker` and `NetExpectedValueController` (`K`) yields `89.72%` Mondrian coverage (`1.05%` maximum stratum gap) and a `+0.1521` interaction synergy ($\hat{\tau}_{CK}$, Holm `p = 4.44e-15`).
4. **Online budget-paced model routing (`PrimalDualParetoBanditRouter`)**: Feeding calibrated uncertainty features into `PrimalDualParetoBanditRouter` routes routine turns to `models/gemini-decision-1` or `gemini-3.5-flash-lite` and escalates `30.9%` of ambiguous turns to `gemini-3.8-flash` or `gemini-3.5-pro`, yielding a `+0.1104` calibration-by-router synergy ($\hat{\tau}_{CP}$) and reducing episode cost by `84.3%` relative to the `R0` generative ceiling (`$0.0482` to `$0.0076`).

```python
from multimodal_decision_models import (
    GeminiDecisionsBackend,
    ModelArmSpec,
    PrimalDualParetoBanditRouter,
    VertexGeminiBackend,
)

stage1_decisions = GeminiDecisionsBackend(
    api_key="example-header-only-api-key",
    model="models/gemini-decision-1",
    include_none_of_above=True,
)
stage2_flash = VertexGeminiBackend.from_ga_preset(
    "gemini-3.8-flash",
    project="example-prod-project",
    location="us-central1",
    access_token_fn=lambda: "ephemeral-oauth-token",
)

router = PrimalDualParetoBanditRouter(
    arms=[
        ModelArmSpec(arm_id=stage1_decisions.model, cost_usd=0.0004, latency_ms=140.0),
        ModelArmSpec(arm_id="gemini-3.5-flash-lite", cost_usd=0.0008, latency_ms=260.0),
        ModelArmSpec(arm_id=stage2_flash.model, cost_usd=0.0045, latency_ms=920.0),
    ],
    context_dim=3,
    budget_per_query_usd=0.0018,
    latency_sla_ms=1200.0,
)
routing_choice = router.select_arm([1.0, 0.28, 0.50])
router.update(routing_choice.chosen_arm, [1.0, 0.28, 0.50], reward=0.92)
assert routing_choice.chosen_arm in {
    "models/gemini-decision-1",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
}
```

## Related guides

- Continue to **[Adversarial security hardening and statistical evaluation science](security-and-eval-science.md)** for `HMAC` canary envelopes, conformal poisoning defense, and `PPI++` evaluation estimators.
- See **[Calibration, conformal sets, and risk control](calibration-and-conformal.md)** and **[Wide-catalog routing and taxonomy tries](wide-catalog-routing.md)** for single-pass calibration foundations.
