# Distillation, tuning, and post-tuning recalibration

Training or distilling a compact `2B`–`4B` System-1 decision model from a larger System-2 reasoning teacher introduces three subtle failure modes that standard cross-entropy SFT misses: `<think>` token probability collapse, hard-label posterior erasure, and `must_ask` safety target corruption.

## Prevent `<think>` token leakage on reasoning models

Modern hybrid reasoning checkpoints (`Qwen3`, `DeepSeek-R1`, `Gemma` thinking variants) emit `<think>` or `<|thought|>` tokens at position 0 when default chat templates remain active. Extracting option-letter logprobs at position 0 without disabling thinking collapses accuracy from `68.00%` to `34.67%` (and inflates Brier score to `0.9854`).

`OpenAILogprobBackend` enforces an explicit `<think>` token leakage detector in its first-token distribution parser: if thinking tags appear in `top_logprobs`, it raises `LogprobAlignmentError` fail-closed rather than fabricating a uniform distribution, prompting you to disable thinking templates (`enable_thinking=False`) or strip completed reasoning traces prior to readout.

## Distill System-2 reasoning into System-1 posteriors (Equation 12)

When humans or frontier verifiers genuinely disagree on borderline cases, training a System-1 student on one-hot argmax labels destroys uncertainty information. `reasoning_to_readout_distillation_loss` in `calibration.py` implements soft posterior distillation across teacher reasoning rollouts:

1. Sample or marginalize teacher posteriors $p_{\text{teacher}}(y \mid x, r)$ conditioned on completed reasoning traces $r$.
2. Optionally debias positional letter preferences across cyclic option shifts (`debiased_cyclic_permutation_probs`).
3. Minimize temperature-scaled KL divergence combined with proper Brier regularization so the student matches the full categorical distribution in a single `15 ms`–`60 ms` forward pass.

On `Hanno-Labs/decision-bench` (`N = 300`), Equation 12 posterior distillation reaches `70.00%` slice accuracy (`71.01%` full) while cutting KL divergence to human annotator consensus by `-57.1%` (`0.5822` down to `0.2499`).

```python
from multimodal_decision_models import reasoning_to_readout_distillation_loss

distill_metrics = reasoning_to_readout_distillation_loss(
    teacher_probs={"approve": 0.68, "manual_review": 0.27, "reject": 0.05},
    student_probs={"approve": 0.59, "manual_review": 0.31, "reject": 0.10},
    base_anchor_probs={"approve": 0.55, "manual_review": 0.30, "reject": 0.15},
    teacher_temperature=1.5,
)
```

## Preserve `must_ask` safety states with Selective Teacher Brier SFT

Unconditional knowledge distillation (`0.3 * one_hot(gold) + 0.7 * p_teacher`) is dangerous on agentic `must_ask` states: if an uncalibrated teacher assigns high probability to executing a tool when the gold action is `ask_confirmation`, unconditional blending corrupts `45 / 80` (`56.25%`, and `323 / 350` on the full set) of safety targets!

`selective_teacher_brier_sft_loss` gates teacher soft-target blending strictly on teacher agreement with the safety stratum:

- When the teacher agrees with the gold `must_ask` / acceptable action set, the student learns the teacher's calibrated uncertainty distribution over acceptable alternatives.
- When the teacher disagrees on a `must_ask` or irreversible boundary state, teacher weight drops to `0.0` so `0 / 80` safety targets are corrupted.

```python
from multimodal_decision_models import selective_teacher_brier_sft_loss

sft_result = selective_teacher_brier_sft_loss(
    teacher_probs={"ask_confirmation": 0.15, "execute_tool": 0.85},
    student_probs={"ask_confirmation": 0.78, "execute_tool": 0.22},
    gold_key="ask_confirmation",
    acceptable_keys=["ask_confirmation"],
    gold_weight_alpha=0.30,
)
# sft_result.teacher_accepted is False; target_distribution stays 100% on gold_key.
```

## Run statistical gates and recalibrate after every checkpoint update

Fine-tuning or swapping a Stage-1 checkpoint changes logit sharpness and invalidates previously fitted conformal quantiles. Before promoting a tuned checkpoint to production:

1. **Refit temperature scaling**: Run `fit_temperature_scaling` on a fresh held-out validation split.
2. **Recalibrate conformal thresholds**: Recompute `calibrate_conformal_quantile` or `calibrate_multislot_joint_conformal` (and `calibrate_ltt_funnel_gates` for two-stage cascades).
3. **Verify non-inferiority with Tango's paired test**: Run `tango_paired_non_inferiority_test` + `holm_bonferroni_adjust` (`eval_suite.py`) across accuracy, Brier score, and `must_ask` safety strata to confirm the student checkpoint stays within your non-inferiority margin $\Delta$ before deployment.
