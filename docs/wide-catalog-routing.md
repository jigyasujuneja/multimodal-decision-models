# Wide-catalog routing and taxonomy tries

Single-token decision readouts achieve constant-time probability extraction when every candidate maps to a single vocabulary token. Once your catalog grows beyond `26` uppercase letters (`[A-Z]`) to hundreds of agent tools or `10,000+` retail SKUs, naive multi-character labels fracture into multiple BPE sub-tokens and corrupt first-token logprobs.

## Select the right routing strategy for catalog size $K$

Use `recommend_wide_routing_strategy` in `hierarchy.py` to select the benchmarked readout geometry for your catalog cardinality $K$ and backend capabilities:

```python
from multimodal_decision_models import recommend_wide_routing_strategy

plan = recommend_wide_routing_strategy(
    num_options=140,
    supports_az_aa_52=True,
    supports_extended_255=False,
    has_embeddings=False,
    supports_multi_slot_canvas=True,
)
# plan.strategy == "mtp_partitioned_1pass" (1.0 pass, zero fine-slot aliasing)
```

| Catalog size $K$ | Recommended strategy | Forward passes | Why this regime wins |
| :--- | :--- | :---: | :--- |
| **$2 \le K \le 26$** | `direct_az` (`A..Z`) | `1.0` | Universal single-token bijection across every LLM and `/v1/systemone` canvas. |
| **$27 \le K \le 52$** | `direct_az_aa` (`A..Z, AA..AZ`) | `1.0` | Supported natively by `dgem` 52-option bijections with zero multi-pass overhead. |
| **$53 \le K \le 255$** (verified BPE) | `direct_universal_255` (`UNIVERSAL_256_DECISION_CODES`) | `1.0` | Curated 256-code table (`A..Z, AA..PT`) omitting BPE-splitting pairs such as `BQ`, `CJ`, `FQ`, and `GZ` (`255/255` single-token across Qwen3, DeepSeek-V3, Gemma-3, Llama-3.2, and SmolLM2). |
| **$K > 26$ with embeddings** | `csls_shortlist_1pass` (`shortlist_wide_question`) | `1.0` | Compresses $10,000+$ items to Top-20 `CSLS` candidates + `none_of_the_above` catch-all + $\tau = 0.07$ cosine log-softmax `prior_logits`. |
| **$53 \le K \le 176$ without embeddings** | `mtp_partitioned_1pass` (`HierarchicalDecisionRouter`) | `1.0` | Places 1 coarse slot + up to 7 disjoint fine-band slots on an 8-slot `/v1/systemone` canvas, eliminating marginal fine-code aliasing. |
| **$K > 176$ without embeddings** | `hesitation_hybrid` (`HierarchicalDecisionRouter`) | `~1.2` | Runs 1-pass taxonomy readout when coarse $\tilde{H}_1 < \tau$, and triggers a conditional Pass-2 beam over top-$b$ coarse branches only when hesitant. |

> **Important:** Avoid unpartitioned 1-pass `mtp_trie` when $K > 26$ and multiple coarse categories share fine codes `A..Z`. Under parallel decoding, slot 2 marginalizes across coarse parents ($\sum_c P(c) P(f \mid c)$), which aliases popular fine indices across unrelated domains. Either partition fine slots (`mtp_partitioned_1pass`), apply Sinkhorn-Knopp iterative scaling (`sinkhorn_dealiased_trie_decode`), or use `hesitation_hybrid`.

## Route $10,000+$ SKUs with `CSLS` shortlists and cosine priors

In high-dimensional embedding spaces, "hub" catalog vectors appear as nearest neighbors to many unrelated queries. `MultimodalEmbeddingIndex` applies Cross-Domain Similarity Local Scaling (`csls_hubness_scores`) combined with CIE $L^*a^*b^*$ color similarity and Lidstone container aspect priors, then converts retrieval similarities into calibrated log-softmax `prior_logits` (`normalize_cosine_prior_logits`, $\tau = 0.07$):

```python
from multimodal_decision_models import (
    DecisionQuestion,
    MultimodalEmbeddingIndex,
    shortlist_wide_question,
)

index = MultimodalEmbeddingIndex()
index.upsert("sku_berry_330ml", [0.91, 0.12, 0.04], title="Sparkling Berry Can 330ml", pack_size=1)
index.upsert("sku_berry_6pk", [0.89, 0.14, 0.05], title="Sparkling Berry 6x330ml Multipack", pack_size=6)
index.upsert("sku_citrus_330ml", [0.15, 0.92, 0.08], title="Sparkling Citrus Can 330ml", pack_size=1)

retrieval = index.query([0.90, 0.13, 0.04], top_k=20, use_csls=True)
if retrieval.fast_exit:
    # Unambiguous Stage-2A match (P1 - P2 >= 0.14 and Z >= 1.65): 0 VLM calls needed!
    selected_sku = retrieval.fast_exit_id
else:
    reduced_question = shortlist_wide_question(
        question=DecisionQuestion(
            type="choice",
            instructions="Match the shelf crop to the exact catalog SKU.",
            criteria={
                "sku_berry_330ml": "Sparkling Berry single 330ml can",
                "sku_berry_6pk": "Sparkling Berry 6x330ml multipack",
                "sku_citrus_330ml": "Sparkling Citrus single 330ml can",
            },
        ),
        scores=retrieval.prior_logits,
        top_k=20,
        catch_all_key="none_of_the_above",
        pairwise_diff=True,
    )
```

Note that `pairwise_diff=True` extracts sibling-unique discriminative tokens (such as `6x330ml` vs. `single`) for repetitive retail SKU titles; keep `pairwise_diff=False` (the default) for concise intent or tool names.

## Debias option order with `PriDe` cyclic permutations

Autoregressive and diffusion models can exhibit mild positional preference toward early letters (`A`, `B`). For offline evaluation, teacher distillation, or high-stakes ties, `debiased_cyclic_permutation_probs` runs $C$ cyclic shifts of the option order, estimates the prior positional bias vector, and factors it out in log-space—cutting option-order flip rates on `MDM-Bench` from `10.25%` to `6.50%`.
