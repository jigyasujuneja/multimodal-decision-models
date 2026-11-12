# Production deployment, Terraform, and MCP operations

Moving calibrated decision models into production requires three operational guarantees beyond offline accuracy: transient upstream timeouts must never silently become false-negative denials, repeated identical payloads should resolve deterministically from cache without redundant GPU calls, and infrastructure must enforce digest-pinned images with zero public unauthenticated endpoints.

## Closed Level-4 Total-Contract state machine (`contracts.py`)

When `total_contract_mode=True` (the default on `CascadeDecisionOrchestrator`), every decision resolves into one of eight closed `DecisionTerminalState` values:

| Terminal state | Meaning | Downstream caller action |
| :--- | :--- | :--- |
| `STAGE2A_FAST_EXIT` | Resolved directly by Stage-2A embedding/photometric margin (`0` model RPCs). | Commit decision immediately. |
| `STAGE1_RESOLVED` | Resolved by Stage 1 with singleton conformal set and $\tilde{H}_{\text{raw}} < \tau$. | Commit decision immediately. |
| `STAGE2_VERIFIED` | Escalated on Stage-1 ambiguity and verified by Stage-2 multimodal/reasoning backend. | Commit verified decision. |
| `GUARD_OVERRIDDEN` | Adjusted deterministically by Layer-5 pack-size or triangle-consistency guardrails. | Commit guarded decision and log override metric. |
| `ABSTAIN_OOS` | Model or conformal set explicitly abstained (`none_of_the_above`, `ambiguous_or_oos`). | Route to human review or clarification prompt. |
| `INVALID_TRIE_PATH` | Unconstrained multi-slot readout produced a tuple outside `TaxonomyTrie`. | Re-decode with `mtp_partitioned_1pass` or `hesitation_hybrid`. |
| `RETRY_PENDING` | Upstream HTTP `408`/`429`/`503` or `sla_timeout_sec` budget exhausted. | Enqueue for asynchronous retry; **never** treat as a `0` payout or hard rejection. |
| `REJECTED_CONTRACT_VIOLATION` | Malformed schema, NaN float, or unaligned single-token logprobs. | Fail closed (`HTTP 502`/`503`) and alert on contract violation metric. |

## Content-addressed SHA-256 decision caching

`ContentAddressedDecisionCache` canonicalizes arbitrary nested states, image byte SHA-256 digests, question schemas, temperature, and conformal parameters into an injective SHA-256 key (`compute_decision_cache_key`) backed by an append-only JSONL ledger:

- Caches resolved decisions (`STAGE2A_FAST_EXIT`, `STAGE1_RESOLVED`, `STAGE2_VERIFIED`, `ABSTAIN_OOS`) including `conformal_set` and `epistemic_variance`.
- Never caches `RETRY_PENDING` or `REJECTED_CONTRACT_VIOLATION` states, so transient upstream outages recover automatically on retry.

## Connect managed Gemini decision and Vertex verifier endpoints

For deployments that pair managed Google AI Studio or Vertex AI endpoints with `CascadeDecisionOrchestrator`, [`backends.py`](../src/multimodal_decision_models/backends.py) provides `GeminiDecisionsBackend` and `VertexGeminiBackend.from_ga_preset`:

- **`GeminiDecisionsBackend` (`POST /v1beta/decisions` on `models/gemini-decision-1`)**: Encodes structured state inside `<untrusted_data>` and scores binary `noul` (`yes_log_likelihood` and `no_log_likelihood`), categorical `choice` (`K <= 255`, or `K <= 254` explicit options when `include_none_of_above=True`), and ordered rubric `score` buckets (`L <= 255`, recording `expected_score` and `variance` in `raw_metadata`). When a `choice` slot exceeds the `255`-option wire limit, `build_decisions_payload` automatically bridges the catalog through `shortlist_wide_question` (`shortlist_top_k=200` default) before dispatching the request.
- **Zero-trust credential and fail-closed transport invariants**: `GeminiDecisionsBackend` requires `HTTPS` for remote hosts (`http://` is permitted exclusively on `127.0.0.1` and `localhost` test servers), sends credentials only in the `x-goog-api-key` HTTP header (`api_key` parameter or `GEMINI_API_KEY` environment variable), rejects query-string secret parameters (`?key=`, `?api_key=`, `?access_token=`) and CR/LF control characters with `SchemaContractViolationError`, normalizes option log-likelihoods in log-space without uniform fallbacks, and maps `HTTP 429`, `502`, `503`, and `504` responses to `UpstreamTransientError` so `CascadeDecisionOrchestrator` emits `RETRY_PENDING`.
- **Approved GA Vertex Gemini presets (`VertexGeminiBackend.from_ga_preset`)**: Configures `VertexGeminiBackend` for the four public GA models in `GA_VERTEX_GEMINI_MODELS`: `gemini-3.5-flash-lite` (`30.0 s` fast readout), `gemini-3.5-flash` (`45.0 s` balanced), `gemini-3.8-flash` (`60.0 s` Stage-2 flash verifier), and `gemini-3.5-pro` (`90.0 s` Stage-2 pro verifier).

```python
from multimodal_decision_models import (
    DecisionQuestion,
    GA_VERTEX_GEMINI_MODELS,
    GeminiDecisionsBackend,
    VertexGeminiBackend,
)

decisions_backend = GeminiDecisionsBackend(
    api_key="example-header-only-api-key",
    model="models/gemini-decision-1",
    include_none_of_above=True,
    shortlist_top_k=120,
)

wide_catalog_criteria = {f"sku_{idx:03d}": f"Catalog item {idx}" for idx in range(300)}
wire_payload, slot_plans = decisions_backend.build_decisions_payload(
    state={"query": "Catalog item 042"},
    questions={
        "policy_ok": DecisionQuestion(
            type="choice",
            instructions="Does the listing satisfy marketplace policy?",
            criteria={"Yes": "Compliant", "No": "Violation"},
            order=["Yes", "No"],
        ),
        "matched_sku": DecisionQuestion(
            type="choice",
            instructions="Select the matching SKU.",
            criteria=wide_catalog_criteria,
        ),
        "condition_grade": DecisionQuestion(
            type="score",
            instructions="Rate packaging condition from 1 to 3.",
            criteria={"1": "Damaged", "2": "Acceptable", "3": "Mint"},
            levels=["1", "2", "3"],
        ),
    },
)
assert slot_plans["policy_ok"]["kind"] == "noul"
assert slot_plans["matched_sku"]["shortlist_bridged"] is True
assert len(slot_plans["matched_sku"]["explicit_keys"]) == 120
assert slot_plans["condition_grade"]["kind"] == "score"

assert GA_VERTEX_GEMINI_MODELS == (
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.5-pro",
)
stage2_verifier = VertexGeminiBackend.from_ga_preset(
    "gemini-3.8-flash",
    project="example-prod-project",
    location="us-central1",
    access_token_fn=lambda: "ephemeral-oauth-token",
)
assert "gemini-3.8-flash:generateContent" in stage2_verifier.endpoint_url()
```

## Run the capability-aware MCP `stdio` server (`dgem-catalog`)

`mcp_server.py` (`dgem-catalog`, version `1.1.0`) exposes `mdm` to MCP-compatible IDEs and agent hosts over `stdin`/`stdout` (supporting both newline-delimited JSON-RPC 2.0 and LSP `Content-Length` framing up to `16 MiB`):

```bash
python3 -m multimodal_decision_models.mcp_server
```

When embedding `mcp_server` inside a custom host, use `build_mcp_tool_schemas` and `serve_stdio` flags to expose only the tools and backend paradigms enabled in that environment:

```python
from multimodal_decision_models import build_mcp_tool_schemas, serve_stdio

schemas = build_mcp_tool_schemas(
    configured_backends=["dgem_systemone", "vertex_gemini"],
    enable_visual_compiler=True,
    enable_context_pruner=False,
)
# Calling disabled tools via tools/call is also rejected fail-closed (-32602).
```

## Deploy to Google Cloud with Terraform (`deploy/terraform/`)

The [`deploy/terraform/`](../deploy/terraform/) root module provisions a least-privilege Cloud Run `v2` orchestrator (`us-central1` default), Artifact Registry, versioned calibration GCS bucket (`public_access_prevention = "enforced"`), Secret Manager credential injection, 6 Cloud Logging funnel metrics + Monitoring dashboard, and an optional GPU-backed Vertex AI `/v1/systemone` endpoint:

```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
# Populate project_id and @sha256:<64-hex> container image digest
terraform init
terraform plan
```

### 3-Bucket capability scope

| Capability | Scope tier | Source of truth |
| :--- | :--- | :--- |
| Cloud Run `v2` orchestrator (`us-central1` default), least-privilege service accounts, versioned calibration GCS bucket, Artifact Registry, Secret Manager env binding, 6 Cloud Logging funnel counters + dashboard, optional Vertex AI Model and Endpoint | **Live in Terraform** | [`deploy/terraform/*.tf`](../deploy/terraform/) |
| `CascadeDecisionOrchestrator`, `DgemSystemOneBackend`, `GeminiDecisionsBackend` (`POST /v1beta/decisions` on `models/gemini-decision-1`), `OpenAILogprobBackend`, `VertexGeminiBackend.from_ga_preset` (`gemini-3.5-flash-lite`, `gemini-3.5-flash`, `gemini-3.8-flash`, `gemini-3.5-pro`), `BlackBoxSampledBackend`, stateless `HMAC-SHA256` conformal sets, `PrimalDualParetoBanditRouter`, `ContentAddressedDecisionCache`, `dgem-catalog` MCP server, and ADK / LangGraph / Genkit guards | **In-Process (`mdm` SDKs)** | [`src/multimodal_decision_models/`](../src/multimodal_decision_models/), [`pkg/mdm/`](../pkg/mdm/), [`sdk/ts/`](../sdk/ts/) |
| `MDM-EpisodeBench` `Tier-2` five-rung and $2^4 = 16$ factorial ablation evaluation (`1,200` episodes, `360 D_cal / 240 D_val / 600 D_test` = `3,600` locked `D_test` turns, `SHA-256: 0bcfd508bd737c84...`) | **Receipt-Grounded Replay** | [`benchmarks/locks/mdm_episodebench_v1.lock.json`](../benchmarks/locks/mdm_episodebench_v1.lock.json), [`benchmarks/results/mdm_factorial_ablation_report.md`](../benchmarks/results/mdm_factorial_ablation_report.md) |
| Multi-region Global Load Balancer failover, Cloud Armor WAF policies, and automated canary traffic splitting across Vertex `DeployedModel` revisions | **Planned / Consumer-Composed** | Compose upstream of `orchestrator_service_uri` in organization networking Terraform |
