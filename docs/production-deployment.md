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
- Never caches `RETRY_PENDING` or `REJECTED_CONTRACT_VIOLATION` states, ensuring transient upstream outages recover automatically on retry.

## Run the capability-aware MCP `stdio` server (`dgem-catalog`)

`mcp_server.py` (`dgem-catalog`, version `1.0.0`) exposes `mdm` to MCP-compatible IDEs and agent hosts over `stdin`/`stdout` (supporting both newline-delimited JSON-RPC 2.0 and LSP `Content-Length` framing up to `16 MiB`):

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
| Cloud Run `v2` orchestrator (`us-central1` default), least-privilege service accounts, versioned calibration GCS bucket, Artifact Registry, Secret Manager env binding, 6 Cloud Logging funnel counters + dashboard, optional Vertex AI Model & Endpoint | **Live in Terraform** | [`deploy/terraform/*.tf`](../deploy/terraform/) |
| `CascadeDecisionOrchestrator`, 4-paradigm backends, stateless `HMAC-SHA256` conformal sets, `ContentAddressedDecisionCache`, `dgem-catalog` MCP server, and ADK / LangGraph / Genkit guards | **In-Process (`mdm` SDKs)** | [`src/multimodal_decision_models/`](../src/multimodal_decision_models/), [`pkg/mdm/`](../pkg/mdm/), [`sdk/ts/`](../sdk/ts/) |
| Multi-region Global Load Balancer failover, Cloud Armor WAF policies, and automated canary traffic splitting across Vertex `DeployedModel` revisions | **Planned / Consumer-Composed** | Compose upstream of `orchestrator_service_uri` in organization networking Terraform |
