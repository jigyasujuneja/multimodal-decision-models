# Declarative Terraform Deployment for `multimodal-decision-models`

This root module provisions a least-privilege Google Cloud reference stack for running `CascadeDecisionOrchestrator` on Cloud Run (`us-central1` default), backed by a versioned calibration artifact bucket, Artifact Registry, Secret Manager secret references, Cloud Logging / Monitoring funnel telemetry, and an optional GPU-backed Vertex AI `/v1/systemone` endpoint.

## Security & Operational Guardrails

- **Digest-Pinned Images Only**: `orchestrator_container_image` and `vertex_serving_container_image` enforce `@sha256:<64-hex>` immutable digests via variable validation.
- **Zero Public Unauthenticated Invokers**: `invoker_members` rejects `allUsers` and `allAuthenticatedUsers` at `terraform plan` time, and `cloudrun_ingress` defaults to `INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER`.
- **Zero Plaintext Secrets in State**: Upstream API credentials are mounted directly into Cloud Run from Secret Manager (`upstream_api_key_secret_id`) rather than Terraform variables.
- **Enforced Bucket Isolation**: `google_storage_bucket.calibration_artifacts` enforces `uniform_bucket_level_access = true`, `public_access_prevention = "enforced"`, object versioning, and optional CMEK (`kms_key_name`).

## 3-Bucket Capability Scope

To keep infrastructure guarantees transparent, capabilities are partitioned into three explicit tiers:

| Capability | Tier | Where Enforced |
| --- | --- | --- |
| Cloud Run `v2` orchestrator (`us-central1` default), least-privilege IAM SAs, versioned calibration GCS bucket, Artifact Registry, Secret Manager injection, 6 Cloud Logging funnel metrics + Monitoring dashboard, optional Vertex AI Model & Endpoint | **Live in Terraform** | [`deploy/terraform/*.tf`](.) |
| `CascadeDecisionOrchestrator` (`STAGE2A_FAST_EXIT`, `STAGE1_RESOLVED`, `STAGE2_VERIFIED`, `ABSTAIN_OOS`, `RETRY_PENDING`, `REJECTED_CONTRACT_VIOLATION`), SHA-256 `ContentAddressedDecisionCache`, `dgem-catalog` MCP `stdio` server, Google ADK / LangGraph / Genkit guards | **In-Process (`mdm` SDK & Runtime)** | [`src/multimodal_decision_models/`](../../src/multimodal_decision_models/), [`pkg/mdm/`](../../pkg/mdm/), [`sdk/ts/`](../../sdk/ts/) |
| Multi-region active-active Global External Application Load Balancer failover, Cloud Armor WAF policies, and automated Vertex `DeployedModel` rollout traffic splitting | **Planned / Consumer-Composed** | Compose upstream of `orchestrator_service_uri` in organization-level networking modules |

## Usage

```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your GCP project_id and @sha256 container digest
terraform init
terraform plan
```

## Offline Validation Test

CI validates HCL structural integrity, variable/output completeness, `@sha256:` digest enforcement, `allUsers` rejection, and public hygiene without requiring cloud credentials:

```bash
python3 deploy/terraform/test_terraform_config.py
```
