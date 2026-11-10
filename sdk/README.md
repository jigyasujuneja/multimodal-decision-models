# Multimodal Decision Models (`mdm`) Multi-Language SDKs

`multimodal-decision-models` ships three zero-dependency client and calibration SDKs sharing identical conformal math, Bayesian Product-of-Experts (`PoE`) prior fusion, and stateless `HMAC-SHA256` tie-breaking vectors:

| Language | Package / Import Path | Primary Entrypoints |
| --- | --- | --- |
| **Python 3.10+** | `multimodal_decision_models` ([`../src/multimodal_decision_models/`](../src/multimodal_decision_models/)) | `CascadeDecisionOrchestrator`, `DgemSystemOneBackend`, `OpenAILogprobBackend`, `VertexGeminiBackend`, `BlackBoxSampledBackend`, `create_adk_before_tool_guard`, `create_langgraph_decision_node` |
| **Go 1.22+** | `github.com/jigyasujuneja/multimodal-decision-models/pkg/mdm` ([`../pkg/mdm/`](../pkg/mdm/)) | `PreGatedBayesianPoEFusion`, `NormalizeCosinePriorLogits`, `ConformalPredictionSet`, `HmacUint64`, `PruneAgentContextBlocks` |
| **TypeScript / Node.js 20+** | `@mdm/client` ([`./ts/`](./ts/)) | `MDMClient`, `conformalPredictionSet`, `calibrateMultiSlotJointConformal`, `fusePriorProbabilities`, `createGenkitToolGuard`, `createGenkitDecisionStep` |

## Cross-Language Determinism Vector

All three runtimes verify bit-exact parity on the canonical `HMAC-SHA256` vector:

- `hmac_uint64("dgem-conformal-v1", "req-42:intent", "u") == 14094744155160802019`
- `hmac_uniform_float64("dgem-conformal-v1", "req-42:intent", "u") == 0.7640776116826353`
