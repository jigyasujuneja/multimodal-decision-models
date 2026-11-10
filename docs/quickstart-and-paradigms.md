# Quickstart and four-paradigm backends

To run calibrated structured decisions without locking your application to a single model family, `multimodal-decision-models` (`mdm`) provides a unified `DecisionQuestion` and `CascadeDecisionOrchestrator` contract across Python, Go, and TypeScript.

## Understand where `mdm` fits

Engineering teams frequently combine three complementary layers:

| Layer | Responsibility | When to use |
| :--- | :--- | :--- |
| **System-1 Serving Engine (`dgem` / `/v1/systemone`)** | Co-denoises up to 8 decision slots in one parallel forward pass (`15 ms` to `180 ms`) and returns calibrated per-option probabilities. | Ultra-low-latency Stage-1 scoring on dedicated GPUs or Vertex AI endpoints. |
| **Managed Single-Pass Decision APIs (`CreateDecision`)** | Hosted single-call System-1 scoring over text and multimodal payloads. | Turnkey single-pass classification where a managed API owns model hosting. |
| **Calibration & Cascade Layer (`multimodal-decision-models`)** | Model-agnostic backend adapters, pre-gated Bayesian `PoE` prior fusion, split/joint/Mondrian conformal sets, visual and audio/video contact-sheet compilers, Action-Stakes agent guards, and Stage-1 to Stage-2 cascade orchestration. | Whenever you require finite-sample coverage guarantees, wide-catalog routing ($K > 26$), multi-model cascades, or strict parity across `/v1/systemone`, `vLLM`, Vertex Gemini, and black-box APIs. |

## Install the SDKs

 All three SDKs have zero external runtime dependencies.

### Python (`>= 3.10`)

```bash
pip install -e .
```

### Go (`>= 1.22`)

```bash
go get github.com/jigyasujuneja/multimodal-decision-models/pkg/mdm
```

### TypeScript / Node.js (`>= 20`)

```bash
npm --prefix sdk/ts ci
npm --prefix sdk/ts run build
```

## Choose an inference backend paradigm

Every Python backend implements `score_questions(state, questions, images=None)` and plugs directly into `CascadeDecisionOrchestrator`:

| Paradigm | Python class | Wire protocol | Probability extraction mechanism |
| :--- | :--- | :--- | :--- |
| **1. Parallel Masked Diffusion** | `DgemSystemOneBackend` | `POST /v1/systemone` or Vertex `:rawPredict` | Single forward pass co-denoising up to 8 `[A-Z]` slots simultaneously. Auto-dispatches `POST {"health": true}` on single-route `:rawPredict` URLs. |
| **2. Autoregressive Token Logprobs** | `OpenAILogprobBackend` | `POST /v1/chat/completions` (`vLLM`, `SGLang`, OpenAI, `Gemma 4`) | Restricted single-token `[A-Z]` (or `UNIVERSAL_256_DECISION_CODES`) `top_logprobs` extraction with `<think>` token leakage guards and zero uniform fallback. |
| **3. Frontier Multimodal Verifier** | `VertexGeminiBackend` | Vertex AI `generateContent` | Structured `responseLogprobs` enum binding with causal Evidence-First `propertyOrdering` (`evidence_read` decoded ahead of choice slots). |
| **4. Logprob-Free Sampled Consensus** | `BlackBoxSampledBackend` | Any JSON/text completion endpoint | Aggregates $M$ discrete draws or verbalized vote maps under a symmetric Dirichlet prior ($\alpha_0 = 0.5$) with exact analytic epistemic variance. |

## Run a two-stage calibrated cascade in Python

The following example calibrates a Regularized Adaptive Prediction Set (`RAPS`) threshold on held-out probabilities and runs a Stage-1 to Stage-2 cascade that escalates only when the conformal set is ambiguous or Stage-1 normalized entropy exceeds `0.16`:

```python
from multimodal_decision_models import (
    BlackBoxSampledBackend,
    CascadeDecisionOrchestrator,
    DecisionQuestion,
    DgemSystemOneBackend,
    calibrate_conformal_quantile,
)

questions = {
    "route": DecisionQuestion(
        type="choice",
        instructions="Classify the incoming support request into a fulfillment queue.",
        criteria={
            "billing_refund": "Charge dispute, invoice error, or refund request.",
            "technical_incident": "API outage, latency spike, or SDK error.",
            "ambiguous_or_oos": "Insufficient context to route safely.",
        },
        order=["billing_refund", "technical_incident", "ambiguous_or_oos"],
    )
}

calibration = calibrate_conformal_quantile(
    calib_probs=[
        {"billing_refund": 0.88, "technical_incident": 0.08, "ambiguous_or_oos": 0.04},
        {"billing_refund": 0.07, "technical_incident": 0.86, "ambiguous_or_oos": 0.07},
        {"billing_refund": 0.44, "technical_incident": 0.46, "ambiguous_or_oos": 0.10},
    ],
    calib_labels=["billing_refund", "technical_incident", "technical_incident"],
    alpha=0.10,
    method="raps",
    randomized=True,
    seed_key="dgem-conformal-v1",
)

stage1 = DgemSystemOneBackend("https://stage1.example.com")
stage2 = BlackBoxSampledBackend(
    "stage2-verifier",
    endpoint_url="https://verifier.example.com",
    model="verifier-pro",
    num_samples=5,
    prior_alpha=0.5,
)

orchestrator = CascadeDecisionOrchestrator(
    stage1_backend=stage1,
    stage2_backend=stage2,
    hesitation_threshold=0.16,
    conformal_q_hat=calibration.q_hat,
    conformal_method="raps",
)
```

## Run the identical policy in TypeScript and Go

Both `@mdm/client` and `pkg/mdm` enforce identical cross-language conformal math and stateless `HMAC-SHA256` tie-breaking:

```ts
import { MDMClient, conformalPredictionSet, hmacUint64 } from "@mdm/client";

// Matches Python hmac_uint64 and Go HmacUint64: 14094744155160802019n
const digest = hmacUint64("dgem-conformal-v1", "req-42:intent", "u");

const client = new MDMClient({
  baseUrl: "https://stage1.example.com",
  bearerToken: process.env.MDM_BEARER_TOKEN,
});

const decision = await client.decideSystemOne({
  state: "Double charge on April invoice #9012.",
  questions: {
    route: {
      type: "choice",
      instructions: "Classify the support request.",
      criteria: {
        billing_refund: "Charge dispute or refund request.",
        technical_incident: "Service outage or SDK bug.",
        ambiguous_or_oos: "Insufficient context.",
      },
      order: ["billing_refund", "technical_incident", "ambiguous_or_oos"],
    },
  },
});

const routeSlot = decision.slots["route"];
if (routeSlot) {
  const conformalSet = conformalPredictionSet(routeSlot.rawProbabilities, 0.18, {
    method: "raps",
    randomized: true,
    seedKey: "dgem-conformal-v1",
    itemKey: "req-9012:route",
  });
  if (conformalSet.shouldEscalate) {
    // Escalate to Stage-2 verifier or human queue.
  }
}
```

```go
package main

import (
	"fmt"

	"github.com/jigyasujuneja/multimodal-decision-models/pkg/mdm"
)

func main() {
	digest := mdm.HmacUint64("dgem-conformal-v1", "req-42:intent", "u")
	setResult, _ := mdm.ConformalPredictionSet(
		map[string]float64{"billing_refund": 0.88, "technical_incident": 0.08, "ambiguous_or_oos": 0.04},
		0.18,
		mdm.ConformalConfig{
			Method:     "raps",
			LamReg:     0.01,
			KReg:       1,
			Randomized: true,
			SeedKey:    "dgem-conformal-v1",
			ItemKey:    "req-9012:route",
		},
	)
	fmt.Println(digest, setResult.PredictionSet, setResult.ShouldEscalate)
}
```

## HTTP zero-trust transport guarantees

Both Python (`backends.py`) and TypeScript (`MDMClient`) enforce three transport invariants out of the box:

1. **Permanent cross-origin credential stripping**: If an HTTP `301`, `302`, `303`, `307`, or `308` redirect crosses `(scheme, host, port)` boundaries, `Authorization`, `X-DGEM-Key`, `X-DGEM-API-Key`, and `X-Goog-Api-Key` headers are stripped permanently, even if a subsequent hop redirects back to the initial origin.
2. **Bounded TCP + TLS handshake latency**: Python `_BoundedConnectHTTPSConnection` caps socket and TLS handshake time at `20.0 s` independently of long Stage-2 read timeouts.
3. **Header and URL injection rejection**: CR/LF characters in URLs or header values and non-HTTP(S) schemes (`file://`) raise `SchemaContractViolationError` / `MDMValidationError` before any socket opens.

## What's next

- Configure temperature scaling, pre-gated prior fusion, and family-wise multi-slot conformal bounds in [Calibration, conformal sets, and risk control](calibration-and-conformal.md).
- Scale beyond `26` options in [Wide-catalog routing and taxonomy tries](wide-catalog-routing.md).
