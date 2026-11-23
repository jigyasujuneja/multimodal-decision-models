# `@mdm/client` — TypeScript SDK for Multimodal Decision Models

Zero-runtime-dependency TypeScript client, calibration library, and Genkit JS tool-safety guard for `multimodal-decision-models` (`v1.0.0`).

## Features

- **Multi-Paradigm HTTP Client (`MDMClient`)**:
  - Parallel `/v1/systemone` masked-diffusion readout and single-route Vertex AI `:rawPredict` auto-dispatch (`POST {"health": true}`).
  - OpenAI-compatible `/v1/chat/completions` single-token `[A..Z]` logprob readout (`decideChatLogprobs`) with strict alignment checks and zero uniform fallback.
  - Permanent credential stripping (`Authorization`, `X-DGEM-Key`, `X-DGEM-API-Key`, `X-Goog-Api-Key`) across cross-origin HTTP redirects, including `307`/`308` bounce-back hops.
- **Exact Cross-Language Calibration & Conformal Parity**:
  - Stateless `HMAC-SHA256` tie-breaking (`hmacUint64`, `hmacUniformFloat64`) matching Python `multimodal_decision_models` and Go `pkg/mdm` bit-for-bit.
  - Split conformal prediction (`lac`, `aps`, `raps`) via `calibrateConformalQuantile` and `conformalPredictionSet`.
  - Simultaneous family-wise multi-slot joint conformal calibration (`calibrateMultiSlotJointConformal`, `multiSlotJointConformalSets`).
  - Temperature scaling (`applyTemperature`), cosine prior normalization (`normalizeCosinePriorLogits`), and pre-gated Bayesian Product-of-Experts (`fusePriorProbabilities`).
- **Genkit JS Action-Stakes Guards**:
  - `createGenkitToolGuard` enforcing $V_{\text{irrev}}$ vs. $V_{\text{rev}}$ confidence gates, `<untrusted_data>` XML delimiter neutralization, joint conformal set checks, and fail-closed transport quarantine.
  - `createGenkitDecisionStep` wrapping multi-slot structured decisions for Genkit flows.

## Quickstart

```ts
import {
  MDMClient,
  conformalPredictionSet,
  createGenkitToolGuard,
} from "@mdm/client";

const client = new MDMClient({
  baseUrl: "https://decision.example.com",
  bearerToken: process.env.MDM_BEARER_TOKEN,
});

const decision = await client.decideSystemOne({
  state: "Refund request #4821 for unopened item within 14-day window.",
  questions: {
    route: {
      type: "choice",
      instructions: "Select the refund disposition.",
      criteria: {
        auto_approve: "Eligible unopened return within 30 days.",
        manual_review: "Requires human inspection.",
        reject: "Outside return policy.",
      },
      order: ["auto_approve", "manual_review", "reject"],
    },
  },
});

const conformal = conformalPredictionSet(
  decision.slots.route.probabilities,
  0.15,
  { method: "raps", seedKey: "dgem-conformal-v1", itemKey: "req-4821:route", randomized: true },
);
```

## Build and Test

```bash
npm --prefix sdk/ts ci
npm --prefix sdk/ts test
```
