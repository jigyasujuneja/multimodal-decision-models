/**
 * Copyright 2026 Jigyasu Juneja
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** Identifies the readout schema type for a decision question slot. */
export type QuestionType = "choice" | "noul" | "score";

/** Identifies the split conformal nonconformity scoring function. */
export type ConformalMethod = "lac" | "aps" | "raps";

/** Controls when prior log-odds are fused with Stage-1 probabilities. */
export type PriorGateMode =
  | "hesitant"
  | "confident"
  | "proportional"
  | "always"
  | "never"
  | "off";

/** Base error class thrown by all `@mdm/client` operations. */
export class MDMError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "MDMError";
  }
}

/** Indicates invalid caller inputs, schema contract violations, or blocked URLs. */
export class MDMValidationError extends MDMError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "MDMValidationError";
  }
}

/** Indicates an HTTP transport, redirect, stream, or JSON parsing failure. */
export class MDMTransportError extends MDMError {
  readonly retryable: boolean;

  constructor(message: string, retryable = true, options?: ErrorOptions) {
    super(message, options);
    this.name = "MDMTransportError";
    this.retryable = retryable;
  }
}

/** Indicates a non-2xx HTTP response returned by an upstream decision service. */
export class MDMHTTPError extends MDMError {
  readonly statusCode: number;
  readonly responseBody: string;
  readonly retryable: boolean;

  constructor(statusCode: number, responseBody: string, options?: ErrorOptions) {
    super(`HTTP ${statusCode}: ${responseBody.slice(0, 256)}`, options);
    this.name = "MDMHTTPError";
    this.statusCode = statusCode;
    this.responseBody = responseBody;
    this.retryable =
      statusCode === 408 ||
      statusCode === 429 ||
      statusCode === 500 ||
      statusCode === 502 ||
      statusCode === 503 ||
      statusCode === 504;
  }
}

/** Defines a single structured choice, noul, or ordinal score decision slot. */
export interface DecisionQuestion {
  readonly type: QuestionType;
  readonly instructions: string;
  readonly criteria?: Readonly<Record<string, string>>;
  readonly optionDescriptions?: Readonly<Record<string, string>>;
  readonly levels?: readonly string[];
  readonly order?: readonly string[];
  readonly priorLogits?: Readonly<Record<string, number>>;
  readonly domainCompetence?: DomainCompetenceProfile;
}

/** Holds the calibrated probability distribution and hesitation summary for one slot. */
export interface SlotReadout {
  readonly slotId: string;
  readonly predicted: string;
  readonly probabilities: Record<string, number>;
  readonly rawProbabilities: Record<string, number>;
  readonly hRaw: number;
  readonly confidence: number;
}

/** Holds the multi-slot probability readouts and latency metadata for one call. */
export interface DecisionResponse {
  readonly slots: Record<string, SlotReadout>;
  readonly latencyMs: number;
  readonly backendName: string;
}

/** Parameters for a parallel `/v1/systemone` or Vertex `:rawPredict` decision call. */
export interface SystemOneDecideRequest {
  readonly state: unknown;
  readonly questions: Readonly<Record<string, DecisionQuestion>>;
  readonly images?: readonly string[];
  readonly priorLogits?: Readonly<Record<string, Readonly<Record<string, number>>>>;
  readonly temperature?: number;
  readonly signal?: AbortSignal;
}

/** Parameters for an OpenAI-compatible `/v1/chat/completions` logprob decision call. */
export interface ChatLogprobDecideRequest {
  readonly model: string;
  readonly state: unknown;
  readonly questions: Readonly<Record<string, DecisionQuestion>>;
  readonly topLogprobs?: number;
  readonly temperature?: number;
  readonly signal?: AbortSignal;
}

/** Configures split conformal calibration and prediction-set construction. */
export interface ConformalConfig {
  readonly method?: ConformalMethod;
  readonly lamReg?: number;
  readonly kReg?: number;
  readonly randomized?: boolean;
  readonly seedKey?: string;
  readonly itemKey?: string;
  readonly abstainLabels?: readonly string[];
}

/** Summarizes a calibrated split conformal quantile and empirical coverage. */
export interface ConformalCalibrationResult {
  readonly qHat: number;
  readonly alpha: number;
  readonly nCalib: number;
  readonly method: ConformalMethod;
  readonly empiricalCalibCoverage: number;
  readonly lamReg: number;
  readonly kReg: number;
}

/** Summarizes a conformal prediction set and abstention escalation decision. */
export interface ConformalSetResult {
  readonly predictionSet: readonly string[];
  readonly scores: Record<string, number>;
  readonly qHat: number;
  readonly topOption: string;
  readonly topProbability: number;
  readonly setSize: number;
  readonly singleton: boolean;
  readonly containsAbstain: boolean;
  readonly shouldEscalate: boolean;
}

/** Summarizes family-wise joint conformal thresholds across multiple slots. */
export interface MultiSlotJointConformalResult {
  readonly jointMaxQHat: number;
  readonly bonferroniSlotQHats: Record<string, number>;
  readonly slotNames: readonly string[];
  readonly alpha: number;
  readonly nCalib: number;
  readonly empiricalJointCoverage: number;
  readonly method: ConformalMethod;
  readonly lamReg: number;
  readonly kReg: number;
}

/** Holds the posterior distribution and hesitation gate result of Bayesian PoE fusion. */
export interface BayesianPoEResult {
  readonly probabilities: Record<string, number>;
  readonly hRaw: number;
  readonly escalated: boolean;
}

/** Configures polynomial per-step alpha spending over finite or unbounded trajectories. */
export interface PolynomialAlphaConfig {
  readonly alphaTotal?: number;
  readonly exponentP?: number;
  readonly horizonMaxSteps?: number;
  readonly markovSlackFactor?: number;
}

/** Configures the Net Expected Value Controller (NEVC) intervention gate. */
export interface NevControllerPolicy {
  readonly taskValue?: number;
  readonly interventionCost?: number;
  readonly fixProbabilityIfFailing?: number;
  readonly regressionProbabilityIfPassing?: number;
  readonly plattSlope?: number;
  readonly plattIntercept?: number;
}

/** Summarizes the NEVC decision, calibrated failure probability, and net value. */
export interface NevDecisionResult {
  readonly shouldIntervene: boolean;
  readonly rawFailProbability: number;
  readonly calibratedFailProbability: number;
  readonly netExpectedValue: number;
  readonly minFailProbabilityThreshold: number;
}

/** Stores held-out Stage-1 and Stage-2 accuracy and Brier competence metrics. */
export interface DomainCompetenceProfile {
  readonly stage1Accuracy: number;
  readonly stage2Accuracy: number;
  readonly numOptions?: number;
  readonly pairedEscalationDelta?: number;
  readonly sampleCount?: number;
  readonly stage1Brier?: number | null;
  readonly stage2Brier?: number | null;
}

/** Configures negative-transfer circuit-breaker thresholds for Stage-2 escalation. */
export interface NegativeTransferOptions {
  readonly margin?: number;
  readonly minDeltaHurt?: number;
  readonly brierDegradationRatio?: number;
}
