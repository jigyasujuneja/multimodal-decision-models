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

export type QuestionType = "choice" | "noul" | "score";
export type ConformalMethod = "lac" | "aps" | "raps";
export type PriorGateMode =
  | "hesitant"
  | "confident"
  | "proportional"
  | "always"
  | "never"
  | "off";

export class MDMError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MDMError";
  }
}

export class MDMValidationError extends MDMError {
  constructor(message: string) {
    super(message);
    this.name = "MDMValidationError";
  }
}

export class MDMTransportError extends MDMError {
  readonly retryable: boolean;

  constructor(message: string, retryable = true) {
    super(message);
    this.name = "MDMTransportError";
    this.retryable = retryable;
  }
}

export class MDMHTTPError extends MDMError {
  readonly statusCode: number;
  readonly responseBody: string;
  readonly retryable: boolean;

  constructor(statusCode: number, responseBody: string) {
    super(`HTTP ${statusCode}: ${responseBody.slice(0, 256)}`);
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

export interface DecisionQuestion {
  readonly type: QuestionType;
  readonly instructions: string;
  readonly criteria?: Record<string, string>;
  readonly levels?: readonly string[];
  readonly order?: readonly string[];
  readonly priorLogits?: Record<string, number>;
}

export interface SlotReadout {
  readonly slotId: string;
  readonly predicted: string;
  readonly probabilities: Record<string, number>;
  readonly rawProbabilities: Record<string, number>;
  readonly hRaw: number;
  readonly confidence: number;
}

export interface DecisionResponse {
  readonly slots: Record<string, SlotReadout>;
  readonly latencyMs: number;
  readonly backendName: string;
}

export interface SystemOneDecideRequest {
  readonly state: unknown;
  readonly questions: Record<string, DecisionQuestion>;
  readonly images?: readonly string[];
  readonly priorLogits?: Record<string, Record<string, number>>;
  readonly temperature?: number;
}

export interface ChatLogprobDecideRequest {
  readonly model: string;
  readonly state: unknown;
  readonly questions: Record<string, DecisionQuestion>;
  readonly topLogprobs?: number;
  readonly temperature?: number;
}

export interface ConformalConfig {
  readonly method?: ConformalMethod;
  readonly lamReg?: number;
  readonly kReg?: number;
  readonly randomized?: boolean;
  readonly seedKey?: string;
  readonly itemKey?: string;
  readonly abstainLabels?: readonly string[];
}

export interface ConformalCalibrationResult {
  readonly qHat: number;
  readonly alpha: number;
  readonly nCalib: number;
  readonly method: ConformalMethod;
  readonly empiricalCalibCoverage: number;
  readonly lamReg: number;
  readonly kReg: number;
}

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

export interface BayesianPoEResult {
  readonly probabilities: Record<string, number>;
  readonly hRaw: number;
  readonly escalated: boolean;
}

export interface PolynomialAlphaConfig {
  readonly alphaTotal?: number;
  readonly exponentP?: number;
  readonly horizonMaxSteps?: number;
  readonly markovSlackFactor?: number;
}

export interface NevControllerPolicy {
  readonly taskValue?: number;
  readonly interventionCost?: number;
  readonly fixProbabilityIfFailing?: number;
  readonly regressionProbabilityIfPassing?: number;
  readonly plattSlope?: number;
  readonly plattIntercept?: number;
}

export interface NevDecisionResult {
  readonly shouldIntervene: boolean;
  readonly rawFailProbability: number;
  readonly calibratedFailProbability: number;
  readonly netExpectedValue: number;
  readonly minFailProbabilityThreshold: number;
}
