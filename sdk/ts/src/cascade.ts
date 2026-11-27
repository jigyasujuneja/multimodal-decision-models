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

import {
  MAX_TEMPERATURE,
  MIN_TEMPERATURE,
  computeDawidSkenePoeWeights,
  fusePriorProbabilities,
  isNegativeTransfer,
  isUniformDistribution,
  normalizedEntropy,
} from "./calibration.js";
import {
  getQuestionOptionKeys,
  isPlainRecord,
  nullProtoRecord,
  throwIfSignalAborted,
} from "./internal.js";
import {
  type DecisionQuestion,
  type DecisionResponse,
  type DomainCompetenceProfile,
  MDMValidationError,
  type NegativeTransferOptions,
  type PriorGateMode,
  type SlotReadout,
  type SystemOneDecideRequest,
} from "./types.js";

/** Default normalized Shannon entropy threshold triggering Stage-2 escalation. */
export const DEFAULT_CASCADE_HESITATION_THRESHOLD = 0.16;

/** Configures gating, speculative execution, pruning, and fusion for `CascadeOrchestrator`. */
export interface CascadeConfig {
  readonly hesitationThreshold?: number;
  readonly temperature?: number;
  readonly priorWeight?: number;
  readonly priorGateMode?: PriorGateMode;
  readonly speculativeStage2?: boolean;
  readonly stage2TopK?: number;
  readonly domainCompetence?: DomainCompetenceProfile;
  readonly negativeTransferOptions?: NegativeTransferOptions;
  readonly useDawidSkenePoeWeights?: boolean;
  readonly voiErrorCost?: number;
  readonly voiStage2ExpectedAccuracy?: number;
  readonly voiStage2RpcCostUsd?: number;
  readonly voiLatencyCostUsd?: number;
}

/** Extends `SlotReadout` with per-slot cascade escalation and pass-count metadata. */
export interface CascadeSlotReadout extends SlotReadout {
  readonly escalated: boolean;
  readonly passesUsed: number;
}

/** Summarizes the multi-slot outcome, escalation trace, and speculative telemetry of a cascade call. */
export interface CascadeDecisionResponse {
  readonly slots: Record<string, CascadeSlotReadout>;
  readonly escalated: boolean;
  readonly escalationReasons: readonly string[];
  readonly speculativeLaunched: boolean;
  readonly speculativeSkippedByCircuitBreaker: boolean;
  readonly latencyMs: number;
  readonly backendName: string;
}

/** Backend interface accepted for Stage 1 and Stage 2 of `CascadeOrchestrator`. */
export type CascadeStageBackend =
  | {
      readonly name?: string;
      decide(request: SystemOneDecideRequest): Promise<DecisionResponse>;
    }
  | {
      readonly name?: string;
      decideSystemOne(request: SystemOneDecideRequest): Promise<DecisionResponse>;
    }
  | ((request: SystemOneDecideRequest) => Promise<DecisionResponse>);

function raceWithAbortSignal<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) {
    return promise;
  }
  throwIfSignalAborted(signal);
  return new Promise<T>((resolve, reject) => {
    const onAbort = (): void => {
      reject(signal.reason ?? new DOMException("The operation was aborted.", "AbortError"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (val) => {
        signal.removeEventListener("abort", onAbort);
        resolve(val);
      },
      (err: unknown) => {
        signal.removeEventListener("abort", onAbort);
        if (signal.aborted) {
          reject(signal.reason ?? err);
          return;
        }
        reject(err);
      },
    );
  });
}

async function invokeCascadeBackend(
  backend: CascadeStageBackend,
  request: SystemOneDecideRequest,
): Promise<DecisionResponse> {
  throwIfSignalAborted(request.signal);
  let callPromise: Promise<DecisionResponse>;
  if (typeof backend === "function") {
    callPromise = backend(request);
  } else if ("decide" in backend && typeof backend.decide === "function") {
    callPromise = backend.decide(request);
  } else if ("decideSystemOne" in backend && typeof backend.decideSystemOne === "function") {
    callPromise = backend.decideSystemOne(request);
  } else {
    throw new MDMValidationError(
      "cascade backend must be a function or implement decide() / decideSystemOne()",
    );
  }
  const response = await raceWithAbortSignal(callPromise, request.signal);
  throwIfSignalAborted(request.signal);
  return response;
}

/** Prunes a choice `DecisionQuestion` to its top-K Stage-1 options while retaining aligned descriptions and priors. */
export function pruneChoiceTopK(
  question: DecisionQuestion,
  stage1Probs: Readonly<Record<string, number>>,
  topK: number,
): DecisionQuestion {
  if (!isPlainRecord(question) || question.type !== "choice") {
    throw new MDMValidationError("pruneChoiceTopK requires a choice DecisionQuestion");
  }
  const optionKeys = getQuestionOptionKeys(question);
  if (optionKeys.length < 2) {
    throw new MDMValidationError("choice DecisionQuestion requires >= 2 options");
  }
  if (typeof topK !== "number" || !Number.isInteger(topK) || topK < 2) {
    throw new MDMValidationError("topK must be an integer >= 2");
  }
  if (!isPlainRecord(stage1Probs)) {
    throw new MDMValidationError("stage1Probs must be a non-empty probability object");
  }

  const indexByKey = new Map<string, number>();
  const validatedScores = new Map<string, number>();
  let totalMass = 0.0;
  for (let idx = 0; idx < optionKeys.length; idx += 1) {
    const key = optionKeys[idx] ?? "";
    if (key.trim() === "") {
      throw new MDMValidationError("choice option keys must be non-empty strings");
    }
    if (indexByKey.has(key)) {
      throw new MDMValidationError(`duplicate choice option key ${JSON.stringify(key)}`);
    }
    indexByKey.set(key, idx);
    const rawVal = Object.hasOwn(stage1Probs, key) ? stage1Probs[key] : 0.0;
    if (typeof rawVal !== "number" || !Number.isFinite(rawVal) || rawVal < 0.0) {
      throw new MDMValidationError(
        `stage1Probs[${JSON.stringify(key)}] must be a finite number >= 0`,
      );
    }
    validatedScores.set(key, rawVal);
    totalMass += rawVal;
  }
  if (!Number.isFinite(totalMass) || totalMass <= 0.0) {
    throw new MDMValidationError("stage1Probs must have positive total mass over question options");
  }

  const rankedKeys = [...optionKeys].sort((left, right) => {
    const leftScore = validatedScores.get(left) ?? 0.0;
    const rightScore = validatedScores.get(right) ?? 0.0;
    if (leftScore !== rightScore) {
      return rightScore - leftScore;
    }
    return (indexByKey.get(left) ?? 0) - (indexByKey.get(right) ?? 0);
  });
  const keptKeys = rankedKeys.slice(0, Math.min(topK, rankedKeys.length));

  let keptCriteria: Record<string, string> | undefined;
  if (question.criteria) {
    const map = nullProtoRecord<string>();
    for (const key of keptKeys) {
      if (Object.hasOwn(question.criteria, key)) {
        const criterion = question.criteria[key];
        if (criterion !== undefined) {
          map[key] = criterion;
        }
      }
    }
    keptCriteria = { ...map };
  }

  let keptDescriptions: Record<string, string> | undefined;
  if (question.optionDescriptions) {
    const map = nullProtoRecord<string>();
    for (const key of keptKeys) {
      if (Object.hasOwn(question.optionDescriptions, key)) {
        const desc = question.optionDescriptions[key];
        if (desc !== undefined) {
          map[key] = desc;
        }
      }
    }
    keptDescriptions = { ...map };
  }

  let keptPriors: Record<string, number> | undefined;
  if (question.priorLogits) {
    const map = nullProtoRecord<number>();
    for (const key of keptKeys) {
      if (Object.hasOwn(question.priorLogits, key)) {
        const prior = question.priorLogits[key];
        if (prior !== undefined) {
          map[key] = prior;
        }
      }
    }
    keptPriors = { ...map };
  }

  return {
    type: "choice",
    instructions: question.instructions,
    ...(keptCriteria !== undefined ? { criteria: keptCriteria } : {}),
    ...(keptDescriptions !== undefined ? { optionDescriptions: keptDescriptions } : {}),
    order: keptKeys,
    ...(keptPriors !== undefined ? { priorLogits: keptPriors } : {}),
    ...(question.domainCompetence !== undefined
      ? { domainCompetence: { ...question.domainCompetence } }
      : {}),
  };
}

function validateCascadeConfig(config: CascadeConfig): Required<
  Pick<
    CascadeConfig,
    | "hesitationThreshold"
    | "temperature"
    | "priorWeight"
    | "priorGateMode"
    | "speculativeStage2"
    | "stage2TopK"
    | "useDawidSkenePoeWeights"
    | "voiStage2ExpectedAccuracy"
    | "voiStage2RpcCostUsd"
    | "voiLatencyCostUsd"
  >
> & {
  readonly domainCompetence?: DomainCompetenceProfile;
  readonly negativeTransferOptions?: NegativeTransferOptions;
  readonly voiErrorCost?: number;
} {
  if (!config || typeof config !== "object" || Array.isArray(config)) {
    throw new MDMValidationError("config must be an object");
  }
  const hesitationThreshold =
    config.hesitationThreshold ?? DEFAULT_CASCADE_HESITATION_THRESHOLD;
  if (
    typeof hesitationThreshold !== "number" ||
    !Number.isFinite(hesitationThreshold) ||
    hesitationThreshold < 0.0
  ) {
    throw new MDMValidationError("hesitationThreshold must be a finite number >= 0");
  }
  const temperature = config.temperature ?? 1.0;
  if (
    typeof temperature !== "number" ||
    !Number.isFinite(temperature) ||
    temperature < MIN_TEMPERATURE ||
    temperature > MAX_TEMPERATURE
  ) {
    throw new MDMValidationError(
      `temperature must lie in [${MIN_TEMPERATURE}, ${MAX_TEMPERATURE}]`,
    );
  }
  const priorWeight = config.priorWeight ?? 0.0;
  if (typeof priorWeight !== "number" || !Number.isFinite(priorWeight) || priorWeight < 0.0) {
    throw new MDMValidationError("priorWeight must be a finite number >= 0");
  }
  const priorGateMode = config.priorGateMode ?? "hesitant";
  const speculativeStage2 = config.speculativeStage2 ?? false;
  if (typeof speculativeStage2 !== "boolean") {
    throw new MDMValidationError("speculativeStage2 must be a boolean");
  }
  const stage2TopK = config.stage2TopK ?? 0;
  if (
    typeof stage2TopK !== "number" ||
    !Number.isInteger(stage2TopK) ||
    stage2TopK < 0 ||
    stage2TopK === 1
  ) {
    throw new MDMValidationError("stage2TopK must be 0 or an integer >= 2");
  }
  const useDawidSkenePoeWeights = config.useDawidSkenePoeWeights ?? false;
  if (typeof useDawidSkenePoeWeights !== "boolean") {
    throw new MDMValidationError("useDawidSkenePoeWeights must be a boolean");
  }
  if (config.domainCompetence !== undefined) {
    isNegativeTransfer(config.domainCompetence, config.negativeTransferOptions ?? {});
  } else if (config.negativeTransferOptions !== undefined) {
    isNegativeTransfer(
      { stage1Accuracy: 0.8, stage2Accuracy: 0.8 },
      config.negativeTransferOptions,
    );
  }
  const voiStage2ExpectedAccuracy = config.voiStage2ExpectedAccuracy ?? 0.95;
  if (
    typeof voiStage2ExpectedAccuracy !== "number" ||
    !Number.isFinite(voiStage2ExpectedAccuracy) ||
    voiStage2ExpectedAccuracy < 0.0 ||
    voiStage2ExpectedAccuracy > 1.0
  ) {
    throw new MDMValidationError("voiStage2ExpectedAccuracy must be a finite number in [0.0, 1.0]");
  }
  if (
    config.voiErrorCost !== undefined &&
    (typeof config.voiErrorCost !== "number" ||
      !Number.isFinite(config.voiErrorCost) ||
      config.voiErrorCost < 0.0)
  ) {
    throw new MDMValidationError("voiErrorCost must be a finite number >= 0");
  }
  const voiStage2RpcCostUsd = config.voiStage2RpcCostUsd ?? 0.005;
  if (
    typeof voiStage2RpcCostUsd !== "number" ||
    !Number.isFinite(voiStage2RpcCostUsd) ||
    voiStage2RpcCostUsd < 0.0
  ) {
    throw new MDMValidationError("voiStage2RpcCostUsd must be a finite number >= 0");
  }
  const voiLatencyCostUsd = config.voiLatencyCostUsd ?? 0.0;
  if (
    typeof voiLatencyCostUsd !== "number" ||
    !Number.isFinite(voiLatencyCostUsd) ||
    voiLatencyCostUsd < 0.0
  ) {
    throw new MDMValidationError("voiLatencyCostUsd must be a finite number >= 0");
  }

  return {
    hesitationThreshold,
    temperature,
    priorWeight,
    priorGateMode,
    speculativeStage2,
    stage2TopK,
    domainCompetence: config.domainCompetence,
    negativeTransferOptions: config.negativeTransferOptions,
    useDawidSkenePoeWeights,
    voiErrorCost: config.voiErrorCost,
    voiStage2ExpectedAccuracy,
    voiStage2RpcCostUsd,
    voiLatencyCostUsd,
  };
}

function validateAndNormalizeSlotDist(
  rawDist: unknown,
  optionKeys: readonly string[],
  slotId: string,
): Record<string, number> {
  if (!isPlainRecord(rawDist)) {
    throw new MDMValidationError(`missing probabilities for slot ${JSON.stringify(slotId)}`);
  }
  const rawValues = new Map<string, number>();
  let total = 0.0;
  for (const key of optionKeys) {
    const val = Object.hasOwn(rawDist, key) ? rawDist[key] : undefined;
    if (typeof val !== "number" || !Number.isFinite(val) || val < 0.0) {
      throw new MDMValidationError(
        `slot ${JSON.stringify(slotId)} missing valid probability for ${JSON.stringify(key)}`,
      );
    }
    rawValues.set(key, val);
    total += val;
  }
  if (!Number.isFinite(total) || total <= 0.0) {
    throw new MDMValidationError(`slot ${JSON.stringify(slotId)} probabilities sum to zero`);
  }
  const normalized = nullProtoRecord<number>();
  for (const key of optionKeys) {
    const val = rawValues.get(key) ?? 0.0;
    normalized[key] = val / total;
  }
  return { ...normalized };
}

function argmaxOption(
  probs: Readonly<Record<string, number>>,
  optionKeys: readonly string[],
): string {
  let bestKey = optionKeys[0] ?? "";
  let bestProb = -1.0;
  for (const key of optionKeys) {
    const prob = Object.hasOwn(probs, key) ? (probs[key] ?? 0.0) : 0.0;
    if (prob > bestProb) {
      bestProb = prob;
      bestKey = key;
    }
  }
  return bestKey;
}

function filterStage2PriorLogits(
  priorLogits: Readonly<Record<string, Readonly<Record<string, number>>>> | undefined,
  targetQuestions: Readonly<Record<string, DecisionQuestion>>,
): Record<string, Record<string, number>> | undefined {
  if (!priorLogits) {
    return undefined;
  }
  const filtered = nullProtoRecord<Record<string, number>>();
  let hasSlot = false;
  for (const [slotId, question] of Object.entries(targetQuestions)) {
    if (!Object.hasOwn(priorLogits, slotId)) {
      continue;
    }
    const slotPriors = priorLogits[slotId];
    if (!slotPriors) {
      continue;
    }
    const keptKeys = new Set(getQuestionOptionKeys(question));
    const filteredSlot = nullProtoRecord<number>();
    let hasKey = false;
    for (const [optKey, value] of Object.entries(slotPriors)) {
      if (keptKeys.has(optKey)) {
        filteredSlot[optKey] = value;
        hasKey = true;
      }
    }
    if (hasKey) {
      filtered[slotId] = { ...filteredSlot };
      hasSlot = true;
    }
  }
  return hasSlot ? { ...filtered } : undefined;
}

/** Orchestrates hesitation-gated Stage-1 to Stage-2 cascades with speculative execution and negative-transfer guards. */
export class CascadeOrchestrator {
  private readonly stage1Backend: CascadeStageBackend;
  private readonly stage2Backend?: CascadeStageBackend;
  private readonly config: ReturnType<typeof validateCascadeConfig>;

  constructor(
    stage1Backend: CascadeStageBackend,
    stage2Backend?: CascadeStageBackend,
    config: CascadeConfig = {},
  ) {
    if (!stage1Backend) {
      throw new MDMValidationError("stage1Backend is required");
    }
    const validated = validateCascadeConfig(config);
    this.stage1Backend = stage1Backend;
    this.stage2Backend = stage2Backend;
    this.config = validated;
  }

  private resolveSlotCompetence(question: DecisionQuestion): DomainCompetenceProfile | undefined {
    return question.domainCompetence ?? this.config.domainCompetence;
  }

  private isSlotNegativeTransfer(profile: DomainCompetenceProfile | undefined): boolean {
    if (profile === undefined) {
      return false;
    }
    return isNegativeTransfer(profile, this.config.negativeTransferOptions ?? {});
  }

  /** Evaluates `request` on Stage 1 and selectively escalates uncertain or uniform slots to Stage 2. */
  async decide(request: SystemOneDecideRequest): Promise<CascadeDecisionResponse> {
    const startedAt = performance.now();
    throwIfSignalAborted(request.signal);

    const effTemp = request.temperature ?? this.config.temperature;
    if (
      typeof effTemp !== "number" ||
      !Number.isFinite(effTemp) ||
      effTemp < MIN_TEMPERATURE ||
      effTemp > MAX_TEMPERATURE
    ) {
      throw new MDMValidationError(
        `temperature must lie in [${MIN_TEMPERATURE}, ${MAX_TEMPERATURE}]`,
      );
    }
    const questionEntries = Object.entries(request.questions ?? {});
    if (questionEntries.length === 0) {
      throw new MDMValidationError("questions must contain at least one slot");
    }
    const questionMap = new Map<string, DecisionQuestion>();
    for (const [slotId, question] of questionEntries) {
      const optionKeys = getQuestionOptionKeys(question);
      if (optionKeys.length < 2) {
        throw new MDMValidationError(`slot ${JSON.stringify(slotId)} requires >= 2 options`);
      }
      if (question.domainCompetence !== undefined) {
        isNegativeTransfer(question.domainCompetence, this.config.negativeTransferOptions ?? {});
      }
      if (
        this.config.useDawidSkenePoeWeights &&
        this.resolveSlotCompetence(question) === undefined
      ) {
        throw new MDMValidationError(
          `useDawidSkenePoeWeights=true requires domainCompetence on config or slot ${JSON.stringify(slotId)}`,
        );
      }
      questionMap.set(slotId, question);
    }

    const stage2AbortController = new AbortController();
    const forwardCallerAbort = (): void => {
      stage2AbortController.abort(request.signal?.reason);
    };
    request.signal?.addEventListener("abort", forwardCallerAbort, { once: true });

    let speculativePromise: Promise<DecisionResponse> | undefined;
    const speculativeCovered = new Set<string>();
    let speculativeLaunched = false;
    let speculativeSkippedByCircuitBreaker = false;

    try {
      if (this.config.speculativeStage2 && this.stage2Backend !== undefined) {
        let allChoiceNegativeTransfer = true;
        const specQuestions = nullProtoRecord<DecisionQuestion>();
        for (const [slotId, question] of questionEntries) {
          const isChoice = question.type === "choice";
          const isBlocked = this.isSlotNegativeTransfer(this.resolveSlotCompetence(question));
          if (!(isChoice && isBlocked)) {
            allChoiceNegativeTransfer = false;
            specQuestions[slotId] = question;
            speculativeCovered.add(slotId);
          }
        }
        if (allChoiceNegativeTransfer) {
          speculativeSkippedByCircuitBreaker = true;
        } else if (speculativeCovered.size > 0) {
          speculativeLaunched = true;
          const specPriorLogits = filterStage2PriorLogits(request.priorLogits, specQuestions);
          speculativePromise = invokeCascadeBackend(this.stage2Backend, {
            state: request.state,
            questions: { ...specQuestions },
            images: request.images,
            ...(specPriorLogits !== undefined ? { priorLogits: specPriorLogits } : {}),
            temperature: effTemp,
            signal: stage2AbortController.signal,
          });
          // Suppress unhandled rejection when fast-exit cancels speculative Stage 2.
          void speculativePromise.catch(() => undefined);
        }
      }

      const stage1Response = await invokeCascadeBackend(this.stage1Backend, {
        ...request,
        temperature: effTemp,
      });

      const slotsOut = new Map<string, CascadeSlotReadout>();
      const candidateEscalated: string[] = [];
      const escalationReasons: string[] = [];
      const stage1UniformBySlot = new Map<string, boolean>();

      for (const [slotId, question] of questionEntries) {
        const optionKeys = getQuestionOptionKeys(question);
        const s1Entry = Object.hasOwn(stage1Response.slots, slotId)
          ? stage1Response.slots[slotId]
          : undefined;
        if (!s1Entry) {
          throw new MDMValidationError(`Stage-1 response missing slot ${JSON.stringify(slotId)}`);
        }
        const rawNorm = validateAndNormalizeSlotDist(
          s1Entry.rawProbabilities ?? s1Entry.probabilities,
          optionKeys,
          slotId,
        );
        const s1Uniform = isUniformDistribution(rawNorm);
        stage1UniformBySlot.set(slotId, s1Uniform);

        const slotPrior =
          (request.priorLogits && Object.hasOwn(request.priorLogits, slotId)
            ? request.priorLogits[slotId]
            : undefined) ?? question.priorLogits;

        let fusedProbs: Record<string, number>;
        let hRaw: number;
        let hesEsc: boolean;
        if (s1Uniform && (!slotPrior || this.config.priorWeight <= 0.0)) {
          fusedProbs = { ...rawNorm };
          hRaw = 1.0;
          hesEsc = 1.0 >= this.config.hesitationThreshold;
        } else {
          const poe = fusePriorProbabilities(rawNorm, {
            priorLogits: slotPrior,
            priorWeight: slotPrior ? this.config.priorWeight : 0.0,
            temperature: effTemp,
            hesitationThreshold: this.config.hesitationThreshold,
            gateMode: this.config.priorGateMode,
          });
          fusedProbs = poe.probabilities;
          hRaw = poe.hRaw;
          hesEsc = poe.escalated;
        }

        const predicted = argmaxOption(fusedProbs, optionKeys);
        let voiEsc = false;
        if (this.config.voiErrorCost !== undefined) {
          const maxProb = fusedProbs[predicted] ?? 0.0;
          const riskStage1 = (1.0 - maxProb) * this.config.voiErrorCost;
          const riskStage2 =
            (1.0 - this.config.voiStage2ExpectedAccuracy) * this.config.voiErrorCost;
          const netVoi =
            riskStage1 -
            (riskStage2 + this.config.voiStage2RpcCostUsd + this.config.voiLatencyCostUsd);
          voiEsc = netVoi > 0.0;
        }
        const uniformRescue = s1Uniform && this.stage2Backend !== undefined;

        if (hesEsc || voiEsc || uniformRescue) {
          candidateEscalated.push(slotId);
          if (hesEsc) {
            escalationReasons.push(
              `${slotId}:h_raw=${hRaw.toFixed(4)}>=${this.config.hesitationThreshold}`,
            );
          }
          if (voiEsc) {
            escalationReasons.push(`${slotId}:voi_positive`);
          }
        }

        slotsOut.set(slotId, {
          slotId,
          predicted,
          probabilities: fusedProbs,
          rawProbabilities: rawNorm,
          hRaw,
          confidence: 1.0 - hRaw,
          escalated: false,
          passesUsed: 1,
        });
      }

      let didEscalate = false;
      let resolvedBackendName = stage1Response.backendName;

      if (candidateEscalated.length > 0 && this.stage2Backend !== undefined) {
        const executableSlots: string[] = [];
        let hadNegTransferBlock = false;
        for (const slotId of candidateEscalated) {
          const question = questionMap.get(slotId);
          if (!question) {
            continue;
          }
          const blocked =
            question.type === "choice" &&
            this.isSlotNegativeTransfer(this.resolveSlotCompetence(question));
          const s1Uniform = stage1UniformBySlot.get(slotId) === true;
          if (blocked && !s1Uniform) {
            hadNegTransferBlock = true;
          } else {
            if (blocked && s1Uniform) {
              escalationReasons.push(`${slotId}:stage1_uniform_abstention_rescue`);
            }
            executableSlots.push(slotId);
          }
        }
        if (hadNegTransferBlock) {
          escalationReasons.push("domain_competence:negative_transfer_short_circuit");
        }

        if (executableSlots.length === 0) {
          stage2AbortController.abort();
        } else {
          throwIfSignalAborted(request.signal);
          didEscalate = true;
          const stage2RawBySlot = new Map<string, Record<string, number>>();
          const evaluatedOptsBySlot = new Map<string, readonly string[]>();

          const needSpec = executableSlots.some((slotId) => speculativeCovered.has(slotId));
          if (speculativePromise !== undefined && needSpec) {
            const specResp = await raceWithAbortSignal(speculativePromise, request.signal);
            resolvedBackendName = specResp.backendName;
            for (const slotId of executableSlots) {
              if (speculativeCovered.has(slotId) && Object.hasOwn(specResp.slots, slotId)) {
                const slotObj = specResp.slots[slotId];
                const question = questionMap.get(slotId);
                if (slotObj && question) {
                  stage2RawBySlot.set(slotId, slotObj.rawProbabilities ?? slotObj.probabilities);
                  evaluatedOptsBySlot.set(slotId, getQuestionOptionKeys(question));
                }
              }
            }
          } else {
            stage2AbortController.abort();
          }

          const twoPhaseQuestions = nullProtoRecord<DecisionQuestion>();
          let hasTwoPhase = false;
          for (const slotId of executableSlots) {
            if (stage2RawBySlot.has(slotId)) {
              continue;
            }
            const baseQuestion = questionMap.get(slotId);
            if (!baseQuestion) {
              continue;
            }
            const baseKeys = getQuestionOptionKeys(baseQuestion);
            const s1Slot = slotsOut.get(slotId);
            if (
              s1Slot &&
              this.config.stage2TopK >= 2 &&
              baseQuestion.type === "choice" &&
              baseKeys.length > this.config.stage2TopK
            ) {
              const prunedQ = pruneChoiceTopK(
                baseQuestion,
                s1Slot.probabilities,
                this.config.stage2TopK,
              );
              twoPhaseQuestions[slotId] = prunedQ;
              evaluatedOptsBySlot.set(slotId, getQuestionOptionKeys(prunedQ));
            } else {
              twoPhaseQuestions[slotId] = baseQuestion;
              evaluatedOptsBySlot.set(slotId, baseKeys);
            }
            hasTwoPhase = true;
          }

          if (hasTwoPhase) {
            const twoPhasePriorLogits = filterStage2PriorLogits(
              request.priorLogits,
              twoPhaseQuestions,
            );
            const twoPhaseResp = await invokeCascadeBackend(this.stage2Backend, {
              state: request.state,
              questions: { ...twoPhaseQuestions },
              images: request.images,
              ...(twoPhasePriorLogits !== undefined ? { priorLogits: twoPhasePriorLogits } : {}),
              temperature: effTemp,
              signal: request.signal,
            });
            resolvedBackendName = twoPhaseResp.backendName;
            for (const slotId of Object.keys(twoPhaseQuestions)) {
              const slotObj = Object.hasOwn(twoPhaseResp.slots, slotId)
                ? twoPhaseResp.slots[slotId]
                : undefined;
              if (!slotObj) {
                throw new MDMValidationError(
                  `Stage-2 response missing slot ${JSON.stringify(slotId)}`,
                );
              }
              stage2RawBySlot.set(slotId, slotObj.rawProbabilities ?? slotObj.probabilities);
            }
          }

          for (const slotId of executableSlots) {
            const question = questionMap.get(slotId);
            const s1Slot = slotsOut.get(slotId);
            if (!question || !s1Slot) {
              continue;
            }
            const fullKeys = getQuestionOptionKeys(question);
            const evalKeys = evaluatedOptsBySlot.get(slotId) ?? fullKeys;
            const normSubS2 = validateAndNormalizeSlotDist(
              stage2RawBySlot.get(slotId),
              evalKeys,
              slotId,
            );
            const s2Uniform = isUniformDistribution(normSubS2);
            const expandedS2Map = nullProtoRecord<number>();
            for (const key of fullKeys) {
              const subProb = Object.hasOwn(normSubS2, key) ? normSubS2[key] : undefined;
              expandedS2Map[key] = subProb ?? 0.0;
            }
            const expandedS2 = { ...expandedS2Map };

            const s1Uniform = stage1UniformBySlot.get(slotId) === true;
            const profile = this.resolveSlotCompetence(question);
            let finalProbs: Record<string, number>;

            if (s1Uniform) {
              finalProbs = expandedS2;
            } else if (s2Uniform) {
              finalProbs = { ...s1Slot.probabilities };
              escalationReasons.push(`${slotId}:stage2_uniform_fallback_preserved_stage1`);
            } else if (this.config.useDawidSkenePoeWeights && profile !== undefined) {
              const numOpts = Math.max(profile.numOptions ?? 2, fullKeys.length);
              const [dsW1, dsW2] = computeDawidSkenePoeWeights(
                profile.stage1Accuracy,
                profile.stage2Accuracy,
                numOpts,
              );
              const dsSum = dsW1 + dsW2;
              const alpha1 = dsW1 / dsSum;
              const alpha2 = dsW2 / dsSum;
              const epsPoe = 1e-12;
              const logits = new Map<string, number>();
              let maxLogit = -Infinity;
              for (const key of fullKeys) {
                const p1 = Math.max(epsPoe, s1Slot.probabilities[key] ?? epsPoe);
                const p2 = Math.max(epsPoe, expandedS2[key] ?? epsPoe);
                const lv = alpha1 * Math.log(p1) + alpha2 * Math.log(p2);
                logits.set(key, lv);
                if (lv > maxLogit) {
                  maxLogit = lv;
                }
              }
              const expMass = new Map<string, number>();
              let denom = 0.0;
              for (const key of fullKeys) {
                const lv = logits.get(key) ?? -Infinity;
                const ev = Math.exp(lv - maxLogit);
                expMass.set(key, ev);
                denom += ev;
              }
              const pooledMap = nullProtoRecord<number>();
              for (const key of fullKeys) {
                pooledMap[key] = (expMass.get(key) ?? 0.0) / denom;
              }
              finalProbs = { ...pooledMap };
            } else {
              finalProbs = expandedS2;
            }

            const finalPred = argmaxOption(finalProbs, fullKeys);
            const finalHRaw = normalizedEntropy(finalProbs);
            slotsOut.set(slotId, {
              slotId,
              predicted: finalPred,
              probabilities: finalProbs,
              rawProbabilities: expandedS2,
              hRaw: finalHRaw,
              confidence: 1.0 - finalHRaw,
              escalated: true,
              passesUsed: 2,
            });
          }
        }
      } else {
        stage2AbortController.abort();
      }

      const slotsRecord = nullProtoRecord<CascadeSlotReadout>();
      for (const [slotId, readout] of slotsOut.entries()) {
        slotsRecord[slotId] = readout;
      }

      return {
        slots: { ...slotsRecord },
        escalated: didEscalate,
        escalationReasons,
        speculativeLaunched,
        speculativeSkippedByCircuitBreaker,
        latencyMs: performance.now() - startedAt,
        backendName: resolvedBackendName,
      };
    } finally {
      stage2AbortController.abort();
      request.signal?.removeEventListener("abort", forwardCallerAbort);
    }
  }
}

/** Runs a one-shot cascade evaluation without constructing a persistent `CascadeOrchestrator`. */
export async function evaluateCascade(
  stage1Backend: CascadeStageBackend,
  stage2Backend: CascadeStageBackend | undefined,
  request: SystemOneDecideRequest,
  config: CascadeConfig = {},
): Promise<CascadeDecisionResponse> {
  const orchestrator = new CascadeOrchestrator(stage1Backend, stage2Backend, config);
  return orchestrator.decide(request);
}
