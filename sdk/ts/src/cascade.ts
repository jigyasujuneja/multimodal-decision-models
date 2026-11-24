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
  type DecisionQuestion,
  type DecisionResponse,
  type DomainCompetenceProfile,
  MDMValidationError,
  type NegativeTransferOptions,
  type PriorGateMode,
  type SlotReadout,
  type SystemOneDecideRequest,
} from "./types.js";

export const DEFAULT_CASCADE_HESITATION_THRESHOLD = 0.16;

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

export interface CascadeSlotReadout extends SlotReadout {
  readonly escalated: boolean;
  readonly passesUsed: number;
}

export interface CascadeDecisionResponse {
  readonly slots: Record<string, CascadeSlotReadout>;
  readonly escalated: boolean;
  readonly escalationReasons: readonly string[];
  readonly speculativeLaunched: boolean;
  readonly speculativeSkippedByCircuitBreaker: boolean;
  readonly latencyMs: number;
  readonly backendName: string;
}

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

function nullProtoRecord<T>(): Record<string, T> {
  return Object.create(null) as Record<string, T>;
}

function getChoiceOptionKeys(question: DecisionQuestion): string[] {
  if (question.type === "score") {
    return question.levels ? [...question.levels] : [];
  }
  const criteria = question.criteria;
  const descriptions = question.optionDescriptions;
  if (question.order && question.order.length > 0) {
    return question.order.filter(
      (key) =>
        (criteria !== undefined && Object.hasOwn(criteria, key)) ||
        (descriptions !== undefined && Object.hasOwn(descriptions, key)),
    );
  }
  if (criteria !== undefined && Object.keys(criteria).length > 0) {
    return Object.keys(criteria);
  }
  return descriptions ? Object.keys(descriptions) : [];
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal || !signal.aborted) {
    return;
  }
  throw signal.reason ?? new DOMException("The operation was aborted.", "AbortError");
}

function raceWithAbortSignal<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) {
    return promise;
  }
  throwIfAborted(signal);
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
  throwIfAborted(request.signal);
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
  throwIfAborted(request.signal);
  return response;
}

export function pruneChoiceTopK(
  question: DecisionQuestion,
  stage1Probs: Record<string, number>,
  topK: number,
): DecisionQuestion {
  if (!question || typeof question !== "object" || question.type !== "choice") {
    throw new MDMValidationError("pruneChoiceTopK requires a choice DecisionQuestion");
  }
  const optionKeys = getChoiceOptionKeys(question);
  if (optionKeys.length < 2) {
    throw new MDMValidationError("choice DecisionQuestion requires >= 2 options");
  }
  if (typeof topK !== "number" || !Number.isInteger(topK) || topK < 2) {
    throw new MDMValidationError("topK must be an integer >= 2");
  }
  if (!stage1Probs || typeof stage1Probs !== "object" || Array.isArray(stage1Probs)) {
    throw new MDMValidationError("stage1Probs must be a non-empty probability object");
  }

  const indexByKey = nullProtoRecord<number>();
  const validatedScores = nullProtoRecord<number>();
  let totalMass = 0.0;
  for (let idx = 0; idx < optionKeys.length; idx += 1) {
    const key = optionKeys[idx] ?? "";
    if (key.trim() === "") {
      throw new MDMValidationError("choice option keys must be non-empty strings");
    }
    if (Object.hasOwn(indexByKey, key)) {
      throw new MDMValidationError(`duplicate choice option key ${JSON.stringify(key)}`);
    }
    indexByKey[key] = idx;
    const rawVal = Object.hasOwn(stage1Probs, key) ? stage1Probs[key] : 0.0;
    if (typeof rawVal !== "number" || !Number.isFinite(rawVal) || rawVal < 0.0) {
      throw new MDMValidationError(
        `stage1Probs[${JSON.stringify(key)}] must be a finite number >= 0`,
      );
    }
    validatedScores[key] = rawVal;
    totalMass += rawVal;
  }
  if (!Number.isFinite(totalMass) || totalMass <= 0.0) {
    throw new MDMValidationError("stage1Probs must have positive total mass over question options");
  }

  const rankedKeys = [...optionKeys].sort((left, right) => {
    const leftScore = validatedScores[left] ?? 0.0;
    const rightScore = validatedScores[right] ?? 0.0;
    if (leftScore !== rightScore) {
      return rightScore - leftScore;
    }
    return (indexByKey[left] ?? 0) - (indexByKey[right] ?? 0);
  });
  const keptKeys = rankedKeys.slice(0, Math.min(topK, rankedKeys.length));

  let keptCriteria: Record<string, string> | undefined;
  if (question.criteria) {
    keptCriteria = nullProtoRecord<string>();
    for (const key of keptKeys) {
      if (Object.hasOwn(question.criteria, key)) {
        keptCriteria[key] = question.criteria[key] as string;
      }
    }
  }

  let keptDescriptions: Record<string, string> | undefined;
  if (question.optionDescriptions) {
    keptDescriptions = nullProtoRecord<string>();
    for (const key of keptKeys) {
      if (Object.hasOwn(question.optionDescriptions, key)) {
        keptDescriptions[key] = question.optionDescriptions[key] as string;
      }
    }
  }

  let keptPriors: Record<string, number> | undefined;
  if (question.priorLogits) {
    keptPriors = nullProtoRecord<number>();
    for (const key of keptKeys) {
      if (Object.hasOwn(question.priorLogits, key)) {
        keptPriors[key] = question.priorLogits[key] as number;
      }
    }
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
    // Validate domainCompetence and negativeTransferOptions upfront.
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
  rawDist: Record<string, number> | undefined,
  optionKeys: readonly string[],
  slotId: string,
): Record<string, number> {
  if (!rawDist || typeof rawDist !== "object") {
    throw new MDMValidationError(`missing probabilities for slot ${JSON.stringify(slotId)}`);
  }
  const normalized = nullProtoRecord<number>();
  let total = 0.0;
  for (const key of optionKeys) {
    const val = Object.hasOwn(rawDist, key) ? rawDist[key] : undefined;
    if (typeof val !== "number" || !Number.isFinite(val) || val < 0.0) {
      throw new MDMValidationError(
        `slot ${JSON.stringify(slotId)} missing valid probability for ${JSON.stringify(key)}`,
      );
    }
    normalized[key] = val;
    total += val;
  }
  if (!Number.isFinite(total) || total <= 0.0) {
    throw new MDMValidationError(`slot ${JSON.stringify(slotId)} probabilities sum to zero`);
  }
  for (const key of optionKeys) {
    normalized[key] = (normalized[key] as number) / total;
  }
  return normalized;
}

function argmaxOption(probs: Record<string, number>, optionKeys: readonly string[]): string {
  let bestKey = optionKeys[0] ?? "";
  let bestProb = -1.0;
  for (const key of optionKeys) {
    const prob = probs[key] ?? 0.0;
    if (prob > bestProb) {
      bestProb = prob;
      bestKey = key;
    }
  }
  return bestKey;
}

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
    if (validated.useDawidSkenePoeWeights && validated.domainCompetence === undefined) {
      throw new MDMValidationError(
        "useDawidSkenePoeWeights=true requires domainCompetence to be set",
      );
    }
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

  async decide(request: SystemOneDecideRequest): Promise<CascadeDecisionResponse> {
    const startedAt = performance.now();
    throwIfAborted(request.signal);

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
    for (const [slotId, question] of questionEntries) {
      const optionKeys = getChoiceOptionKeys(question);
      if (optionKeys.length < 2) {
        throw new MDMValidationError(`slot ${JSON.stringify(slotId)} requires >= 2 options`);
      }
      if (question.domainCompetence !== undefined) {
        isNegativeTransfer(question.domainCompetence, this.config.negativeTransferOptions ?? {});
      }
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
          speculativePromise = invokeCascadeBackend(this.stage2Backend, {
            state: request.state,
            questions: specQuestions,
            images: request.images,
            priorLogits: request.priorLogits,
            temperature: request.temperature ?? this.config.temperature,
            signal: stage2AbortController.signal,
          });
          // Suppress unhandled rejection when fast-exit cancels speculative Stage 2.
          speculativePromise.catch(() => undefined);
        }
      }

      const stage1Response = await invokeCascadeBackend(this.stage1Backend, {
        ...request,
        temperature: request.temperature ?? this.config.temperature,
      });

      const slotsOut = nullProtoRecord<CascadeSlotReadout>();
      const candidateEscalated: string[] = [];
      const escalationReasons: string[] = [];
      const stage1UniformBySlot = nullProtoRecord<boolean>();

      for (const [slotId, question] of questionEntries) {
        const optionKeys = getChoiceOptionKeys(question);
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
        stage1UniformBySlot[slotId] = s1Uniform;

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

        slotsOut[slotId] = {
          slotId,
          predicted,
          probabilities: fusedProbs,
          rawProbabilities: rawNorm,
          hRaw,
          confidence: 1.0 - hRaw,
          escalated: false,
          passesUsed: 1,
        };
      }

      let didEscalate = false;
      let resolvedBackendName = stage1Response.backendName;

      if (candidateEscalated.length > 0 && this.stage2Backend !== undefined) {
        const executableSlots: string[] = [];
        let hadNegTransferBlock = false;
        for (const slotId of candidateEscalated) {
          const question = request.questions[slotId] as DecisionQuestion;
          const blocked =
            question.type === "choice" &&
            this.isSlotNegativeTransfer(this.resolveSlotCompetence(question));
          const s1Uniform = Boolean(stage1UniformBySlot[slotId]);
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
          throwIfAborted(request.signal);
          didEscalate = true;
          const stage2RawBySlot = nullProtoRecord<Record<string, number>>();
          const evaluatedOptsBySlot = nullProtoRecord<readonly string[]>();

          const needSpec = executableSlots.some((slotId) => speculativeCovered.has(slotId));
          if (speculativePromise !== undefined && needSpec) {
            const specResp = await raceWithAbortSignal(speculativePromise, request.signal);
            resolvedBackendName = specResp.backendName;
            for (const slotId of executableSlots) {
              if (speculativeCovered.has(slotId) && Object.hasOwn(specResp.slots, slotId)) {
                const slotObj = specResp.slots[slotId] as SlotReadout;
                stage2RawBySlot[slotId] = slotObj.rawProbabilities ?? slotObj.probabilities;
                evaluatedOptsBySlot[slotId] = getChoiceOptionKeys(
                  request.questions[slotId] as DecisionQuestion,
                );
              }
            }
          } else {
            stage2AbortController.abort();
          }

          const twoPhaseQuestions = nullProtoRecord<DecisionQuestion>();
          let hasTwoPhase = false;
          for (const slotId of executableSlots) {
            if (Object.hasOwn(stage2RawBySlot, slotId)) {
              continue;
            }
            const baseQuestion = request.questions[slotId] as DecisionQuestion;
            const baseKeys = getChoiceOptionKeys(baseQuestion);
            if (
              this.config.stage2TopK >= 2 &&
              baseQuestion.type === "choice" &&
              baseKeys.length > this.config.stage2TopK
            ) {
              const s1Slot = slotsOut[slotId] as CascadeSlotReadout;
              const prunedQ = pruneChoiceTopK(
                baseQuestion,
                s1Slot.probabilities,
                this.config.stage2TopK,
              );
              twoPhaseQuestions[slotId] = prunedQ;
              evaluatedOptsBySlot[slotId] = getChoiceOptionKeys(prunedQ);
            } else {
              twoPhaseQuestions[slotId] = baseQuestion;
              evaluatedOptsBySlot[slotId] = baseKeys;
            }
            hasTwoPhase = true;
          }

          if (hasTwoPhase) {
            const twoPhaseResp = await invokeCascadeBackend(this.stage2Backend, {
              state: request.state,
              questions: twoPhaseQuestions,
              images: request.images,
              priorLogits: request.priorLogits,
              temperature: request.temperature ?? this.config.temperature,
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
              stage2RawBySlot[slotId] = slotObj.rawProbabilities ?? slotObj.probabilities;
            }
          }

          for (const slotId of executableSlots) {
            const question = request.questions[slotId] as DecisionQuestion;
            const fullKeys = getChoiceOptionKeys(question);
            const evalKeys = evaluatedOptsBySlot[slotId] ?? fullKeys;
            const normSubS2 = validateAndNormalizeSlotDist(
              stage2RawBySlot[slotId],
              evalKeys,
              slotId,
            );
            const s2Uniform = isUniformDistribution(normSubS2);
            const expandedS2 = nullProtoRecord<number>();
            for (const key of fullKeys) {
              expandedS2[key] = Object.hasOwn(normSubS2, key) ? (normSubS2[key] as number) : 0.0;
            }

            const s1Slot = slotsOut[slotId] as CascadeSlotReadout;
            const s1Uniform = Boolean(stage1UniformBySlot[slotId]);
            const profile = this.resolveSlotCompetence(question);
            let finalProbs: Record<string, number>;

            if (s1Uniform && !s2Uniform) {
              finalProbs = expandedS2;
            } else if (!s1Uniform && s2Uniform) {
              finalProbs = { ...s1Slot.probabilities };
              escalationReasons.push(`${slotId}:stage2_uniform_fallback_preserved_stage1`);
            } else if (s1Uniform && s2Uniform) {
              finalProbs = expandedS2;
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
              const logits = nullProtoRecord<number>();
              let maxLogit = -Infinity;
              for (const key of fullKeys) {
                const p1 = Math.max(epsPoe, s1Slot.probabilities[key] ?? epsPoe);
                const p2 = Math.max(epsPoe, expandedS2[key] ?? epsPoe);
                const lv = alpha1 * Math.log(p1) + alpha2 * Math.log(p2);
                logits[key] = lv;
                if (lv > maxLogit) {
                  maxLogit = lv;
                }
              }
              finalProbs = nullProtoRecord<number>();
              let denom = 0.0;
              for (const key of fullKeys) {
                const ev = Math.exp((logits[key] as number) - maxLogit);
                finalProbs[key] = ev;
                denom += ev;
              }
              for (const key of fullKeys) {
                finalProbs[key] = (finalProbs[key] as number) / denom;
              }
            } else {
              finalProbs = expandedS2;
            }

            const finalPred = argmaxOption(finalProbs, fullKeys);
            const finalHRaw = normalizedEntropy(finalProbs);
            slotsOut[slotId] = {
              slotId,
              predicted: finalPred,
              probabilities: finalProbs,
              rawProbabilities: expandedS2,
              hRaw: finalHRaw,
              confidence: 1.0 - finalHRaw,
              escalated: true,
              passesUsed: 2,
            };
          }
        }
      } else {
        stage2AbortController.abort();
      }

      return {
        slots: slotsOut,
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

export async function evaluateCascade(
  stage1Backend: CascadeStageBackend,
  stage2Backend: CascadeStageBackend | undefined,
  request: SystemOneDecideRequest,
  config: CascadeConfig = {},
): Promise<CascadeDecisionResponse> {
  const orchestrator = new CascadeOrchestrator(stage1Backend, stage2Backend, config);
  return orchestrator.decide(request);
}
