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

import { createHmac } from "node:crypto";
import {
  type BayesianPoEResult,
  type ConformalCalibrationResult,
  type ConformalConfig,
  type ConformalMethod,
  type ConformalSetResult,
  MDMValidationError,
  type MultiSlotJointConformalResult,
  type PriorGateMode,
} from "./types.js";

export const DEFAULT_COSINE_PRIOR_TEMPERATURE = 0.07;
export const DEFAULT_CONFORMAL_SEED_KEY = "dgem-conformal-v1";
export const CONFORMAL_SCORE_TOLERANCE = 1e-12;
export const MAX_TEMPERATURE = 20.0;

export const DEFAULT_ABSTAIN_LABELS: readonly string[] = [
  "0",
  "abstain",
  "insufficient_evidence",
  "none",
  "none_of_the_above",
  "none_visible",
  "oos",
  "uncertain",
  "unknown",
];

interface OptionProbability {
  readonly name: string;
  readonly prob: number;
}

function nullProtoRecord<T = number>(): Record<string, T> {
  return Object.create(null) as Record<string, T>;
}

function hasOwn(target: object, key: string): boolean {
  return Object.hasOwn(target, key);
}

function normalizeProbabilities(
  probabilities: Record<string, number>,
): OptionProbability[] {
  if (!probabilities || typeof probabilities !== "object") {
    throw new MDMValidationError("probability map must be an object");
  }
  const keys = Object.keys(probabilities).sort();
  if (keys.length === 0) {
    throw new MDMValidationError("probability map must be non-empty");
  }
  let totalMass = 0.0;
  const rawItems: OptionProbability[] = [];
  for (const key of keys) {
    if (!hasOwn(probabilities, key)) {
      continue;
    }
    const val = probabilities[key];
    if (typeof val !== "number" || !Number.isFinite(val) || val < 0.0) {
      throw new MDMValidationError(
        `probability for ${JSON.stringify(key)} must be finite and >= 0`,
      );
    }
    rawItems.push({ name: key, prob: val });
    totalMass += val;
  }
  if (!Number.isFinite(totalMass) || totalMass <= 0.0) {
    throw new MDMValidationError("probability map total mass must be > 0");
  }
  const normalized = rawItems.map((item) => ({
    name: item.name,
    prob: item.prob / totalMass,
  }));
  normalized.sort((left, right) => {
    if (right.prob !== left.prob) {
      return right.prob - left.prob;
    }
    return left.name < right.name ? -1 : left.name > right.name ? 1 : 0;
  });
  return normalized;
}

export function shannonEntropy(probabilities: Record<string, number>): number {
  const ordered = normalizeProbabilities(probabilities);
  let entropyNats = 0.0;
  for (const item of ordered) {
    if (item.prob > 0.0) {
      entropyNats -= item.prob * Math.log(item.prob);
    }
  }
  return Math.max(0.0, entropyNats);
}

export function normalizedEntropy(
  probabilities: Record<string, number>,
): number {
  const ordered = normalizeProbabilities(probabilities);
  if (ordered.length <= 1) {
    return 0.0;
  }
  let entropyNats = 0.0;
  for (const item of ordered) {
    if (item.prob > 0.0) {
      entropyNats -= item.prob * Math.log(item.prob);
    }
  }
  return Math.max(0.0, Math.min(1.0, entropyNats / Math.log(ordered.length)));
}

export function applyTemperature(
  probabilities: Record<string, number>,
  temperature: number,
): Record<string, number> {
  if (
    typeof temperature !== "number" ||
    !Number.isFinite(temperature) ||
    temperature <= 0.0 ||
    temperature > MAX_TEMPERATURE
  ) {
    throw new MDMValidationError(
      `temperature must lie in (0, ${MAX_TEMPERATURE}] (got ${String(temperature)})`,
    );
  }
  const ordered = normalizeProbabilities(probabilities);
  if (Math.abs(temperature - 1.0) <= 1e-12) {
    const cloned = nullProtoRecord<number>();
    for (const item of ordered) {
      cloned[item.name] = item.prob;
    }
    return { ...cloned };
  }

  let maxLogit = -Infinity;
  const scaled: Array<{ name: string; logit: number }> = [];
  for (const item of ordered) {
    const logit = item.prob <= 0.0 ? -1e9 : Math.log(item.prob) / temperature;
    scaled.push({ name: item.name, logit });
    if (logit > maxLogit) {
      maxLogit = logit;
    }
  }
  let normalizer = 0.0;
  const expScores: Array<{ name: string; score: number }> = [];
  for (const item of scaled) {
    const score = Math.exp(item.logit - maxLogit);
    expScores.push({ name: item.name, score });
    normalizer += score;
  }
  const out = nullProtoRecord<number>();
  for (const item of expScores) {
    out[item.name] = item.score / normalizer;
  }
  return { ...out };
}

export function normalizeCosinePriorLogits(
  cosineScores: Record<string, number>,
  temperature = DEFAULT_COSINE_PRIOR_TEMPERATURE,
): Record<string, number> {
  const keys = Object.keys(cosineScores);
  if (keys.length === 0) {
    return {};
  }
  const tau =
    typeof temperature === "number" && Number.isFinite(temperature) && temperature > 0.0
      ? temperature
      : DEFAULT_COSINE_PRIOR_TEMPERATURE;

  const scaled = nullProtoRecord<number>();
  let maxScaled = -Infinity;
  for (const key of keys) {
    const rawVal = hasOwn(cosineScores, key) ? cosineScores[key] : undefined;
    const cleanVal =
      typeof rawVal === "number" && Number.isFinite(rawVal) ? rawVal : -1.0;
    const score = cleanVal / tau;
    scaled[key] = score;
    if (score > maxScaled) {
      maxScaled = score;
    }
  }
  let sumExp = 0.0;
  for (const key of keys) {
    const val = hasOwn(scaled, key) ? scaled[key] : undefined;
    sumExp += Math.exp((val ?? -Infinity) - maxScaled);
  }
  const logSumExp = maxScaled + Math.log(sumExp);
  const out = nullProtoRecord<number>();
  for (const key of keys) {
    const val = hasOwn(scaled, key) ? scaled[key] : undefined;
    out[key] = (val ?? 0.0) - logSumExp;
  }
  return { ...out };
}

export function fusePriorProbabilities(
  rawProbabilities: Record<string, number>,
  options: {
    readonly priorLogits?: Record<string, number>;
    readonly priorWeight?: number;
    readonly hesitationThreshold?: number;
    readonly gateMode?: PriorGateMode;
  } = {},
): BayesianPoEResult {
  const ordered = normalizeProbabilities(rawProbabilities);
  const baseMap = nullProtoRecord<number>();
  for (const item of ordered) {
    baseMap[item.name] = item.prob;
  }
  const hRaw = normalizedEntropy(baseMap);
  const threshold = options.hesitationThreshold ?? 0.16;
  const escalated = threshold > 0.0 && hRaw >= threshold;
  const priorLogits = options.priorLogits;
  const priorWeight = options.priorWeight ?? 0.0;
  const gateMode: PriorGateMode = options.gateMode ?? "hesitant";

  if (
    gateMode === "off" ||
    gateMode === "never" ||
    !priorLogits ||
    Object.keys(priorLogits).length === 0 ||
    !Number.isFinite(priorWeight) ||
    priorWeight <= 0.0 ||
    ordered.length <= 1
  ) {
    return { probabilities: { ...baseMap }, hRaw, escalated };
  }

  let effectiveWeight = priorWeight;
  if (gateMode === "confident" && threshold > 0.0 && hRaw >= threshold) {
    return { probabilities: { ...baseMap }, hRaw, escalated };
  }
  if (gateMode === "hesitant" && threshold > 0.0 && hRaw < threshold) {
    return { probabilities: { ...baseMap }, hRaw, escalated };
  }
  if (gateMode === "proportional") {
    effectiveWeight = priorWeight * hRaw;
    if (effectiveWeight <= 1e-9) {
      return { probabilities: { ...baseMap }, hRaw, escalated };
    }
  }

  let maxLogit = -Infinity;
  const logits = nullProtoRecord<number>();
  for (const item of ordered) {
    const baseProb = Math.max(1e-9, item.prob);
    let priorVal = 0.0;
    if (hasOwn(priorLogits, item.name)) {
      const candidatePrior = priorLogits[item.name];
      if (typeof candidatePrior !== "number" || !Number.isFinite(candidatePrior)) {
        throw new MDMValidationError(
          `priorLogits[${JSON.stringify(item.name)}] must be a finite number`,
        );
      }
      priorVal = candidatePrior;
    }
    const combined = Math.log(baseProb) + effectiveWeight * priorVal;
    logits[item.name] = combined;
    if (combined > maxLogit) {
      maxLogit = combined;
    }
  }

  let total = 0.0;
  const unnormalized = nullProtoRecord<number>();
  for (const item of ordered) {
    const itemLogit = hasOwn(logits, item.name) ? logits[item.name] : undefined;
    const expVal = Math.exp((itemLogit ?? -Infinity) - maxLogit);
    unnormalized[item.name] = expVal;
    total += expVal;
  }
  const fused = nullProtoRecord<number>();
  for (const item of ordered) {
    const itemMass = hasOwn(unnormalized, item.name) ? unnormalized[item.name] : undefined;
    fused[item.name] = (itemMass ?? 0.0) / total;
  }
  return { probabilities: { ...fused }, hRaw, escalated };
}

export function hmacUint64(
  seedKey: string,
  itemKey: string,
  tag: string,
): bigint {
  if (
    typeof seedKey !== "string" ||
    typeof itemKey !== "string" ||
    typeof tag !== "string"
  ) {
    throw new MDMValidationError("seedKey, itemKey, and tag must be strings");
  }
  const effectiveKey = seedKey !== "" ? seedKey : DEFAULT_CONFORMAL_SEED_KEY;
  const mac = createHmac("sha256", Buffer.from(effectiveKey, "utf8"));
  mac.update(Buffer.from(itemKey, "utf8"));
  mac.update(Buffer.from([0]));
  mac.update(Buffer.from(tag, "utf8"));
  const digest = mac.digest();
  return digest.readBigUInt64BE(0);
}

export function hmacUniformFloat64(
  seedKey: string,
  itemKey: string,
  tag: string,
): number {
  const top53 = Number(hmacUint64(seedKey, itemKey, tag) >> 11n);
  return top53 * (1.0 / 9007199254740992.0);
}

function partitionTieGroups(
  ordered: readonly OptionProbability[],
): OptionProbability[][] {
  const groups: OptionProbability[][] = [];
  for (const item of ordered) {
    const prevGroup = groups[groups.length - 1];
    const prevLead = prevGroup?.[0];
    if (
      !prevGroup ||
      !prevLead ||
      Math.abs(item.prob - prevLead.prob) > CONFORMAL_SCORE_TOLERANCE
    ) {
      groups.push([item]);
    } else {
      prevGroup.push(item);
    }
  }
  return groups;
}

function validateConformalConfig(config: ConformalConfig = {}): {
  method: ConformalMethod;
  lamReg: number;
  kReg: number;
  randomized: boolean;
  seedKey: string;
  itemKey: string;
} {
  const rawMethod = (config.method ?? "raps").toLowerCase();
  if (rawMethod !== "lac" && rawMethod !== "aps" && rawMethod !== "raps") {
    throw new MDMValidationError(
      `unsupported conformal method ${JSON.stringify(config.method)}`,
    );
  }
  const lamReg = config.lamReg ?? 0.01;
  if (typeof lamReg !== "number" || !Number.isFinite(lamReg) || lamReg < 0.0) {
    throw new MDMValidationError("lamReg must be finite and >= 0");
  }
  const kReg = config.kReg ?? 1;
  if (!Number.isInteger(kReg) || kReg < 0) {
    throw new MDMValidationError("kReg must be an integer >= 0");
  }
  const randomized = Boolean(config.randomized);
  const seedKey = config.seedKey ?? DEFAULT_CONFORMAL_SEED_KEY;
  const itemKey = config.itemKey ?? "";
  return {
    method: rawMethod,
    lamReg,
    kReg,
    randomized,
    seedKey,
    itemKey,
  };
}

function computeNonconformityScores(
  ordered: readonly OptionProbability[],
  method: ConformalMethod,
  lamReg: number,
  kReg: number,
  randomized: boolean,
  seedKey: string,
  itemKey: string,
): Record<string, number> {
  const scores = nullProtoRecord<number>();
  const groups = partitionTieGroups(ordered);

  if (!randomized) {
    let cumStrictlyGreater = 0.0;
    let strictCount = 0;
    for (const grp of groups) {
      const lead = grp[0];
      if (!lead) {
        continue;
      }
      const pGrp = lead.prob;
      const strictRank = strictCount + 1;
      let grpScore: number;
      if (method === "lac") {
        grpScore = 1.0 - pGrp;
      } else if (method === "aps") {
        grpScore = cumStrictlyGreater;
      } else {
        const penalty = lamReg * Math.max(0, strictRank - kReg);
        grpScore = cumStrictlyGreater + penalty;
      }
      for (const member of grp) {
        scores[member.name] = grpScore;
      }
      cumStrictlyGreater += pGrp * grp.length;
      strictCount += grp.length;
    }
    return scores;
  }

  const uDraw = hmacUniformFloat64(seedKey, itemKey, "u");
  const randomizedOrder: OptionProbability[] = [];
  for (const grp of groups) {
    if (grp.length > 1) {
      const decorated = grp.map((member) => ({
        member,
        hash: hmacUint64(seedKey, itemKey, `tie:${member.name}`),
      }));
      decorated.sort((left, right) => {
        if (left.hash !== right.hash) {
          return left.hash < right.hash ? -1 : 1;
        }
        return left.member.name < right.member.name
          ? -1
          : left.member.name > right.member.name
            ? 1
            : 0;
      });
      for (const entry of decorated) {
        randomizedOrder.push(entry.member);
      }
    } else if (grp[0]) {
      randomizedOrder.push(grp[0]);
    }
  }

  let cumBefore = 0.0;
  for (let idx = 0; idx < randomizedOrder.length; idx += 1) {
    const entry = randomizedOrder[idx];
    if (!entry) {
      continue;
    }
    const rankOneBased = idx + 1;
    if (method === "lac") {
      scores[entry.name] = 1.0 - entry.prob;
    } else if (method === "aps") {
      scores[entry.name] = cumBefore + uDraw * entry.prob;
    } else {
      const penalty = lamReg * Math.max(0, rankOneBased - kReg);
      scores[entry.name] = cumBefore + uDraw * entry.prob + penalty;
    }
    cumBefore += entry.prob;
  }
  return scores;
}

export function calibrateConformalQuantile(
  calibProbabilities: readonly Record<string, number>[],
  calibLabels: readonly string[],
  alpha = 0.1,
  config: ConformalConfig & { readonly itemKeys?: readonly string[] } = {},
): ConformalCalibrationResult {
  if (
    typeof alpha !== "number" ||
    !Number.isFinite(alpha) ||
    alpha <= 0.0 ||
    alpha >= 1.0
  ) {
    throw new MDMValidationError("alpha must lie strictly in (0, 1)");
  }
  if (
    calibProbabilities.length === 0 ||
    calibProbabilities.length !== calibLabels.length
  ) {
    throw new MDMValidationError(
      "calibProbabilities and calibLabels must be non-empty arrays of equal length",
    );
  }
  if (config.itemKeys && config.itemKeys.length !== calibProbabilities.length) {
    throw new MDMValidationError("itemKeys length must match calibProbabilities");
  }
  const validated = validateConformalConfig(config);
  const goldScores: number[] = [];

  for (let idx = 0; idx < calibProbabilities.length; idx += 1) {
    const dist = calibProbabilities[idx];
    const goldLabel = calibLabels[idx];
    if (!dist || typeof goldLabel !== "string") {
      throw new MDMValidationError(`invalid calibration pair at index ${idx}`);
    }
    const ordered = normalizeProbabilities(dist);
    const perExampleKey = config.itemKeys?.[idx] ?? `cal:${idx}`;
    const itemScores = computeNonconformityScores(
      ordered,
      validated.method,
      validated.lamReg,
      validated.kReg,
      validated.randomized,
      validated.seedKey,
      perExampleKey,
    );
    const goldScore = hasOwn(itemScores, goldLabel)
      ? itemScores[goldLabel]
      : undefined;
    if (typeof goldScore !== "number" || !Number.isFinite(goldScore)) {
      throw new MDMValidationError(
        `gold label ${JSON.stringify(goldLabel)} missing at calibration index ${idx}`,
      );
    }
    goldScores.push(goldScore);
  }

  const nCalib = goldScores.length;
  const sortedScores = [...goldScores].sort((a, b) => a - b);
  const orderIdx = Math.ceil((nCalib + 1) * (1.0 - alpha) - 1e-12);
  let qHat: number;
  let empiricalCoverage: number;
  if (orderIdx > nCalib) {
    qHat = Infinity;
    empiricalCoverage = 1.0;
  } else {
    qHat = sortedScores[Math.max(0, orderIdx - 1)] ?? Infinity;
    let covered = 0;
    for (const s of goldScores) {
      if (s <= qHat + CONFORMAL_SCORE_TOLERANCE) {
        covered += 1;
      }
    }
    empiricalCoverage = covered / nCalib;
  }

  return {
    qHat,
    alpha,
    nCalib,
    method: validated.method,
    empiricalCalibCoverage: empiricalCoverage,
    lamReg: validated.lamReg,
    kReg: validated.kReg,
  };
}

export function conformalPredictionSet(
  probabilities: Record<string, number>,
  qHat: number,
  config: ConformalConfig = {},
): ConformalSetResult {
  if (typeof qHat !== "number" || Number.isNaN(qHat) || qHat < 0.0) {
    throw new MDMValidationError(`qHat must be >= 0 or +Infinity (got ${String(qHat)})`);
  }
  const ordered = normalizeProbabilities(probabilities);
  const validated = validateConformalConfig(config);
  const scores = computeNonconformityScores(
    ordered,
    validated.method,
    validated.lamReg,
    validated.kReg,
    validated.randomized,
    validated.seedKey,
    validated.itemKey,
  );

  const abstainSource = config.abstainLabels ?? DEFAULT_ABSTAIN_LABELS;
  const abstainLookup = new Set<string>();
  for (const label of abstainSource) {
    const trimmed = label.trim().toLowerCase();
    if (trimmed !== "") {
      abstainLookup.add(trimmed);
    }
  }

  const predictionSet: string[] = [];
  let containsAbstain = false;
  for (const item of ordered) {
    const itemScore = hasOwn(scores, item.name)
      ? (scores[item.name] ?? Infinity)
      : Infinity;
    if (!Number.isFinite(qHat) || itemScore <= qHat + CONFORMAL_SCORE_TOLERANCE) {
      predictionSet.push(item.name);
      if (abstainLookup.has(item.name.trim().toLowerCase())) {
        containsAbstain = true;
      }
    }
  }

  const topEntry = ordered[0];
  const topOption = topEntry?.name ?? "";
  const topProbability = topEntry?.prob ?? 0.0;
  const singleton = predictionSet.length === 1;

  return {
    predictionSet,
    scores: { ...scores },
    qHat,
    topOption,
    topProbability,
    setSize: predictionSet.length,
    singleton,
    containsAbstain,
    shouldEscalate: !singleton || containsAbstain,
  };
}

export function calibrateMultiSlotJointConformal(
  calibSlotProbabilities: readonly Record<string, Record<string, number>>[],
  calibSlotLabels: readonly Record<string, string>[],
  alpha = 0.1,
  config: ConformalConfig = {},
): MultiSlotJointConformalResult {
  if (
    typeof alpha !== "number" ||
    !Number.isFinite(alpha) ||
    alpha <= 0.0 ||
    alpha >= 1.0
  ) {
    throw new MDMValidationError("alpha must lie strictly in (0, 1)");
  }
  if (
    calibSlotProbabilities.length === 0 ||
    calibSlotProbabilities.length !== calibSlotLabels.length
  ) {
    throw new MDMValidationError(
      "calibSlotProbabilities and calibSlotLabels must be non-empty arrays of equal length",
    );
  }
  const firstRow = calibSlotProbabilities[0];
  if (!firstRow) {
    throw new MDMValidationError("calibSlotProbabilities[0] is missing");
  }
  const slotNames = Object.keys(firstRow).sort();
  if (slotNames.length === 0) {
    throw new MDMValidationError("calibSlotProbabilities items must contain at least one slot");
  }
  const validated = validateConformalConfig({ ...config, randomized: false });

  const maxScores: number[] = [];
  const perSlotProbs = nullProtoRecord<Record<string, number>[]>();
  const perSlotLabels = nullProtoRecord<string[]>();
  for (const slotName of slotNames) {
    perSlotProbs[slotName] = [];
    perSlotLabels[slotName] = [];
  }

  for (let idx = 0; idx < calibSlotProbabilities.length; idx += 1) {
    const rowProbs = calibSlotProbabilities[idx];
    const rowLabels = calibSlotLabels[idx];
    if (!rowProbs || !rowLabels) {
      throw new MDMValidationError(`missing row at calibration index ${idx}`);
    }
    let rowMax = 0.0;
    for (const slotName of slotNames) {
      const slotDist = hasOwn(rowProbs, slotName) ? rowProbs[slotName] : undefined;
      const gold = hasOwn(rowLabels, slotName) ? rowLabels[slotName] : undefined;
      if (!slotDist || typeof gold !== "string") {
        throw new MDMValidationError(
          `missing slot ${JSON.stringify(slotName)} at calibration index ${idx}`,
        );
      }
      const ordered = normalizeProbabilities(slotDist);
      const slotScores = computeNonconformityScores(
        ordered,
        validated.method,
        validated.lamReg,
        validated.kReg,
        false,
        validated.seedKey,
        "",
      );
      const goldScore = hasOwn(slotScores, gold) ? slotScores[gold] : undefined;
      if (typeof goldScore !== "number" || !Number.isFinite(goldScore)) {
        throw new MDMValidationError(
          `label ${JSON.stringify(gold)} missing from slot ${JSON.stringify(slotName)} at index ${idx}`,
        );
      }
      if (goldScore > rowMax) {
        rowMax = goldScore;
      }
      perSlotProbs[slotName]?.push(slotDist);
      perSlotLabels[slotName]?.push(gold);
    }
    maxScores.push(rowMax);
  }

  const nCalib = maxScores.length;
  const sortedMax = [...maxScores].sort((a, b) => a - b);
  const orderIdx = Math.ceil((nCalib + 1) * (1.0 - alpha) - 1e-12);
  let jointMaxQHat: number;
  let empiricalJointCoverage: number;
  if (orderIdx > nCalib) {
    jointMaxQHat = Infinity;
    empiricalJointCoverage = 1.0;
  } else {
    jointMaxQHat = sortedMax[Math.max(0, orderIdx - 1)] ?? Infinity;
    let covered = 0;
    for (const val of maxScores) {
      if (val <= jointMaxQHat + CONFORMAL_SCORE_TOLERANCE) {
        covered += 1;
      }
    }
    empiricalJointCoverage = covered / nCalib;
  }

  const bonferroniAlpha = alpha / slotNames.length;
  const bonferroniSlotQHats = nullProtoRecord<number>();
  for (const slotName of slotNames) {
    const slotCal = calibrateConformalQuantile(
      perSlotProbs[slotName] ?? [],
      perSlotLabels[slotName] ?? [],
      bonferroniAlpha,
      {
        method: validated.method,
        lamReg: validated.lamReg,
        kReg: validated.kReg,
        randomized: false,
      },
    );
    bonferroniSlotQHats[slotName] = slotCal.qHat;
  }

  return {
    jointMaxQHat,
    bonferroniSlotQHats: { ...bonferroniSlotQHats },
    slotNames,
    alpha,
    nCalib,
    empiricalJointCoverage,
    method: validated.method,
    lamReg: validated.lamReg,
    kReg: validated.kReg,
  };
}

export function multiSlotJointConformalSets(
  slotProbabilities: Record<string, Record<string, number>>,
  jointCalibration: MultiSlotJointConformalResult,
  options: {
    readonly useBonferroni?: boolean;
    readonly abstainLabels?: readonly string[];
  } = {},
): Record<string, ConformalSetResult> {
  const out = nullProtoRecord<ConformalSetResult>();
  for (const slotName of jointCalibration.slotNames) {
    const dist = hasOwn(slotProbabilities, slotName)
      ? slotProbabilities[slotName]
      : undefined;
    if (!dist || typeof dist !== "object") {
      throw new MDMValidationError(
        `missing slot ${JSON.stringify(slotName)} in slotProbabilities`,
      );
    }
    const bonfQ = hasOwn(jointCalibration.bonferroniSlotQHats, slotName)
      ? jointCalibration.bonferroniSlotQHats[slotName]
      : undefined;
    const qHat = options.useBonferroni
      ? (bonfQ ?? jointCalibration.jointMaxQHat)
      : jointCalibration.jointMaxQHat;
    out[slotName] = conformalPredictionSet(dist, qHat, {
      method: jointCalibration.method,
      lamReg: jointCalibration.lamReg,
      kReg: jointCalibration.kReg,
      randomized: false,
      abstainLabels: options.abstainLabels,
    });
  }
  return { ...out };
}
