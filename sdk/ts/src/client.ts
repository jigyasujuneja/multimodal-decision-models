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

import { applyTemperature, normalizedEntropy } from "./calibration.js";
import {
  type ChatLogprobDecideRequest,
  type DecisionQuestion,
  type DecisionResponse,
  MDMHTTPError,
  MDMTransportError,
  MDMValidationError,
  type SlotReadout,
  type SystemOneDecideRequest,
} from "./types.js";

const CREDENTIAL_HEADER_NAMES = new Set([
  "authorization",
  "x-dgem-key",
  "x-dgem-api-key",
  "x-goog-api-key",
]);

const REDIRECT_STATUS_CODES = new Set([301, 302, 303, 307, 308]);
const ASCII_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

export type TokenProvider = (
  targetUrl: string,
) => string | undefined | Promise<string | undefined>;

export interface MDMClientOptions {
  readonly baseUrl: string;
  readonly bearerToken?: string;
  readonly tokenProvider?: TokenProvider;
  readonly apiKey?: string;
  readonly apiKeyHeader?: string;
  readonly timeoutMs?: number;
  readonly maxRedirects?: number;
  readonly fetchImpl?: typeof fetch;
}

export function stripTrailingSlashes(input: string): string {
  let end = input.length;
  while (end > 0 && input.charCodeAt(end - 1) === 47) {
    end -= 1;
  }
  return input.slice(0, end);
}

export function parseUrlOrigin(rawUrl: string): string {
  if (rawUrl.includes("\r") || rawUrl.includes("\n")) {
    throw new MDMValidationError("URL must not contain CR or LF characters");
  }
  let parsed: URL;
  try {
    parsed = new URL(rawUrl.trim());
  } catch (err) {
    throw new MDMValidationError(`invalid URL ${JSON.stringify(rawUrl)}: ${String(err)}`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new MDMValidationError(
      `URL must use http:// or https:// (got ${JSON.stringify(rawUrl)})`,
    );
  }
  if (!parsed.hostname) {
    throw new MDMValidationError(
      `URL must specify a non-empty hostname (got ${JSON.stringify(rawUrl)})`,
    );
  }
  if (parsed.username !== "" || parsed.password !== "") {
    throw new MDMValidationError("URL must not embed credentials in userinfo");
  }
  const defaultPort = parsed.protocol === "https:" ? "443" : "80";
  const effectivePort = parsed.port || defaultPort;
  return `${parsed.protocol}//${parsed.hostname.toLowerCase()}:${effectivePort}`;
}

export function isRawPredictEndpoint(rawUrl: string): boolean {
  let parsed: URL;
  try {
    parsed = new URL(rawUrl.trim());
  } catch {
    return false;
  }
  const cleanPath = stripTrailingSlashes(parsed.pathname);
  return cleanPath.endsWith(":rawPredict") || cleanPath.endsWith(":predict");
}

function validateHeaderValue(name: string, value: string): string {
  if (
    name.includes("\r") ||
    name.includes("\n") ||
    value.includes("\r") ||
    value.includes("\n")
  ) {
    throw new MDMValidationError(
      `header ${JSON.stringify(name)} must not contain CR or LF characters`,
    );
  }
  return value;
}

function stripCredentialHeaders(headers: Record<string, string>): void {
  for (const key of Object.keys(headers)) {
    if (CREDENTIAL_HEADER_NAMES.has(key.toLowerCase())) {
      delete headers[key];
    }
  }
}

function getQuestionOptionKeys(question: DecisionQuestion): string[] {
  if (question.type === "score") {
    return question.levels ? [...question.levels] : [];
  }
  const criteria = question.criteria ?? {};
  if (question.order && question.order.length > 0) {
    return question.order.filter((key) => Object.prototype.hasOwnProperty.call(criteria, key));
  }
  return Object.keys(criteria);
}

export class MDMClient {
  private readonly baseUrl: string;
  private readonly bearerToken?: string;
  private readonly tokenProvider?: TokenProvider;
  private readonly apiKey?: string;
  private readonly apiKeyHeader: string;
  private readonly timeoutMs: number;
  private readonly maxRedirects: number;
  private readonly fetchImpl: typeof fetch;

  constructor(options: MDMClientOptions) {
    if (!options || typeof options.baseUrl !== "string" || options.baseUrl.trim() === "") {
      throw new MDMValidationError("MDMClient requires a non-empty baseUrl");
    }
    const cleanBase = stripTrailingSlashes(options.baseUrl.trim());
    parseUrlOrigin(cleanBase);
    this.baseUrl = cleanBase;
    this.bearerToken = options.bearerToken;
    this.tokenProvider = options.tokenProvider;
    this.apiKey = options.apiKey;
    this.apiKeyHeader = options.apiKeyHeader ?? "X-DGEM-API-Key";
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.maxRedirects = options.maxRedirects ?? 5;
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch;
    if (typeof this.fetchImpl !== "function") {
      throw new MDMValidationError("global fetch is unavailable; supply fetchImpl");
    }
  }

  private async buildHeaders(targetUrl: string): Promise<Record<string, string>> {
    const headers: Record<string, string> = {
      Accept: "application/json",
      "Content-Type": "application/json",
    };
    const resolvedToken = this.tokenProvider
      ? await this.tokenProvider(targetUrl)
      : this.bearerToken;
    if (resolvedToken) {
      headers["Authorization"] = validateHeaderValue(
        "Authorization",
        `Bearer ${resolvedToken}`,
      );
    }
    if (this.apiKey) {
      headers[this.apiKeyHeader] = validateHeaderValue(this.apiKeyHeader, this.apiKey);
    }
    return headers;
  }

  private async requestJson(
    method: "GET" | "POST",
    targetUrl: string,
    body?: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    const initialOrigin = parseUrlOrigin(targetUrl);
    let currentUrl = targetUrl;
    let currentMethod: "GET" | "POST" = method;
    let currentBody = body ? JSON.stringify(body) : undefined;
    const headers = await this.buildHeaders(targetUrl);
    let crossedOrigin = false;

    for (let hop = 0; hop <= this.maxRedirects; hop += 1) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      let response: Response;
      try {
        response = await this.fetchImpl(currentUrl, {
          method: currentMethod,
          headers: { ...headers },
          body: currentMethod === "POST" ? currentBody : undefined,
          redirect: "manual",
          signal: controller.signal,
        });
      } catch (err) {
        clearTimeout(timer);
        throw new MDMTransportError(`HTTP transport failed: ${String(err)}`, true);
      }
      clearTimeout(timer);

      if (REDIRECT_STATUS_CODES.has(response.status)) {
        const locationHeader = response.headers.get("location");
        if (!locationHeader) {
          throw new MDMTransportError(
            `HTTP ${response.status} redirect missing Location header`,
            false,
          );
        }
        if (hop === this.maxRedirects) {
          throw new MDMTransportError(
            `exceeded maxRedirects (${this.maxRedirects})`,
            false,
          );
        }
        const resolvedNext = new URL(locationHeader, currentUrl).toString();
        const nextOrigin = parseUrlOrigin(resolvedNext);
        if (nextOrigin !== initialOrigin) {
          crossedOrigin = true;
        }
        if (crossedOrigin) {
          stripCredentialHeaders(headers);
        }
        if (
          response.status === 301 ||
          response.status === 302 ||
          response.status === 303
        ) {
          currentMethod = "GET";
          currentBody = undefined;
          delete headers["Content-Type"];
        }
        currentUrl = resolvedNext;
        continue;
      }

      const textBody = await response.text();
      if (!response.ok) {
        throw new MDMHTTPError(response.status, textBody);
      }
      let parsedJson: unknown;
      try {
        parsedJson = textBody ? JSON.parse(textBody) : {};
      } catch (err) {
        throw new MDMTransportError(`invalid JSON response: ${String(err)}`, false);
      }
      if (!parsedJson || typeof parsedJson !== "object" || Array.isArray(parsedJson)) {
        throw new MDMValidationError("expected JSON object response");
      }
      return parsedJson as Record<string, unknown>;
    }

    throw new MDMTransportError("redirect loop exhausted", false);
  }

  async checkHealth(): Promise<Record<string, unknown>> {
    if (isRawPredictEndpoint(this.baseUrl)) {
      return this.requestJson("POST", this.baseUrl, { health: true });
    }
    const endpoint = this.baseUrl.endsWith("/healthz")
      ? this.baseUrl
      : `${this.baseUrl}/healthz`;
    return this.requestJson("GET", endpoint);
  }

  async decideSystemOne(request: SystemOneDecideRequest): Promise<DecisionResponse> {
    const startedAt = performance.now();
    const questionEntries = Object.entries(request.questions);
    if (questionEntries.length === 0) {
      throw new MDMValidationError("questions must contain at least one slot");
    }
    const wireQuestions: Record<string, unknown> = {};
    for (const [slotId, question] of questionEntries) {
      const optionKeys = getQuestionOptionKeys(question);
      if (optionKeys.length < 2) {
        throw new MDMValidationError(`slot ${JSON.stringify(slotId)} requires >= 2 options`);
      }
      wireQuestions[slotId] = {
        type: question.type,
        instructions: question.instructions,
        ...(question.criteria ? { criteria: question.criteria } : {}),
        ...(question.levels ? { levels: [...question.levels] } : {}),
        ...(question.order ? { order: [...question.order] } : {}),
      };
    }

    const endpoint =
      isRawPredictEndpoint(this.baseUrl) || this.baseUrl.endsWith("/v1/systemone")
        ? this.baseUrl
        : `${this.baseUrl}/v1/systemone`;

    const payload: Record<string, unknown> = {
      state: request.state,
      questions: wireQuestions,
      ...(request.images && request.images.length > 0 ? { images: [...request.images] } : {}),
    };
    const responseJson = await this.requestJson("POST", endpoint, payload);
    const unwrapped =
      Array.isArray(responseJson["predictions"]) &&
      responseJson["predictions"][0] &&
      typeof responseJson["predictions"][0] === "object"
        ? (responseJson["predictions"][0] as Record<string, unknown>)
        : responseJson;
    const rawAnswers = (unwrapped["answers"] ?? unwrapped["slots"] ?? unwrapped) as Record<
      string,
      unknown
    >;

    const temperature = request.temperature ?? 1.0;
    const slots: Record<string, SlotReadout> = {};
    for (const [slotId, question] of questionEntries) {
      const entry = rawAnswers[slotId];
      if (!entry || typeof entry !== "object") {
        throw new MDMValidationError(
          `upstream response missing slot ${JSON.stringify(slotId)}`,
        );
      }
      const entryRecord = entry as Record<string, unknown>;
      const rawDist = (entryRecord["probabilities"] ?? entryRecord["scores"] ?? entryRecord) as Record<
        string,
        unknown
      >;
      const optionKeys = getQuestionOptionKeys(question);
      const rawProbs: Record<string, number> = {};
      for (const optionKey of optionKeys) {
        const val = rawDist[optionKey];
        if (typeof val !== "number" || !Number.isFinite(val) || val < 0.0) {
          throw new MDMValidationError(
            `slot ${JSON.stringify(slotId)} missing valid probability for ${JSON.stringify(optionKey)}`,
          );
        }
        rawProbs[optionKey] = val;
      }
      const scaledProbs = applyTemperature(rawProbs, temperature);
      const hRaw = normalizedEntropy(rawProbs);
      let bestKey = optionKeys[0] ?? "";
      let bestProb = -1.0;
      for (const [optKey, optProb] of Object.entries(scaledProbs)) {
        if (optProb > bestProb) {
          bestProb = optProb;
          bestKey = optKey;
        }
      }
      slots[slotId] = {
        slotId,
        predicted: bestKey,
        probabilities: scaledProbs,
        rawProbabilities: applyTemperature(rawProbs, 1.0),
        hRaw,
        confidence: bestProb,
      };
    }

    return {
      slots,
      latencyMs: performance.now() - startedAt,
      backendName: isRawPredictEndpoint(this.baseUrl)
        ? "vertex_raw_predict"
        : "dgem_systemone",
    };
  }

  async decideChatLogprobs(request: ChatLogprobDecideRequest): Promise<DecisionResponse> {
    const startedAt = performance.now();
    const questionEntries = Object.entries(request.questions);
    if (questionEntries.length === 0) {
      throw new MDMValidationError("questions must contain at least one slot");
    }
    const endpoint = this.baseUrl.endsWith("/v1/chat/completions")
      ? this.baseUrl
      : `${this.baseUrl}/v1/chat/completions`;
    const topLogprobs = request.topLogprobs ?? 20;
    const temperature = request.temperature ?? 1.0;
    const stateText =
      typeof request.state === "string" ? request.state : JSON.stringify(request.state);

    const slots: Record<string, SlotReadout> = {};
    for (const [slotId, question] of questionEntries) {
      const optionKeys = getQuestionOptionKeys(question);
      if (optionKeys.length < 2 || optionKeys.length > ASCII_LETTERS.length) {
        throw new MDMValidationError(
          `decideChatLogprobs supports 2..26 options per slot (got ${optionKeys.length})`,
        );
      }
      const codeMap: Record<string, string> = {};
      const optionLines: string[] = [];
      for (let idx = 0; idx < optionKeys.length; idx += 1) {
        const letter = ASCII_LETTERS[idx] ?? "A";
        const key = optionKeys[idx] ?? "";
        codeMap[letter] = key;
        const desc = question.criteria?.[key] ?? key;
        optionLines.push(`${letter}: ${key} - ${desc}`);
      }
      const prompt = [
        `State:\n${stateText}`,
        `Question (${slotId}): ${question.instructions}`,
        "Options:",
        ...optionLines,
        "Reply with the single uppercase option letter only.",
      ].join("\n");

      const responseJson = await this.requestJson("POST", endpoint, {
        model: request.model,
        messages: [{ role: "user", content: prompt }],
        max_tokens: 1,
        temperature: 0.0,
        logprobs: true,
        top_logprobs: topLogprobs,
      });

      const choices = responseJson["choices"];
      const firstChoice =
        Array.isArray(choices) && choices[0] && typeof choices[0] === "object"
          ? (choices[0] as Record<string, unknown>)
          : undefined;
      const logprobsObj =
        firstChoice?.["logprobs"] && typeof firstChoice["logprobs"] === "object"
          ? (firstChoice["logprobs"] as Record<string, unknown>)
          : undefined;
      const contentArr = logprobsObj?.["content"];
      const firstTokenEntry =
        Array.isArray(contentArr) && contentArr[0] && typeof contentArr[0] === "object"
          ? (contentArr[0] as Record<string, unknown>)
          : undefined;
      const topCandidates = firstTokenEntry?.["top_logprobs"];
      if (!Array.isArray(topCandidates) || topCandidates.length === 0) {
        throw new MDMValidationError(
          `chat/completions response missing top_logprobs for slot ${JSON.stringify(slotId)}`,
        );
      }

      const letterLogprobs: Record<string, number> = {};
      for (const cand of topCandidates) {
        if (!cand || typeof cand !== "object") {
          continue;
        }
        const candObj = cand as Record<string, unknown>;
        const rawTok = typeof candObj["token"] === "string" ? candObj["token"].trim() : "";
        const lp = candObj["logprob"];
        if (
          rawTok in codeMap &&
          typeof lp === "number" &&
          Number.isFinite(lp) &&
          !(rawTok in letterLogprobs)
        ) {
          letterLogprobs[rawTok] = lp;
        }
      }
      if (Object.keys(letterLogprobs).length === 0) {
        throw new MDMValidationError(
          `no aligned single-token [A..Z] logprobs found for slot ${JSON.stringify(slotId)}`,
        );
      }

      let maxLp = -Infinity;
      for (const val of Object.values(letterLogprobs)) {
        if (val > maxLp) {
          maxLp = val;
        }
      }
      const rawProbs: Record<string, number> = {};
      let sumExp = 0.0;
      for (const [letter, optKey] of Object.entries(codeMap)) {
        const lp = letterLogprobs[letter] ?? maxLp - 20.0;
        const mass = Math.exp(lp - maxLp);
        rawProbs[optKey] = mass;
        sumExp += mass;
      }
      for (const optKey of Object.keys(rawProbs)) {
        rawProbs[optKey] = (rawProbs[optKey] ?? 0.0) / sumExp;
      }
      const scaledProbs = applyTemperature(rawProbs, temperature);
      const hRaw = normalizedEntropy(rawProbs);
      let bestKey = optionKeys[0] ?? "";
      let bestProb = -1.0;
      for (const [optKey, optProb] of Object.entries(scaledProbs)) {
        if (optProb > bestProb) {
          bestProb = optProb;
          bestKey = optKey;
        }
      }
      slots[slotId] = {
        slotId,
        predicted: bestKey,
        probabilities: scaledProbs,
        rawProbabilities: rawProbs,
        hRaw,
        confidence: bestProb,
      };
    }

    return {
      slots,
      latencyMs: performance.now() - startedAt,
      backendName: "openai_logprobs",
    };
  }
}
