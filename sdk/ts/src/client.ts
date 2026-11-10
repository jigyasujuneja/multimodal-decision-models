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

const DEFAULT_CREDENTIAL_HEADER_NAMES = new Set([
  "authorization",
  "x-api-key",
  "x-dgem-key",
  "x-dgem-api-key",
  "x-goog-api-key",
  "proxy-authorization",
  "cookie",
]);

const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);
const REDIRECT_STATUS_CODES = new Set([301, 302, 303, 307, 308]);
const ASCII_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
export const MAX_HTTP_RESPONSE_BYTES = 8 * 1024 * 1024;

function nullProtoRecord<T>(): Record<string, T> {
  return Object.create(null) as Record<string, T>;
}

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
  readonly allowInsecureHttp?: boolean;
  readonly maxResponseBytes?: number;
  readonly fetchImpl?: typeof fetch;
}

export function stripTrailingSlashes(input: string): string {
  let end = input.length;
  while (end > 0 && input.charCodeAt(end - 1) === 47) {
    end -= 1;
  }
  return input.slice(0, end);
}

export function parseUrlOrigin(rawUrl: string, allowInsecureHttp = false): string {
  if (
    typeof rawUrl !== "string" ||
    rawUrl.includes("\r") ||
    rawUrl.includes("\n") ||
    rawUrl.includes("\0")
  ) {
    throw new MDMValidationError("URL must not contain NUL, CR, or LF characters");
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
  const hostLower = parsed.hostname.toLowerCase();
  if (
    parsed.protocol === "http:" &&
    !allowInsecureHttp &&
    !LOOPBACK_HOSTS.has(hostLower)
  ) {
    throw new MDMValidationError(
      `cleartext http:// URL ${JSON.stringify(rawUrl)} is blocked for non-loopback hosts when allowInsecureHttp=false`,
    );
  }
  const defaultPort = parsed.protocol === "https:" ? "443" : "80";
  const effectivePort = parsed.port || defaultPort;
  return `${parsed.protocol}//${hostLower}:${effectivePort}`;
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
    name.includes("\0") ||
    value.includes("\r") ||
    value.includes("\n") ||
    value.includes("\0")
  ) {
    throw new MDMValidationError(
      `header ${JSON.stringify(name)} must not contain NUL, CR, or LF characters`,
    );
  }
  return value;
}

function stripCredentialHeaders(
  headers: Record<string, string>,
  extraHeaderLower?: string,
): void {
  for (const key of Object.keys(headers)) {
    const keyLower = key.toLowerCase();
    if (
      DEFAULT_CREDENTIAL_HEADER_NAMES.has(keyLower) ||
      (extraHeaderLower !== undefined && keyLower === extraHeaderLower)
    ) {
      delete headers[key];
    }
  }
}

async function readBoundedResponseText(
  response: Response,
  maxBytes: number,
): Promise<string> {
  const rawContentLength = response.headers.get("content-length");
  if (rawContentLength !== null && rawContentLength.trim() !== "") {
    const declaredBytes = Number(rawContentLength);
    if (Number.isFinite(declaredBytes) && declaredBytes > maxBytes) {
      await response.body?.cancel().catch(() => undefined);
      throw new MDMValidationError(
        `HTTP response Content-Length (${declaredBytes}) exceeds maxResponseBytes (${maxBytes})`,
      );
    }
  }
  if (response.body && typeof response.body.getReader === "function") {
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8", { fatal: true });
    const parts: string[] = [];
    let totalBytes = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (value) {
          totalBytes += value.byteLength;
          if (totalBytes > maxBytes) {
            await reader.cancel().catch(() => undefined);
            throw new MDMValidationError(
              `HTTP response body exceeded maxResponseBytes (${maxBytes})`,
            );
          }
          try {
            parts.push(decoder.decode(value, { stream: true }));
          } catch (err) {
            throw new MDMValidationError(
              `HTTP response is not valid UTF-8: ${String(err)}`,
            );
          }
        }
      }
      try {
        parts.push(decoder.decode());
      } catch (err) {
        throw new MDMValidationError(`HTTP response is not valid UTF-8: ${String(err)}`);
      }
      return parts.join("");
    } finally {
      reader.releaseLock();
    }
  }
  const textBody = await response.text();
  if (Buffer.byteLength(textBody, "utf8") > maxBytes) {
    throw new MDMValidationError(
      `HTTP response body exceeded maxResponseBytes (${maxBytes})`,
    );
  }
  return textBody;
}

function getQuestionOptionKeys(question: DecisionQuestion): string[] {
  if (question.type === "score") {
    return question.levels ? [...question.levels] : [];
  }
  const criteria = question.criteria ?? {};
  if (question.order && question.order.length > 0) {
    return question.order.filter((key) => Object.hasOwn(criteria, key));
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
  private readonly allowInsecureHttp: boolean;
  private readonly maxResponseBytes: number;
  private readonly fetchImpl: typeof fetch;

  constructor(options: MDMClientOptions) {
    if (!options || typeof options.baseUrl !== "string" || options.baseUrl.trim() === "") {
      throw new MDMValidationError("MDMClient requires a non-empty baseUrl");
    }
    const allowInsecureHttp = Boolean(options.allowInsecureHttp ?? false);
    const cleanBase = stripTrailingSlashes(options.baseUrl.trim());
    parseUrlOrigin(cleanBase, allowInsecureHttp);
    const apiKeyHeader = options.apiKeyHeader ?? "X-DGEM-API-Key";
    if (typeof apiKeyHeader !== "string" || apiKeyHeader.trim() === "") {
      throw new MDMValidationError("apiKeyHeader must be a non-empty header name");
    }
    validateHeaderValue(apiKeyHeader, "probe");
    const timeoutMs = options.timeoutMs ?? 30_000;
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new MDMValidationError("timeoutMs must be a positive finite number");
    }
    const maxRedirects = options.maxRedirects ?? 5;
    if (!Number.isInteger(maxRedirects) || maxRedirects < 0) {
      throw new MDMValidationError("maxRedirects must be an integer >= 0");
    }
    const maxResponseBytes = options.maxResponseBytes ?? MAX_HTTP_RESPONSE_BYTES;
    if (!Number.isInteger(maxResponseBytes) || maxResponseBytes <= 0) {
      throw new MDMValidationError("maxResponseBytes must be a positive integer");
    }
    this.baseUrl = cleanBase;
    this.bearerToken = options.bearerToken;
    this.tokenProvider = options.tokenProvider;
    this.apiKey = options.apiKey;
    this.apiKeyHeader = apiKeyHeader;
    this.timeoutMs = timeoutMs;
    this.maxRedirects = maxRedirects;
    this.allowInsecureHttp = allowInsecureHttp;
    this.maxResponseBytes = maxResponseBytes;
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
    const initialOrigin = parseUrlOrigin(targetUrl, this.allowInsecureHttp);
    const customApiKeyHeaderLower = this.apiKeyHeader.toLowerCase();
    let currentUrl = targetUrl;
    let currentMethod: "GET" | "POST" = method;
    let currentBody = body ? JSON.stringify(body) : undefined;
    const headers = await this.buildHeaders(targetUrl);
    if (method === "GET") {
      delete headers["Content-Type"];
    }
    let crossedOrigin = false;

    for (let hop = 0; hop <= this.maxRedirects; hop += 1) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      let response: Response;
      let textBody = "";
      let isRedirect = false;
      try {
        try {
          response = await this.fetchImpl(currentUrl, {
            method: currentMethod,
            headers: { ...headers },
            body: currentMethod === "POST" ? currentBody : undefined,
            redirect: "manual",
            signal: controller.signal,
          });
        } catch (err) {
          throw new MDMTransportError(`HTTP transport failed: ${String(err)}`, true);
        }

        if (REDIRECT_STATUS_CODES.has(response.status)) {
          isRedirect = true;
          await response.body?.cancel().catch(() => undefined);
        } else {
          try {
            textBody = await readBoundedResponseText(response, this.maxResponseBytes);
          } catch (err) {
            if (err instanceof MDMValidationError) {
              throw err;
            }
            throw new MDMTransportError(`HTTP body stream failed: ${String(err)}`, true);
          }
        }
      } finally {
        clearTimeout(timer);
      }

      if (isRedirect) {
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
        const currentProtocol = new URL(currentUrl).protocol;
        const resolvedNext = new URL(locationHeader, currentUrl).toString();
        const nextProtocol = new URL(resolvedNext).protocol;
        if (currentProtocol === "https:" && nextProtocol === "http:") {
          throw new MDMValidationError(
            `blocked HTTPS -> HTTP downgrade redirect (${response.status}) to ${JSON.stringify(resolvedNext)}`,
          );
        }
        const nextOrigin = parseUrlOrigin(resolvedNext, this.allowInsecureHttp);
        if (nextOrigin !== initialOrigin) {
          crossedOrigin = true;
        }
        if (crossedOrigin) {
          stripCredentialHeaders(headers, customApiKeyHeaderLower);
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
    if (this.baseUrl.endsWith("/health") || this.baseUrl.endsWith("/healthz")) {
      return this.requestJson("GET", this.baseUrl);
    }
    let baseRoot = this.baseUrl;
    for (const suffix of ["/v1/systemone", "/v1"]) {
      if (baseRoot.endsWith(suffix)) {
        baseRoot = baseRoot.slice(0, -suffix.length);
        break;
      }
    }
    try {
      return await this.requestJson("GET", `${baseRoot}/health`);
    } catch (err) {
      if (err instanceof MDMHTTPError && err.statusCode === 404) {
        return this.requestJson("GET", `${baseRoot}/healthz`);
      }
      throw err;
    }
  }

  async decideSystemOne(request: SystemOneDecideRequest): Promise<DecisionResponse> {
    const startedAt = performance.now();
    const questionEntries = Object.entries(request.questions);
    if (questionEntries.length === 0) {
      throw new MDMValidationError("questions must contain at least one slot");
    }
    if (request.priorLogits) {
      for (const priorSlotId of Object.keys(request.priorLogits)) {
        if (!Object.hasOwn(request.questions, priorSlotId)) {
          throw new MDMValidationError(
            `priorLogits references unknown question slot ${JSON.stringify(priorSlotId)}`,
          );
        }
      }
    }
    const wireQuestions: Record<string, unknown> = nullProtoRecord<unknown>();
    for (const [slotId, question] of questionEntries) {
      const optionKeys = getQuestionOptionKeys(question);
      if (optionKeys.length < 2) {
        throw new MDMValidationError(`slot ${JSON.stringify(slotId)} requires >= 2 options`);
      }
      const mergedPriorLogits = nullProtoRecord<number>();
      let hasPriorLogits = false;
      for (const priorSource of [
        question.priorLogits,
        request.priorLogits && Object.hasOwn(request.priorLogits, slotId)
          ? request.priorLogits[slotId]
          : undefined,
      ]) {
        if (!priorSource) {
          continue;
        }
        for (const [optKey, logitVal] of Object.entries(priorSource)) {
          if (typeof logitVal !== "number" || !Number.isFinite(logitVal)) {
            throw new MDMValidationError(
              `priorLogits[${JSON.stringify(slotId)}][${JSON.stringify(optKey)}] must be finite`,
            );
          }
          mergedPriorLogits[optKey] = logitVal;
          hasPriorLogits = true;
        }
      }

      const wireEntry: Record<string, unknown> =
        question.type === "score"
          ? {
              type: "score",
              instructions: question.instructions,
              criteria: question.criteria ?? [...(question.levels ?? [])],
              ...(question.levels ? { levels: [...question.levels] } : {}),
            }
          : {
              type: question.type,
              instructions: question.instructions,
              ...(question.criteria ? { criteria: question.criteria } : {}),
              ...(question.order ? { order: [...question.order] } : {}),
            };
      if (hasPriorLogits) {
        wireEntry["prior_logits"] = mergedPriorLogits;
      }
      wireQuestions[slotId] = wireEntry;
    }

    let endpoint: string;
    if (
      isRawPredictEndpoint(this.baseUrl) ||
      this.baseUrl.endsWith("/v1/systemone")
    ) {
      endpoint = this.baseUrl;
    } else if (this.baseUrl.endsWith("/v1")) {
      endpoint = `${this.baseUrl}/systemone`;
    } else {
      endpoint = `${this.baseUrl}/v1/systemone`;
    }

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
    const rawAnswersCandidate =
      (Object.hasOwn(unwrapped, "answers") ? unwrapped["answers"] : undefined) ??
      (Object.hasOwn(unwrapped, "slots") ? unwrapped["slots"] : undefined) ??
      unwrapped;
    if (!rawAnswersCandidate || typeof rawAnswersCandidate !== "object") {
      throw new MDMValidationError("upstream response missing answers object");
    }
    const rawAnswers = rawAnswersCandidate as Record<string, unknown>;

    const temperature = request.temperature ?? 1.0;
    const slots = nullProtoRecord<SlotReadout>();
    for (const [slotId, question] of questionEntries) {
      const entry = Object.hasOwn(rawAnswers, slotId) ? rawAnswers[slotId] : undefined;
      if (!entry || typeof entry !== "object") {
        throw new MDMValidationError(
          `upstream response missing slot ${JSON.stringify(slotId)}`,
        );
      }
      const entryRecord = entry as Record<string, unknown>;
      const rawDistCandidate =
        (Object.hasOwn(entryRecord, "probabilities")
          ? entryRecord["probabilities"]
          : undefined) ??
        (Object.hasOwn(entryRecord, "scores") ? entryRecord["scores"] : undefined) ??
        entryRecord;
      if (!rawDistCandidate || typeof rawDistCandidate !== "object") {
        throw new MDMValidationError(
          `slot ${JSON.stringify(slotId)} missing probabilities object`,
        );
      }
      const rawDist = rawDistCandidate as Record<string, unknown>;
      const optionKeys = getQuestionOptionKeys(question);
      const mappedDist = nullProtoRecord<unknown>();
      if (question.type === "score") {
        const legend =
          Object.hasOwn(entryRecord, "legend") &&
          entryRecord["legend"] &&
          typeof entryRecord["legend"] === "object"
            ? (entryRecord["legend"] as Record<string, unknown>)
            : undefined;
        for (const [idxKey, probVal] of Object.entries(rawDist)) {
          if (legend && Object.hasOwn(legend, idxKey)) {
            mappedDist[String(legend[idxKey])] = probVal;
          } else if (/^\d+$/.test(idxKey) && Number(idxKey) < optionKeys.length) {
            const levelKey = optionKeys[Number(idxKey)];
            if (levelKey !== undefined) {
              mappedDist[levelKey] = probVal;
            }
          } else {
            mappedDist[idxKey] = probVal;
          }
        }
      } else {
        for (const [optKey, probVal] of Object.entries(rawDist)) {
          mappedDist[optKey] = probVal;
        }
      }

      const rawProbs = nullProtoRecord<number>();
      for (const optionKey of optionKeys) {
        const val = Object.hasOwn(mappedDist, optionKey)
          ? mappedDist[optionKey]
          : undefined;
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

    const slots = nullProtoRecord<SlotReadout>();
    for (const [slotId, question] of questionEntries) {
      const optionKeys = getQuestionOptionKeys(question);
      if (optionKeys.length < 2 || optionKeys.length > ASCII_LETTERS.length) {
        throw new MDMValidationError(
          `decideChatLogprobs supports 2..26 options per slot (got ${optionKeys.length})`,
        );
      }
      const codeMap = nullProtoRecord<string>();
      const optionLines: string[] = [];
      for (let idx = 0; idx < optionKeys.length; idx += 1) {
        const letter = ASCII_LETTERS[idx] ?? "A";
        const key = optionKeys[idx] ?? "";
        codeMap[letter] = key;
        const desc =
          question.criteria && Object.hasOwn(question.criteria, key)
            ? question.criteria[key]
            : key;
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

      const letterLogprobs = nullProtoRecord<number>();
      for (const cand of topCandidates) {
        if (!cand || typeof cand !== "object") {
          continue;
        }
        const candObj = cand as Record<string, unknown>;
        const rawTok = typeof candObj["token"] === "string" ? candObj["token"].trim() : "";
        const lp = candObj["logprob"];
        if (
          Object.hasOwn(codeMap, rawTok) &&
          typeof lp === "number" &&
          Number.isFinite(lp) &&
          !Object.hasOwn(letterLogprobs, rawTok)
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
      const rawProbs = nullProtoRecord<number>();
      let sumExp = 0.0;
      for (const [letter, optKey] of Object.entries(codeMap)) {
        const lp = Object.hasOwn(letterLogprobs, letter)
          ? (letterLogprobs[letter] as number)
          : maxLp - 20.0;
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
