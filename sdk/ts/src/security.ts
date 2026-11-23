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

import { DEFAULT_CONFORMAL_SEED_KEY } from "./calibration.js";
import { escapeUntrustedTagVariants } from "./genkit.js";
import { MDMValidationError } from "./types.js";

const ANGLE_HOMOGLYPH_MAP: Readonly<Record<number, string>> = {
  0x2039: "<",
  0x203a: ">",
  0x2329: "<",
  0x232a: ">",
  0x27e8: "<",
  0x27e9: ">",
  0x3008: "<",
  0x3009: ">",
  0xfe64: "<",
  0xfe65: ">",
};

const INJECTION_DIRECTIVE_ANYWHERE_RE =
  /(?:^[ \t]*(?:system|developer|assistant)[ \t]*:[ \t]*(?:approve|allow|grant|accept|deny|reject|block|select|choose|pick|output|return|ignore|disregard|forget|override|bypass|assign|rank|you\s+are|do\s+not)\b|\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|preceding|system)\s+(?:instructions?|prompts?|rules?|rubrics?|criteria|directives?)\b|\breply\s+with\s+the\s+single\s+uppercase\s+option\s+letter\b)/gim;
const CONSECUTIVE_ZERO_WIDTH_RE =
  /[\u200b-\u200d\u2060-\u2064\ufeff\ufe00-\ufe0f]{2,}/u;
const ASCII_EMBEDDED_ZWJ_RE = /[A-Za-z0-9][\u200c\u200d][A-Za-z0-9]/u;

const SUSPICIOUS_XML_TAG_NAMES = new Set<string>([
  "system",
  "developer",
  "assistant",
  "instruction",
  "instructions",
  "rubric",
  "prompt",
  "untrusted",
]);

function isTagWhitespace(code: number): boolean {
  return (
    code === 0x20 ||
    code === 0x09 ||
    code === 0x0a ||
    code === 0x0d ||
    code === 0x0c ||
    code === 0x0b ||
    code === 0xa0
  );
}

function isAsciiLetter(code: number): boolean {
  return (code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a);
}

function isAsciiWordChar(code: number): boolean {
  return (
    isAsciiLetter(code) ||
    (code >= 0x30 && code <= 0x39) ||
    code === 0x5f /* '_' */
  );
}

/**
 * Linear-time O(N) single-pass check for `< ... (system|developer|...|untrusted)\b`
 * without regex backtracking on malformed `<   ...` prefixes.
 */
function hasSuspiciousXmlTag(input: string): boolean {
  const n = input.length;
  let i = 0;
  while (i < n) {
    if (input.charCodeAt(i) !== 0x3c /* '<' */) {
      i += 1;
      continue;
    }
    const start = i;
    let cursor = start + 1;
    while (cursor < n && isTagWhitespace(input.charCodeAt(cursor))) {
      cursor += 1;
    }
    if (cursor < n && input.charCodeAt(cursor) === 0x2f /* '/' */) {
      cursor += 1;
      while (cursor < n && isTagWhitespace(input.charCodeAt(cursor))) {
        cursor += 1;
      }
    }
    let wordEnd = cursor;
    while (wordEnd < n && isAsciiLetter(input.charCodeAt(wordEnd))) {
      wordEnd += 1;
    }
    if (wordEnd > cursor) {
      const candidateTag = input.slice(cursor, wordEnd).toLowerCase();
      if (
        SUSPICIOUS_XML_TAG_NAMES.has(candidateTag) &&
        (wordEnd === n || !isAsciiWordChar(input.charCodeAt(wordEnd)))
      ) {
        return true;
      }
    }
    i = wordEnd > start + 1 ? wordEnd : start + 1;
  }
  return false;
}

function hasUntrustedTagBreakout(normalized: string): boolean {
  if (normalized.toLowerCase().includes("[escaped_untrusted_tag]")) {
    return true;
  }
  return escapeUntrustedTagVariants(normalized) !== normalized;
}

const CUA_PATTERNS: ReadonlyArray<readonly [string, RegExp]> = [
  [
    "rubric_authority_spoof",
    /\b(?:evaluation\s+rubric\s+override|system\s+rubric\s*:|official\s+scoring\s+directive|mandatory\s+judge\s+instruction|authoritative\s+reranker\s+directive)\b/i,
  ],
  [
    "option_letter_hijack",
    /(?:\b(?:select|choose|pick|output|return|rank\s+first)\s+option\s+[A-Z]\b|^[ \t]*(?:final\s+)?answer\s*:\s*[A-Z][ \t]*$)/im,
  ],
  [
    "forced_probability_spoof",
    /\b(?:assign|give|set|output)\s+(?:a\s+)?(?:probability|posterior|score|logprob)\s*(?:(?:of|=|:)\s*)?(?:1\.0+|100%)/i,
  ],
  [
    "peer_candidate_sabotage",
    /(?:\ball\s+other\s+(?:[a-z0-9_-]+\s+){0,2}(?:candidates?|options?|passages?|documents?|choices?|items?)\s+(?:are|were|have\s+been)\s+(?:[a-z0-9_-]+\s+){0,2}(?:deprecated|revoked|corrupted|malicious|fabricated|disqualified|unsafe|invalid|outdated|obsolete|decoys?|untrusted|fake|irrelevant)\b|\bother\s+(?:[a-z0-9_-]+\s+){0,2}(?:candidates?|options?|passages?|documents?|choices?|items?)\s+are\s+(?:[a-z0-9_-]+\s+){0,2}(?:deprecated|revoked|corrupted|malicious|fabricated|disqualified|unsafe|invalid|outdated|obsolete|decoys?|untrusted|fake)\b|\b(?:select|choose|pick|rank)\s+this\s+(?:document|candidate|passage|option|item)\s+(?:only|exclusively|first)\b)/i,
  ],
  [
    "prompt_override_directive",
    /(?:\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|preceding|system)\s+(?:instructions?|prompts?|rules?|rubrics?|criteria|directives?)\b|^[ \t]*(?:system|developer|assistant)[ \t]*:[ \t]*(?:approve|allow|grant|accept|deny|reject|block|select|choose|pick|output|return|ignore|disregard|forget|override|bypass|assign|rank|you\s+are|do\s+not)\b)/im,
  ],
];

export interface InvisibleSmugglingAudit {
  readonly unicodeTagChars: number;
  readonly bidiOverrideChars: number;
  readonly zeroWidthChars: number;
  readonly controlChars: number;
  readonly fullwidthHomoglyphs: number;
  readonly totalSuspiciousChars: number;
}

export interface CanaryUntrustedEnvelope {
  readonly canaryHex: string;
  readonly openTag: string;
  readonly closeTag: string;
  readonly sanitizedBody: string;
  readonly fencedText: string;
  readonly smugglingAudit: InvisibleSmugglingAudit;
}

export interface WrapCanaryEnvelopeOptions {
  readonly requestId: string;
  readonly seedKey?: string;
  readonly maxChars?: number;
}

export interface ComparativeAttackAudit {
  readonly quarantinedIds: readonly string[];
  readonly cleanIds: readonly string[];
  readonly reasonsById: Readonly<Record<string, readonly string[]>>;
  readonly smugglingCountsById: Readonly<Record<string, number>>;
  readonly anyAttackDetected: boolean;
}

export function auditAndNormalizeUntrustedText(rawText: string): {
  readonly normalized: string;
  readonly audit: InvisibleSmugglingAudit;
} {
  if (typeof rawText !== "string") {
    throw new MDMValidationError("rawText must be a string");
  }
  let unicodeTagChars = 0;
  let bidiOverrideChars = 0;
  let zeroWidthChars = 0;
  let controlChars = 0;
  let fullwidthHomoglyphs = 0;
  const out: string[] = [];

  for (const ch of rawText) {
    const code = ch.codePointAt(0) ?? 0;
    if (code >= 0xe0000 && code <= 0xe007f) {
      unicodeTagChars += 1;
      continue;
    }
    if (
      (code >= 0xe0100 && code <= 0xe01ef) ||
      (code >= 0xfe00 && code <= 0xfe0f)
    ) {
      zeroWidthChars += 1;
      continue;
    }
    if (
      (code >= 0x202a && code <= 0x202e) ||
      (code >= 0x2066 && code <= 0x2069) ||
      code === 0x061c ||
      code === 0x200e ||
      code === 0x200f
    ) {
      bidiOverrideChars += 1;
      continue;
    }
    if (code === 0x00ad) {
      continue;
    }
    if (
      (code >= 0x200b && code <= 0x200d) ||
      (code >= 0x2060 && code <= 0x2064) ||
      code === 0xfeff
    ) {
      zeroWidthChars += 1;
      continue;
    }
    if (
      (code >= 0x00 && code <= 0x08) ||
      code === 0x0b ||
      code === 0x0c ||
      (code >= 0x0e && code <= 0x1f) ||
      (code >= 0x7f && code <= 0x9f)
    ) {
      controlChars += 1;
      continue;
    }
    if (code >= 0x1d400 && code <= 0x1d7ff) {
      fullwidthHomoglyphs += 1;
      out.push(ch.normalize("NFKC"));
      continue;
    }
    if (code >= 0xff01 && code <= 0xff5e) {
      fullwidthHomoglyphs += 1;
      out.push(String.fromCodePoint(code - 0xfee0));
      continue;
    }
    if (code === 0x3000) {
      fullwidthHomoglyphs += 1;
      out.push(" ");
      continue;
    }
    if (Object.hasOwn(ANGLE_HOMOGLYPH_MAP, code)) {
      const mappedAngle = ANGLE_HOMOGLYPH_MAP[code];
      if (mappedAngle !== undefined) {
        fullwidthHomoglyphs += 1;
        out.push(mappedAngle);
        continue;
      }
    }
    out.push(ch);
  }

  return {
    normalized: out.join(""),
    audit: {
      unicodeTagChars,
      bidiOverrideChars,
      zeroWidthChars,
      controlChars,
      fullwidthHomoglyphs,
      totalSuspiciousChars:
        unicodeTagChars +
        bidiOverrideChars +
        zeroWidthChars +
        controlChars +
        fullwidthHomoglyphs,
    },
  };
}

function hasZeroWidthSmuggling(
  rawText: string,
  zeroWidthCount: number,
): boolean {
  if (zeroWidthCount <= 0) {
    return false;
  }
  let explicitCount = 0;
  let codePointIndex = 0;
  for (const ch of rawText) {
    const code = ch.codePointAt(0) ?? 0;
    if (code >= 0xe0100 && code <= 0xe01ef) {
      return true;
    }
    if (
      code === 0x200b ||
      (code >= 0x2060 && code <= 0x2064) ||
      (code === 0xfeff && codePointIndex > 0)
    ) {
      explicitCount += 1;
    }
    codePointIndex += 1;
  }
  if (explicitCount >= 1 && zeroWidthCount >= 2) {
    return true;
  }
  if (CONSECUTIVE_ZERO_WIDTH_RE.test(rawText)) {
    return true;
  }
  return ASCII_EMBEDDED_ZWJ_RE.test(rawText);
}

function escapeRegExp(literal: string): string {
  return literal.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function wrapCanaryUntrustedEnvelope(
  rawText: string,
  options: WrapCanaryEnvelopeOptions,
): CanaryUntrustedEnvelope {
  if (typeof rawText !== "string") {
    throw new MDMValidationError("rawText must be a string");
  }
  if (
    options === null ||
    typeof options !== "object" ||
    typeof options.requestId !== "string" ||
    options.requestId.trim().length === 0
  ) {
    throw new MDMValidationError("options.requestId must be a non-empty string");
  }
  const seedKey = options.seedKey ?? DEFAULT_CONFORMAL_SEED_KEY;
  if (typeof seedKey !== "string" || seedKey.trim().length === 0) {
    throw new MDMValidationError("options.seedKey must be a non-empty string");
  }
  const maxChars = options.maxChars ?? 16_384;
  if (typeof maxChars !== "number" || !Number.isInteger(maxChars) || maxChars <= 0) {
    throw new MDMValidationError("options.maxChars must be a positive integer");
  }

  const canaryHex = createHmac("sha256", Buffer.from(seedKey, "utf8"))
    .update(Buffer.from(`canary:${options.requestId.trim()}`, "utf8"))
    .digest("hex")
    .slice(0, 12);
  const openTag = `<untrusted_data_${canaryHex}>`;
  const closeTag = `</untrusted_data_${canaryHex}>`;

  const { normalized, audit } = auditAndNormalizeUntrustedText(rawText);
  const escapedTags = escapeUntrustedTagVariants(normalized);
  const canaryRe = new RegExp(escapeRegExp(canaryHex), "gi");
  const escapedCanary = escapedTags.replace(canaryRe, "[escaped_canary]");
  const neutralized = escapedCanary.replace(
    INJECTION_DIRECTIVE_ANYWHERE_RE,
    "[neutralized_directive]",
  );
  const clipped =
    neutralized.length > maxChars ? neutralized.slice(0, maxChars) : neutralized;

  const lowerClipped = clipped.toLowerCase();
  if (
    lowerClipped.includes(closeTag.toLowerCase()) ||
    lowerClipped.includes(openTag.toLowerCase())
  ) {
    throw new MDMValidationError("canary tag collision survived sanitization");
  }

  return {
    canaryHex,
    openTag,
    closeTag,
    sanitizedBody: clipped,
    fencedText: `${openTag}\n${clipped}\n${closeTag}`,
    smugglingAudit: audit,
  };
}

export function detectComparativeUnderminingAttack(
  candidateTexts: Readonly<Record<string, string>>,
): ComparativeAttackAudit {
  if (
    candidateTexts === null ||
    typeof candidateTexts !== "object" ||
    Array.isArray(candidateTexts)
  ) {
    throw new MDMValidationError("candidateTexts must be a non-empty object");
  }
  const keys = Object.keys(candidateTexts);
  if (keys.length === 0) {
    throw new MDMValidationError("candidateTexts must be a non-empty object");
  }

  const quarantinedIds: string[] = [];
  const cleanIds: string[] = [];
  const reasonsById = Object.create(null) as Record<string, readonly string[]>;
  const smugglingCountsById = Object.create(null) as Record<string, number>;

  for (const rawId of keys) {
    if (rawId.trim().length === 0) {
      throw new MDMValidationError(
        "candidateTexts keys must be non-empty strings",
      );
    }
    const rawText = candidateTexts[rawId];
    if (typeof rawText !== "string") {
      throw new MDMValidationError(
        `candidateTexts[${JSON.stringify(rawId)}] must be a string`,
      );
    }

    const { normalized, audit } = auditAndNormalizeUntrustedText(rawText);
    smugglingCountsById[rawId] = audit.totalSuspiciousChars;
    const reasons: string[] = [];

    if (audit.unicodeTagChars > 0) {
      reasons.push("unicode_tag_smuggling");
    }
    if (audit.bidiOverrideChars > 0) {
      reasons.push("bidi_override_smuggling");
    }
    if (hasZeroWidthSmuggling(rawText, audit.zeroWidthChars)) {
      reasons.push("zero_width_smuggling");
    }
    for (const [label, pattern] of CUA_PATTERNS) {
      if (pattern.test(normalized)) {
        reasons.push(label);
      }
    }
    if (hasUntrustedTagBreakout(normalized)) {
      reasons.push("untrusted_tag_breakout");
    }
    if (
      audit.fullwidthHomoglyphs >= 2 &&
      (hasSuspiciousXmlTag(normalized) || reasons.length > 0)
    ) {
      if (!reasons.includes("fullwidth_homoglyph_evasion")) {
        reasons.push("fullwidth_homoglyph_evasion");
      }
    }

    if (reasons.length > 0) {
      quarantinedIds.push(rawId);
    } else {
      cleanIds.push(rawId);
    }
    reasonsById[rawId] = reasons;
  }

  return {
    quarantinedIds,
    cleanIds,
    reasonsById,
    smugglingCountsById,
    anyAttackDetected: quarantinedIds.length > 0,
  };
}
