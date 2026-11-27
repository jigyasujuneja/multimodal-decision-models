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

import type { DecisionQuestion } from "./types.js";

/** Maps angle-bracket homoglyph code points to ASCII `<` and `>`. */
export const ANGLE_HOMOGLYPH_MAP: ReadonlyMap<number, string> = new Map<number, string>([
  [0x2039, "<"],
  [0x203a, ">"],
  [0x2329, "<"],
  [0x232a, ">"],
  [0x27e8, "<"],
  [0x27e9, ">"],
  [0x3008, "<"],
  [0x3009, ">"],
  [0xfe64, "<"],
  [0xfe65, ">"],
]);

/** Returns true when value is a non-null, non-array plain record object. */
export function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Creates a null-prototype dictionary to guard against prototype-key pollution. */
export function nullProtoRecord<T>(): Record<string, T> {
  return Object.create(null) as Record<string, T>;
}

/** Extracts the ordered option keys declared on a DecisionQuestion. */
export function getQuestionOptionKeys(question: DecisionQuestion): string[] {
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

/** Throws the abort reason immediately if the supplied AbortSignal is aborted. */
export function throwIfSignalAborted(signal?: AbortSignal): void {
  if (!signal || !signal.aborted) {
    return;
  }
  throw signal.reason ?? new DOMException("The operation was aborted.", "AbortError");
}

/** Cancels a readable stream or reader while suppressing secondary cancellation errors. */
export async function cancelStreamQuietly(
  cancelable: { cancel(): Promise<unknown> } | null | undefined,
): Promise<void> {
  if (!cancelable) {
    return;
  }
  try {
    await cancelable.cancel();
  } catch {
    // Stream cancellation failure is secondary to the primary error being propagated.
  }
}

/** Formats an unknown thrown value into a safe message string. */
export function formatUnknownError(err: unknown): string {
  if (err instanceof Error) {
    return err.message;
  }
  // Custom fetch implementations or AbortSignal.reason may reject with non-Error values.
  return String(err);
}

/** Returns true when the UTF-16 code unit is ASCII or non-breaking tag whitespace. */
export function isTagWhitespace(code: number): boolean {
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

/** Replaces `< ... untrusted_... >` tag variants in linear O(N) time without regex backtracking. */
export function escapeUntrustedTagVariantsInternal(input: string): string {
  const n = input.length;
  let lastEmit = 0;
  let i = 0;
  const parts: string[] = [];

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
    if (
      cursor + 10 <= n &&
      input.slice(cursor, cursor + 10).toLowerCase() === "untrusted_"
    ) {
      let scan = cursor + 10;
      while (
        scan < n &&
        input.charCodeAt(scan) !== 0x3e /* '>' */ &&
        input.charCodeAt(scan) !== 0x3c /* '<' */
      ) {
        scan += 1;
      }
      if (scan < n && input.charCodeAt(scan) === 0x3e /* '>' */) {
        if (start > lastEmit) {
          parts.push(input.slice(lastEmit, start));
        }
        parts.push("[escaped_untrusted_tag]");
        i = scan + 1;
        lastEmit = i;
        continue;
      }
      i = scan;
      continue;
    }
    i = cursor > start + 1 ? cursor : start + 1;
  }

  if (lastEmit === 0) {
    return input;
  }
  if (lastEmit < n) {
    parts.push(input.slice(lastEmit));
  }
  return parts.join("");
}

function toLowerAscii(text: string): string {
  const chars = new Array<string>(text.length);
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    chars[i] = code >= 0x41 && code <= 0x5a ? String.fromCharCode(code + 0x20) : text.charAt(i);
  }
  return chars.join("");
}

/** Replaces all ASCII case-insensitive occurrences of needle in text in O(N) time. */
export function replaceAsciiCaseInsensitive(
  text: string,
  needle: string,
  replacement: string,
): string {
  if (needle.length === 0 || text.length < needle.length) {
    return text;
  }
  const lowerText = toLowerAscii(text);
  const lowerNeedle = toLowerAscii(needle);
  const parts: string[] = [];
  let cursor = 0;
  while (cursor < text.length) {
    const matchIndex = lowerText.indexOf(lowerNeedle, cursor);
    if (matchIndex === -1) {
      parts.push(text.slice(cursor));
      break;
    }
    parts.push(text.slice(cursor, matchIndex), replacement);
    cursor = matchIndex + lowerNeedle.length;
  }
  return parts.join("");
}

/** Truncates text to at most maxChars Unicode code points without splitting UTF-16 surrogate pairs. */
export function truncateCodePoints(text: string, maxChars: number): string {
  if (text.length <= maxChars) {
    return text;
  }
  const codePoints = Array.from(text);
  return codePoints.length > maxChars ? codePoints.slice(0, maxChars).join("") : text;
}
