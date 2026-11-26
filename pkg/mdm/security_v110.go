// Copyright 2026 Jigyasu Juneja
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package mdm

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

// InvisibleSmugglingAudit records invisible smuggling and homoglyph normalization counts.
type InvisibleSmugglingAudit struct {
	UnicodeTagChars     int
	BidiOverrideChars   int
	ZeroWidthChars      int
	ControlChars        int
	FullwidthHomoglyphs int
}

// TotalSuspiciousChars returns the total count of stripped or folded characters.
func (a InvisibleSmugglingAudit) TotalSuspiciousChars() int {
	return a.UnicodeTagChars + a.BidiOverrideChars + a.ZeroWidthChars + a.ControlChars + a.FullwidthHomoglyphs
}

// CanaryUntrustedEnvelope wraps untrusted text in an HMAC-salted per-request XML boundary.
type CanaryUntrustedEnvelope struct {
	CanaryHex      string
	OpenTag        string
	CloseTag       string
	SanitizedBody  string
	FencedText     string
	SmugglingAudit InvisibleSmugglingAudit
}

// ComparativeAttackAudit summarizes CUA, JMA, and invisible smuggling detection across candidates.
type ComparativeAttackAudit struct {
	QuarantinedIDs      []string
	CleanIDs            []string
	ReasonsByID         map[string][]string
	SmugglingCountsByID map[string]int
	AnyAttackDetected   bool
}

type cuaNamedPattern struct {
	label   string
	pattern *regexp.Regexp
}

var (
	canaryTagFamilyRE = regexp.MustCompile(`(?i)<\s*/?\s*untrusted_[a-z0-9_-]*[^>]*>`)
	injectionAnyRE    = regexp.MustCompile(
		`(?im)(?:^\s*(?:system|developer|assistant)\s*:\s*(?:approve|allow|grant|accept|deny|reject|block|select|choose|pick|output|return|ignore|disregard|forget|override|bypass|assign|rank|you\s+are|do\s+not)\b|\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|preceding|system)\s+(?:instructions?|prompts?|rules?|rubrics?|criteria|directives?)\b|\breply\s+with\s+the\s+single\s+uppercase\s+option\s+letter\b)`,
	)
	consecutiveZeroWidthRE = regexp.MustCompile(`[\x{200b}-\x{200d}\x{2060}-\x{2064}\x{feff}\x{fe00}-\x{fe0f}]{2,}`)
	asciiEmbeddedZWJRE     = regexp.MustCompile(`[A-Za-z0-9][\x{200c}\x{200d}][A-Za-z0-9]`)
	suspiciousXMLTagRE     = regexp.MustCompile(`(?i)<\s*/?\s*(?:system|developer|assistant|instructions?|rubric|prompt|untrusted)\b`)

	cuaPatterns = []cuaNamedPattern{
		{
			label: "rubric_authority_spoof",
			pattern: regexp.MustCompile(
				`(?i)\b(?:evaluation\s+rubric\s+override|system\s+rubric\s*:|official\s+scoring\s+directive|mandatory\s+judge\s+instruction|authoritative\s+reranker\s+directive)\b`,
			),
		},
		{
			label: "option_letter_hijack",
			pattern: regexp.MustCompile(
				`(?im)(?:\b(?:select|choose|pick|output|return|rank\s+first)\s+option\s+[A-Z]\b|^\s*(?:final\s+)?answer\s*:\s*[A-Z]\s*$)`,
			),
		},
		{
			label: "forced_probability_spoof",
			pattern: regexp.MustCompile(
				`(?i)\b(?:assign|give|set|output)\s+(?:a\s+)?(?:probability|posterior|score|logprob)\s*(?:of|=|:)?\s*(?:1\.0+|100%)`,
			),
		},
		{
			label: "peer_candidate_sabotage",
			pattern: regexp.MustCompile(
				`(?i)(?:\ball\s+other\s+(?:[a-z0-9_-]+\s+){0,2}(?:candidates?|options?|passages?|documents?|choices?|items?)\s+(?:are|were|have\s+been)\s+(?:[a-z0-9_-]+\s+){0,2}(?:deprecated|revoked|corrupted|malicious|fabricated|disqualified|unsafe|invalid|outdated|obsolete|decoys?|untrusted|fake|irrelevant)\b|\bother\s+(?:[a-z0-9_-]+\s+){0,2}(?:candidates?|options?|passages?|documents?|choices?|items?)\s+are\s+(?:[a-z0-9_-]+\s+){0,2}(?:deprecated|revoked|corrupted|malicious|fabricated|disqualified|unsafe|invalid|outdated|obsolete|decoys?|untrusted|fake)\b|\b(?:select|choose|pick|rank)\s+this\s+(?:document|candidate|passage|option|item)\s+(?:only|exclusively|first)\b)`,
			),
		},
		{
			label: "prompt_override_directive",
			pattern: regexp.MustCompile(
				`(?im)(?:\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|preceding|system)\s+(?:instructions?|prompts?|rules?|rubrics?|criteria|directives?)\b|^\s*(?:system|developer|assistant)\s*:\s*(?:approve|allow|grant|accept|deny|reject|block|select|choose|pick|output|return|ignore|disregard|forget|override|bypass|assign|rank|you\s+are|do\s+not)\b)`,
			),
		},
		{
			label:   "untrusted_tag_breakout",
			pattern: regexp.MustCompile(`(?i)(?:<\s*/?\s*untrusted_[a-z0-9_-]*[^>]*>|\[escaped_untrusted_tag\])`),
		},
	}
)

var mathLetterlikeHoles = map[rune]rune{
	0x2102: 'C', // Double-Struck Capital C
	0x210A: 'g', // Script Small g
	0x210B: 'H', // Script Capital H
	0x210C: 'H', // Black-Letter Capital H
	0x210D: 'H', // Double-Struck Capital H
	0x210E: 'h', // Planck Constant (Italic Small h)
	0x2110: 'I', // Script Capital I
	0x2111: 'I', // Black-Letter Capital I
	0x2112: 'L', // Script Capital L
	0x2113: 'l', // Script Small l
	0x2115: 'N', // Double-Struck Capital N
	0x2119: 'P', // Double-Struck Capital P
	0x211A: 'Q', // Double-Struck Capital Q
	0x211B: 'R', // Script Capital R
	0x211C: 'R', // Black-Letter Capital R
	0x211D: 'R', // Double-Struck Capital R
	0x2124: 'Z', // Double-Struck Capital Z
	0x2128: 'Z', // Black-Letter Capital Z
	0x212C: 'B', // Script Capital B
	0x212D: 'C', // Black-Letter Capital C
	0x212F: 'e', // Script Small e
	0x2130: 'E', // Script Capital E
	0x2131: 'F', // Script Capital F
	0x2133: 'M', // Script Capital M
	0x2134: 'o', // Script Small o
}

func foldMathAlphanumericRune(r rune) (rune, bool) {
	if mapped, ok := mathLetterlikeHoles[r]; ok {
		return mapped, true
	}
	switch {
	// Mathematical Bold
	case r >= 0x1D400 && r <= 0x1D419:
		return 'A' + (r - 0x1D400), true
	case r >= 0x1D41A && r <= 0x1D433:
		return 'a' + (r - 0x1D41A), true
	// Mathematical Italic (hole at U+1D455 -> U+210E)
	case r >= 0x1D434 && r <= 0x1D44D:
		return 'A' + (r - 0x1D434), true
	case r >= 0x1D44E && r <= 0x1D454:
		return 'a' + (r - 0x1D44E), true
	case r >= 0x1D456 && r <= 0x1D467:
		return 'a' + (r - 0x1D44E), true
	// Mathematical Bold Italic
	case r >= 0x1D468 && r <= 0x1D481:
		return 'A' + (r - 0x1D468), true
	case r >= 0x1D482 && r <= 0x1D49B:
		return 'a' + (r - 0x1D482), true
	// Mathematical Script (holes in Letterlike Symbols)
	case r == 0x1D49C,
		r >= 0x1D49E && r <= 0x1D49F,
		r == 0x1D4A2,
		r >= 0x1D4A5 && r <= 0x1D4A6,
		r >= 0x1D4A9 && r <= 0x1D4AC,
		r >= 0x1D4AE && r <= 0x1D4B5:
		return 'A' + (r - 0x1D49C), true
	case r >= 0x1D4B6 && r <= 0x1D4B9,
		r == 0x1D4BB,
		r >= 0x1D4BD && r <= 0x1D4C3,
		r >= 0x1D4C5 && r <= 0x1D4CF:
		return 'a' + (r - 0x1D4B6), true
	// Mathematical Bold Script
	case r >= 0x1D4D0 && r <= 0x1D4E9:
		return 'A' + (r - 0x1D4D0), true
	case r >= 0x1D4EA && r <= 0x1D503:
		return 'a' + (r - 0x1D4EA), true
	// Mathematical Fraktur (holes in Letterlike Symbols)
	case r >= 0x1D504 && r <= 0x1D505,
		r >= 0x1D507 && r <= 0x1D50A,
		r >= 0x1D50D && r <= 0x1D514,
		r >= 0x1D516 && r <= 0x1D51C:
		return 'A' + (r - 0x1D504), true
	case r >= 0x1D51E && r <= 0x1D537:
		return 'a' + (r - 0x1D51E), true
	// Mathematical Double-Struck (holes in Letterlike Symbols)
	case r >= 0x1D538 && r <= 0x1D539,
		r >= 0x1D53B && r <= 0x1D53E,
		r >= 0x1D540 && r <= 0x1D544,
		r == 0x1D546,
		r >= 0x1D54A && r <= 0x1D550:
		return 'A' + (r - 0x1D538), true
	case r >= 0x1D552 && r <= 0x1D56B:
		return 'a' + (r - 0x1D552), true
	// Mathematical Bold Fraktur
	case r >= 0x1D56C && r <= 0x1D585:
		return 'A' + (r - 0x1D56C), true
	case r >= 0x1D586 && r <= 0x1D59F:
		return 'a' + (r - 0x1D586), true
	// Mathematical Sans-Serif
	case r >= 0x1D5A0 && r <= 0x1D5B9:
		return 'A' + (r - 0x1D5A0), true
	case r >= 0x1D5BA && r <= 0x1D5D3:
		return 'a' + (r - 0x1D5BA), true
	// Mathematical Sans-Serif Bold
	case r >= 0x1D5D4 && r <= 0x1D5ED:
		return 'A' + (r - 0x1D5D4), true
	case r >= 0x1D5EE && r <= 0x1D607:
		return 'a' + (r - 0x1D5EE), true
	// Mathematical Sans-Serif Italic
	case r >= 0x1D608 && r <= 0x1D621:
		return 'A' + (r - 0x1D608), true
	case r >= 0x1D622 && r <= 0x1D63B:
		return 'a' + (r - 0x1D622), true
	// Mathematical Sans-Serif Bold Italic
	case r >= 0x1D63C && r <= 0x1D655:
		return 'A' + (r - 0x1D63C), true
	case r >= 0x1D656 && r <= 0x1D66F:
		return 'a' + (r - 0x1D656), true
	// Mathematical Monospace
	case r >= 0x1D670 && r <= 0x1D689:
		return 'A' + (r - 0x1D670), true
	case r >= 0x1D68A && r <= 0x1D6A3:
		return 'a' + (r - 0x1D68A), true
	// Mathematical Digits (Bold, Double-Struck, Sans-Serif, Sans-Serif Bold, Monospace)
	case r >= 0x1D7CE && r <= 0x1D7FF:
		return '0' + (r-0x1D7CE)%10, true
	default:
		return r, false
	}
}

// AuditAndNormalizeUntrustedText strips invisible smuggling characters and folds ASCII homoglyphs.
func AuditAndNormalizeUntrustedText(rawText string) (string, InvisibleSmugglingAudit) {
	var audit InvisibleSmugglingAudit
	var b strings.Builder
	b.Grow(len(rawText))

	for _, r := range rawText {
		if r >= 0xE0000 && r <= 0xE007F {
			audit.UnicodeTagChars++
			continue
		}
		if (r >= 0xE0100 && r <= 0xE01EF) || (r >= 0xFE00 && r <= 0xFE0F) {
			audit.ZeroWidthChars++
			continue
		}
		if (r >= 0x202A && r <= 0x202E) || (r >= 0x2066 && r <= 0x2069) || r == 0x061C || r == 0x200E || r == 0x200F {
			audit.BidiOverrideChars++
			continue
		}
		if r == 0x00AD {
			continue
		}
		if (r >= 0x200B && r <= 0x200D) || (r >= 0x2060 && r <= 0x2064) || r == 0xFEFF {
			audit.ZeroWidthChars++
			continue
		}
		if (r >= 0x00 && r <= 0x08) || r == 0x0B || r == 0x0C || (r >= 0x0E && r <= 0x1F) || (r >= 0x7F && r <= 0x9F) {
			audit.ControlChars++
			continue
		}
		if folded, ok := foldMathAlphanumericRune(r); ok {
			audit.FullwidthHomoglyphs++
			b.WriteRune(folded)
			continue
		}
		if r >= 0xFF01 && r <= 0xFF5E {
			audit.FullwidthHomoglyphs++
			b.WriteRune(r - 0xFEE0)
			continue
		}
		if r == 0x3000 {
			audit.FullwidthHomoglyphs++
			b.WriteByte(' ')
			continue
		}
		switch r {
		case 0x2039, 0x2329, 0x27E8, 0x3008, 0xFE64:
			audit.FullwidthHomoglyphs++
			b.WriteByte('<')
		case 0x203A, 0x232A, 0x27E9, 0x3009, 0xFE65:
			audit.FullwidthHomoglyphs++
			b.WriteByte('>')
		default:
			b.WriteRune(r)
		}
	}
	return b.String(), audit
}

func hasZeroWidthSmuggling(rawText string, zeroWidthCount int) bool {
	if zeroWidthCount <= 0 {
		return false
	}
	explicitCount := 0
	runeIdx := 0
	for _, r := range rawText {
		if r >= 0xE0100 && r <= 0xE01EF {
			return true
		}
		if r == 0x200B || (r >= 0x2060 && r <= 0x2064) || (r == 0xFEFF && runeIdx > 0) {
			explicitCount++
		}
		runeIdx++
	}
	if explicitCount >= 1 && zeroWidthCount >= 2 {
		return true
	}
	if consecutiveZeroWidthRE.MatchString(rawText) {
		return true
	}
	return asciiEmbeddedZWJRE.MatchString(rawText)
}

func truncateRunes(input string, maxChars int) string {
	if utf8.RuneCountInString(input) <= maxChars {
		return input
	}
	var b strings.Builder
	count := 0
	for _, r := range input {
		if count >= maxChars {
			break
		}
		b.WriteRune(r)
		count++
	}
	return b.String()
}

// toLowerASCII lowercases only ASCII bytes ('A'..'Z') so byte length and offsets stay strictly 1:1 on all UTF-8 input.
func toLowerASCII(s string) string {
	buf := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		b := s[i]
		if b >= 'A' && b <= 'Z' {
			b += 'a' - 'A'
		}
		buf[i] = b
	}
	return string(buf)
}

// replaceASCIICaseInsensitive replaces every ASCII case-insensitive occurrence of needle in text in O(N) time.
func replaceASCIICaseInsensitive(text, needle, replacement string) string {
	if needle == "" || len(text) < len(needle) {
		return text
	}
	lowerText := toLowerASCII(text)
	lowerNeedle := toLowerASCII(needle)
	var out strings.Builder
	out.Grow(len(text))
	start := 0
	for {
		idx := strings.Index(lowerText[start:], lowerNeedle)
		if idx < 0 {
			out.WriteString(text[start:])
			break
		}
		matchStart := start + idx
		out.WriteString(text[start:matchStart])
		out.WriteString(replacement)
		start = matchStart + len(lowerNeedle)
	}
	return out.String()
}

// WrapCanaryUntrustedEnvelope wraps untrusted text in an unpredictable per-request HMAC-salted XML canary boundary.
// Pass seedKey == "" to use DefaultConformalSeedKey and maxChars <= 0 to default to 16384 characters.
func WrapCanaryUntrustedEnvelope(
	rawText string,
	requestID string,
	seedKey string,
	maxChars int,
) (CanaryUntrustedEnvelope, error) {
	cleanReqID := strings.TrimSpace(requestID)
	if cleanReqID == "" {
		return CanaryUntrustedEnvelope{}, fmt.Errorf("mdm: requestID must be a non-empty string")
	}
	effectiveSeed := seedKey
	if effectiveSeed == "" {
		effectiveSeed = DefaultConformalSeedKey
	} else if strings.TrimSpace(effectiveSeed) == "" {
		return CanaryUntrustedEnvelope{}, fmt.Errorf("mdm: seedKey must not be whitespace-only")
	}
	if maxChars <= 0 {
		maxChars = 16384
	}

	mac := hmac.New(sha256.New, []byte(effectiveSeed))
	_, _ = mac.Write([]byte("canary:" + cleanReqID))
	canaryHex := hex.EncodeToString(mac.Sum(nil))[:12]
	openTag := "<untrusted_data_" + canaryHex + ">"
	closeTag := "</untrusted_data_" + canaryHex + ">"

	normalized, audit := AuditAndNormalizeUntrustedText(rawText)
	escapedTags := canaryTagFamilyRE.ReplaceAllString(normalized, "[escaped_untrusted_tag]")
	escapedCanary := replaceASCIICaseInsensitive(escapedTags, canaryHex, "[escaped_canary]")
	neutralized := injectionAnyRE.ReplaceAllString(escapedCanary, "[neutralized_directive]")
	clipped := truncateRunes(neutralized, maxChars)

	lowerClipped := strings.ToLower(clipped)
	if strings.Contains(lowerClipped, strings.ToLower(closeTag)) || strings.Contains(lowerClipped, strings.ToLower(openTag)) {
		return CanaryUntrustedEnvelope{}, fmt.Errorf("mdm: canary tag collision survived sanitization")
	}

	return CanaryUntrustedEnvelope{
		CanaryHex:      canaryHex,
		OpenTag:        openTag,
		CloseTag:       closeTag,
		SanitizedBody:  clipped,
		FencedText:     openTag + "\n" + clipped + "\n" + closeTag,
		SmugglingAudit: audit,
	}, nil
}

// DetectComparativeUnderminingAttack audits candidate passages or options for CUA, JMA, and Unicode smuggling.
func DetectComparativeUnderminingAttack(candidateTexts map[string]string) (ComparativeAttackAudit, error) {
	if len(candidateTexts) == 0 {
		return ComparativeAttackAudit{}, fmt.Errorf("mdm: candidateTexts must be a non-empty map")
	}

	orderedIDs := make([]string, 0, len(candidateTexts))
	for id := range candidateTexts {
		if strings.TrimSpace(id) == "" {
			return ComparativeAttackAudit{}, fmt.Errorf("mdm: candidateTexts keys must be non-empty strings")
		}
		orderedIDs = append(orderedIDs, id)
	}
	sort.Strings(orderedIDs)

	var quarantined []string
	clean := make([]string, 0, len(orderedIDs))
	reasonsByID := make(map[string][]string, len(orderedIDs))
	smugglingCounts := make(map[string]int, len(orderedIDs))

	for _, id := range orderedIDs {
		rawText := candidateTexts[id]
		normalized, audit := AuditAndNormalizeUntrustedText(rawText)
		smugglingCounts[id] = audit.TotalSuspiciousChars()
		var reasons []string

		if audit.UnicodeTagChars > 0 {
			reasons = append(reasons, "unicode_tag_smuggling")
		}
		if audit.BidiOverrideChars > 0 {
			reasons = append(reasons, "bidi_override_smuggling")
		}
		if hasZeroWidthSmuggling(rawText, audit.ZeroWidthChars) {
			reasons = append(reasons, "zero_width_smuggling")
		}
		for _, entry := range cuaPatterns {
			if entry.pattern.MatchString(normalized) {
				reasons = append(reasons, entry.label)
			}
		}
		if audit.FullwidthHomoglyphs >= 2 && (suspiciousXMLTagRE.MatchString(normalized) || len(reasons) > 0) {
			reasons = append(reasons, "fullwidth_homoglyph_evasion")
		}

		if len(reasons) > 0 {
			quarantined = append(quarantined, id)
		} else {
			clean = append(clean, id)
		}
		reasonsByID[id] = reasons
	}

	return ComparativeAttackAudit{
		QuarantinedIDs:      quarantined,
		CleanIDs:            clean,
		ReasonsByID:         reasonsByID,
		SmugglingCountsByID: smugglingCounts,
		AnyAttackDetected:   len(quarantined) > 0,
	}, nil
}
