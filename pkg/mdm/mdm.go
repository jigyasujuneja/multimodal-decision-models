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

// Package mdm provides standalone Go reference primitives for calibrated multimodal decision models:
// pre-gated log-linear prior fusion, dense-prior wide-option shortlisting with sibling-unique contrastive tags,
// hierarchical semantic taxonomy trie decoding, Set-of-Marks spatial geometry, and transitive triangle-consistency checking.
package mdm

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const (
	MaxOptionsPerSlot             = 26
	MaxExtendedOptionsPerSlot     = 256
	MaxSlotsPerPass               = 8
	DefaultCosinePriorTemperature = 0.07
)

// Universal256DecisionCodes lists 256 uppercase 1-letter and 2-letter decision codes (A..Z, AA..PT)
// verified to tokenize as a single BPE token (both bare and space-prefixed) across Qwen3, DeepSeek-V3,
// Gemma-3, Llama-3.2, and SmolLM2 tokenizers (arXiv:2610.03935 Section 6.1 & DecisionBench 1.0).
var Universal256DecisionCodes = [256]string{
	"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
	"Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF",
	"AG", "AH", "AI", "AK", "AL", "AM", "AN", "AP", "AR", "AS", "AT", "AU", "AV", "AW", "BA", "BB",
	"BC", "BD", "BE", "BI", "BL", "BM", "BO", "BP", "BR", "BS", "BT", "BU", "BY", "CA", "CB", "CC",
	"CD", "CE", "CF", "CG", "CH", "CI", "CL", "CM", "CN", "CO", "CP", "CR", "CS", "CT", "CU", "CV",
	"DA", "DB", "DC", "DD", "DE", "DF", "DH", "DI", "DL", "DM", "DN", "DO", "DP", "DR", "DS", "DT",
	"EA", "EC", "ED", "EE", "EF", "EG", "EL", "EM", "EN", "EP", "ER", "ES", "ET", "EU", "EV", "EX",
	"FA", "FC", "FD", "FE", "FI", "FL", "FM", "FO", "FP", "FR", "FS", "FT", "GA", "GB", "GC", "GD",
	"GE", "GH", "GI", "GL", "GM", "GN", "GO", "GP", "GR", "GS", "GT", "GW", "HA", "HC", "HD", "HE",
	"HF", "HI", "HL", "HO", "HP", "HR", "HS", "HT", "HV", "IA", "IB", "IC", "ID", "IE", "IF", "IG",
	"II", "IL", "IM", "IN", "IO", "IP", "IR", "IS", "IT", "IU", "IV", "IX", "JC", "JD", "JO", "JS",
	"KA", "KB", "KE", "KS", "LA", "LC", "LD", "LE", "LG", "LI", "LL", "LM", "LO", "LP", "LS", "LT",
	"MA", "MB", "MC", "MD", "ME", "MH", "MI", "ML", "MM", "MN", "MO", "MP", "MR", "MS", "MT", "MW",
	"MY", "NA", "NC", "ND", "NE", "NF", "NG", "NH", "NI", "NL", "NM", "NO", "NP", "NR", "NS", "NT",
	"NV", "NW", "NY", "OA", "OB", "OC", "OD", "OF", "OH", "OK", "OM", "ON", "OP", "OR", "OS", "OT",
	"PA", "PC", "PD", "PE", "PF", "PG", "PH", "PI", "PK", "PL", "PM", "PO", "PP", "PR", "PS", "PT",
}

// BuildExtended255CodeBijection maps up to 256 option keys to universal single-token uppercase codes.
func BuildExtended255CodeBijection(optionKeys []string) (map[string]string, error) {
	if len(optionKeys) < 2 {
		return nil, fmt.Errorf("question requires >= 2 options, got %d", len(optionKeys))
	}
	if len(optionKeys) > MaxExtendedOptionsPerSlot {
		return nil, fmt.Errorf("question has %d options, exceeding extended ceiling %d", len(optionKeys), MaxExtendedOptionsPerSlot)
	}
	seen := make(map[string]struct{}, len(optionKeys))
	out := make(map[string]string, len(optionKeys))
	for i, k := range optionKeys {
		if _, dup := seen[k]; dup {
			return nil, fmt.Errorf("duplicate option key %q", k)
		}
		seen[k] = struct{}{}
		out[Universal256DecisionCodes[i]] = k
	}
	return out, nil
}

// NormalizeCosinePriorLogits converts raw cosine similarities into temperature-scaled log-softmax
// prior logits: log P(k) = (cos_k / tau) - logsumexp_j(cos_j / tau).
func NormalizeCosinePriorLogits(cosines map[string]float64, temperature float64) map[string]float64 {
	if len(cosines) == 0 {
		return map[string]float64{}
	}
	tau := temperature
	if math.IsNaN(tau) || math.IsInf(tau, 0) || tau <= 0 {
		tau = DefaultCosinePriorTemperature
	}
	scaled := make(map[string]float64, len(cosines))
	maxS := math.Inf(-1)
	for k, v := range cosines {
		fv := v
		if math.IsNaN(fv) || math.IsInf(fv, 0) {
			fv = -1.0
		}
		s := fv / tau
		scaled[k] = s
		if s > maxS {
			maxS = s
		}
	}
	sumExp := 0.0
	for _, s := range scaled {
		sumExp += math.Exp(s - maxS)
	}
	lse := maxS + math.Log(sumExp)
	out := make(map[string]float64, len(scaled))
	for k, s := range scaled {
		out[k] = s - lse
	}
	return out
}

// PackedBracketPassCount returns exact forward passes for tournament brackets when up to
// maxSlotsPerPass Round-1 brackets pack onto one canvas pass: 1 + ceil(ceil(K/B)/S).
func PackedBracketPassCount(numOptions, bracketSize, maxSlotsPerPass int) int {
	if numOptions <= MaxOptionsPerSlot {
		return 1
	}
	if bracketSize < 2 {
		bracketSize = 20
	}
	if maxSlotsPerPass < 1 {
		maxSlotsPerPass = MaxSlotsPerPass
	}
	round1 := (numOptions + bracketSize - 1) / bracketSize
	return 1 + (round1+maxSlotsPerPass-1)/maxSlotsPerPass
}

// NormalizeAndScale normalizes raw option probabilities with temperature scaling T > 0,
// returning (scaledProbs, argmaxWinner, shannonEntropy, normalizedEntropy).
func NormalizeAndScale(raw map[string]float64, optKeys []string, tempScale float64) (map[string]float64, string, float64, float64) {
	if tempScale <= 0 {
		tempScale = 1.0
	}
	if len(optKeys) == 0 {
		return map[string]float64{}, "", 0, 0
	}
	scaled := make(map[string]float64, len(optKeys))
	sum := 0.0
	for _, k := range optKeys {
		p := raw[k]
		if math.IsNaN(p) || math.IsInf(p, 0) || p <= 1e-9 {
			p = 1e-6
		}
		sp := math.Pow(p, 1.0/tempScale)
		scaled[k] = sp
		sum += sp
	}
	winner := optKeys[0]
	bestP := -1.0
	h := 0.0
	for _, k := range optKeys {
		p := scaled[k] / sum
		scaled[k] = p
		if p > bestP {
			bestP = p
			winner = k
		}
		if p > 0 {
			h -= p * math.Log(p)
		}
	}
	normH := 0.0
	if len(optKeys) > 1 {
		normH = h / math.Log(float64(len(optKeys)))
	}
	return scaled, winner, h, normH
}

// FusePriorWithGate computes raw normalized hesitation before applying log-linear prior fusion
// according to gateMode ("always", "confident", "hesitant", "proportional", "never", or "off").
func FusePriorWithGate(
	rawProbs map[string]float64,
	priorLogits map[string]float64,
	optKeys []string,
	tempScale float64,
	priorWeight float64,
	hesitationGate float64,
	gateMode string,
) (probs map[string]float64, winner string, normH, rawNormH float64, fused bool) {
	baseProbs, baseWinner, _, baseNormH := NormalizeAndScale(rawProbs, optKeys, tempScale)
	rawNormH = baseNormH
	if gateMode == "off" || gateMode == "never" || len(priorLogits) == 0 || priorWeight <= 0 || len(optKeys) <= 1 {
		return baseProbs, baseWinner, baseNormH, rawNormH, false
	}

	effWeight := priorWeight
	switch gateMode {
	case "always":
		// Always fuse priorLogits while preserving pre-fusion rawNormH for cascade gating.
	case "", "confident":
		if hesitationGate > 0 && rawNormH >= hesitationGate {
			return baseProbs, baseWinner, baseNormH, rawNormH, false
		}
	case "hesitant":
		if hesitationGate > 0 && rawNormH < hesitationGate {
			return baseProbs, baseWinner, baseNormH, rawNormH, false
		}
	case "proportional":
		effWeight = priorWeight * rawNormH
		if effWeight <= 1e-9 {
			return baseProbs, baseWinner, baseNormH, rawNormH, false
		}
	default:
		return baseProbs, baseWinner, baseNormH, rawNormH, false
	}

	logits := make(map[string]float64, len(optKeys))
	maxLogit := math.Inf(-1)
	for _, k := range optKeys {
		p := math.Max(1e-9, baseProbs[k])
		l := math.Log(p) + effWeight*priorLogits[k]
		logits[k] = l
		if l > maxLogit {
			maxLogit = l
		}
	}
	unnorm := make(map[string]float64, len(optKeys))
	for _, k := range optKeys {
		unnorm[k] = math.Exp(logits[k] - maxLogit)
	}
	fusedProbs, fusedWinner, _, fusedNormH := NormalizeAndScale(unnorm, optKeys, 1.0)
	return fusedProbs, fusedWinner, fusedNormH, rawNormH, true
}

// TopKFromPriorLogits selects up to topK highest-scoring options from priorLogits (default M=20)
// while preserving any catchAllKey (if present in criteria) within MaxOptionsPerSlot. Returns nil when priorLogits is empty.
func TopKFromPriorLogits(criteria map[string]string, priorLogits map[string]float64, topK int, catchAllKey string) []string {
	if len(priorLogits) == 0 {
		return nil
	}
	if topK <= 0 || topK > MaxOptionsPerSlot {
		topK = 20
	}
	keys := make([]string, 0, len(criteria))
	for k := range criteria {
		if k != catchAllKey {
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool {
		si, sj := priorLogits[keys[i]], priorLogits[keys[j]]
		if si != sj {
			return si > sj
		}
		return keys[i] < keys[j]
	})
	hasCatchAll := catchAllKey != "" && criteria[catchAllKey] != ""
	budget := topK
	if hasCatchAll && budget+1 > MaxOptionsPerSlot {
		budget = MaxOptionsPerSlot - 1
	}
	if budget > len(keys) {
		budget = len(keys)
	}
	out := append([]string(nil), keys[:budget]...)
	if hasCatchAll {
		out = append(out, catchAllKey)
	}
	return out
}

var (
	reCamelSplit = regexp.MustCompile(`([a-z0-9])([A-Z])`)
	reNonAlnum   = regexp.MustCompile(`[^a-zA-Z0-9]+`)
	mdmStopwords = map[string]bool{
		"a": true, "an": true, "and": true, "are": true, "as": true, "at": true,
		"be": true, "by": true, "for": true, "from": true, "in": true, "into": true,
		"is": true, "it": true, "of": true, "on": true, "or": true, "that": true,
		"the": true, "their": true, "this": true, "to": true, "user": true, "when": true, "with": true,
	}
)

func splitIdentifierTokens(s string) []string {
	expanded := reCamelSplit.ReplaceAllString(s, "$1 $2")
	parts := reNonAlnum.Split(strings.ToLower(expanded), -1)
	seen := make(map[string]bool, len(parts))
	var out []string
	for _, p := range parts {
		if len(p) < 2 || mdmStopwords[p] || seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	return out
}

// BuildContrastiveCriteria appends up to maxTags sibling-unique "[distinguishing: ...]" cues
// to shortlisted option descriptions so near-neighbor options are easier to separate in 1 pass.
func BuildContrastiveCriteria(criteria map[string]string, selectedKeys []string, catchAllKey string, maxTags int) map[string]string {
	if maxTags <= 0 {
		maxTags = 2
	}
	var nonCatch []string
	for _, k := range selectedKeys {
		if k != catchAllKey {
			nonCatch = append(nonCatch, k)
		}
	}
	tokSets := make(map[string][]string, len(nonCatch))
	docFreq := make(map[string]int)
	for _, k := range nonCatch {
		toks := splitIdentifierTokens(k + " " + criteria[k])
		tokSets[k] = toks
		for _, tok := range toks {
			docFreq[tok]++
		}
	}
	out := make(map[string]string, len(selectedKeys))
	for _, k := range selectedKeys {
		base := criteria[k]
		if base == "" {
			base = k
		}
		if k == catchAllKey || len(nonCatch) <= 1 {
			out[k] = base
			continue
		}
		toks := tokSets[k]
		var unique []string
		for _, t := range toks {
			if docFreq[t] == 1 {
				unique = append(unique, t)
			}
		}
		if len(unique) == 0 {
			half := len(nonCatch)/2 + 1
			if half < 2 {
				half = 2
			}
			for _, t := range toks {
				if docFreq[t] < half {
					unique = append(unique, t)
				}
			}
			sort.SliceStable(unique, func(i, j int) bool {
				return docFreq[unique[i]] < docFreq[unique[j]]
			})
		}
		if len(unique) > maxTags {
			unique = unique[:maxTags]
		}
		if len(unique) > 0 {
			out[k] = fmt.Sprintf("%s [distinguishing: %s]", base, strings.Join(unique, ", "))
		} else {
			out[k] = base
		}
	}
	return out
}

// TaxonomyLeaf maps a leaf option key to its (CoarseKey, FineKey) coordinates in a 2-level taxonomy trie.
type TaxonomyLeaf struct {
	LeafKey   string
	CoarseKey string
	FineKey   string
}

// DecodeTaxonomyTrie computes exact joint leaf probabilities strictly over valid root-to-leaf paths
// via s(e|x) = log P_coarse(c_1(e)|x) + log P_fine(c_2(e)|x), eliminating invalid cross-slot combinations.
func DecodeTaxonomyTrie(coarseProbs, fineProbs map[string]float64, leaves []TaxonomyLeaf, catchAllKey string) (map[string]float64, string) {
	if len(leaves) == 0 {
		return map[string]float64{}, ""
	}
	const eps = 1e-12
	cTot := 0.0
	for _, v := range coarseProbs {
		if v > 0 {
			cTot += v
		}
	}
	if cTot <= 0 {
		cTot = 1.0
	}
	fTot := 0.0
	for _, v := range fineProbs {
		if v > 0 {
			fTot += v
		}
	}
	if fTot <= 0 {
		fTot = 1.0
	}

	logScores := make(map[string]float64, len(leaves))
	keys := make([]string, 0, len(leaves))
	maxLP := math.Inf(-1)
	for _, leaf := range leaves {
		keys = append(keys, leaf.LeafKey)
		pc := math.Max(eps, math.Max(0, coarseProbs[leaf.CoarseKey])/cTot)
		pf := math.Max(eps, math.Max(0, fineProbs[leaf.FineKey])/fTot)
		var lp float64
		if catchAllKey != "" && leaf.LeafKey == catchAllKey {
			lp = 0.5 * (math.Log(pc) + math.Log(pf))
		} else {
			lp = math.Log(pc) + math.Log(pf)
		}
		logScores[leaf.LeafKey] = lp
		if lp > maxLP {
			maxLP = lp
		}
	}
	sort.Strings(keys)

	exps := make(map[string]float64, len(keys))
	denom := 0.0
	for _, k := range keys {
		ev := math.Exp(logScores[k] - maxLP)
		exps[k] = ev
		denom += ev
	}
	probs := make(map[string]float64, len(keys))
	winner := keys[0]
	bestP := -1.0
	for _, k := range keys {
		p := exps[k] / denom
		probs[k] = p
		if p > bestP {
			bestP = p
			winner = k
		}
	}
	return probs, winner
}

// CandidateMark represents an interactive UI element with [y_min, x_min, y_max, x_max] coordinates in [0, 100].
type CandidateMark struct {
	ID    string
	Label string
	Box   [4]float64
}

// SelectContainingOrNearestMark returns the smallest-area mark containing (cy, cx), or the nearest mark by center distance.
func SelectContainingOrNearestMark(cy, cx float64, marks []CandidateMark) (CandidateMark, float64, bool) {
	if len(marks) == 0 {
		return CandidateMark{}, math.Inf(1), false
	}
	bestContainIdx := -1
	bestArea := math.Inf(1)
	bestContainDist := math.Inf(1)

	bestNearIdx := 0
	bestNearDist := math.Inf(1)

	for i, m := range marks {
		mcy := (m.Box[0] + m.Box[2]) / 2.0
		mcx := (m.Box[1] + m.Box[3]) / 2.0
		dist := math.Hypot(cy-mcy, cx-mcx)
		if dist < bestNearDist {
			bestNearDist = dist
			bestNearIdx = i
		}
		if cy >= m.Box[0] && cy <= m.Box[2] && cx >= m.Box[1] && cx <= m.Box[3] {
			area := math.Max(0, m.Box[2]-m.Box[0]) * math.Max(0, m.Box[3]-m.Box[1])
			if area < bestArea || (area == bestArea && dist < bestContainDist) {
				bestArea = area
				bestContainDist = dist
				bestContainIdx = i
			}
		}
	}
	if bestContainIdx >= 0 {
		return marks[bestContainIdx], 0, true
	}
	return marks[bestNearIdx], bestNearDist, true
}

var (
	grid4x4Rows = [4]string{"r0", "r1", "r2", "r3"}
	grid4x4Cols = [4]string{"c0", "c1", "c2", "c3"}
)

// AssignPointToGrid4x4 maps normalized (cy, cx) in [0, 100] to a 4x4 cell ("r0_c0" .. "r3_c3").
func AssignPointToGrid4x4(cy, cx float64) string {
	r := int((cy / 100.0) * 4.0)
	if r < 0 {
		r = 0
	} else if r > 3 {
		r = 3
	}
	c := int((cx / 100.0) * 4.0)
	if c < 0 {
		c = 0
	} else if c > 3 {
		c = 3
	}
	return grid4x4Rows[r] + "_" + grid4x4Cols[c]
}

// AssignBoxToGrid4x4 maps a [y_min, x_min, y_max, x_max] box center to its 4x4 cell.
func AssignBoxToGrid4x4(box [4]float64) string {
	return AssignPointToGrid4x4((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)
}

// Grid4x4CellBounds returns [y_min, x_min, y_max, x_max] bounds in [0, 100] for a 4x4 cell.
func Grid4x4CellBounds(cell string) ([4]float64, bool) {
	parts := strings.Split(strings.TrimSpace(cell), "_")
	if len(parts) != 2 {
		return [4]float64{}, false
	}
	r, c := -1, -1
	for idx, name := range grid4x4Rows {
		if parts[0] == name {
			r = idx
			break
		}
	}
	for idx, name := range grid4x4Cols {
		if parts[1] == name {
			c = idx
			break
		}
	}
	if r < 0 || c < 0 {
		return [4]float64{}, false
	}
	return [4]float64{float64(r) * 25.0, float64(c) * 25.0, float64(r+1) * 25.0, float64(c+1) * 25.0}, true
}

// ExpectedMarkCenterFromProbabilities computes the probability-weighted expected (cy, cx) across
// candidate Set-of-Marks boxes, excluding catchAllKey mass.
func ExpectedMarkCenterFromProbabilities(probs map[string]float64, marks []CandidateMark, catchAllKey string) (float64, float64, bool) {
	mass := 0.0
	sumY := 0.0
	sumX := 0.0
	for _, m := range marks {
		mid := strings.TrimSpace(m.ID)
		if mid == "" || mid == catchAllKey {
			continue
		}
		p := probs[mid]
		if !math.IsNaN(p) && !math.IsInf(p, 0) && p > 0 {
			mcy := (m.Box[0] + m.Box[2]) / 2.0
			mcx := (m.Box[1] + m.Box[3]) / 2.0
			mass += p
			sumY += p * mcy
			sumX += p * mcx
		}
	}
	if mass <= 1e-12 {
		return 0, 0, false
	}
	return sumY / mass, sumX / mass, true
}

// SafeExpectedMarkCenter computes the probability-weighted expected (cy, cx) across candidate Set-of-Marks boxes,
// snapping deterministically to the highest-probability candidate mark when spatial dispersion exceeds maxDispersion
// or when the centroid falls outside all candidate boxes (bimodal/non-convex safety guard).
func SafeExpectedMarkCenter(
	probs map[string]float64,
	marks []CandidateMark,
	catchAllKey string,
	maxDispersion float64,
) (float64, float64, CandidateMark, bool) {
	if maxDispersion <= 0 {
		maxDispersion = 18.0
	}
	type weightedMark struct {
		p, cy, cx, area float64
		mark            CandidateMark
	}
	var wms []weightedMark
	var validMarks []CandidateMark
	mass := 0.0
	for _, m := range marks {
		mid := strings.TrimSpace(m.ID)
		if mid == "" || mid == catchAllKey {
			continue
		}
		p := probs[mid]
		if !math.IsNaN(p) && !math.IsInf(p, 0) && p > 0 {
			mcy := (m.Box[0] + m.Box[2]) / 2.0
			mcx := (m.Box[1] + m.Box[3]) / 2.0
			area := math.Max(0, m.Box[2]-m.Box[0]) * math.Max(0, m.Box[3]-m.Box[1])
			wms = append(wms, weightedMark{p: p, cy: mcy, cx: mcx, area: area, mark: m})
			validMarks = append(validMarks, m)
			mass += p
		}
	}
	if mass <= 1e-12 || len(wms) == 0 {
		return 0, 0, CandidateMark{}, false
	}
	ey, ex := 0.0, 0.0
	for _, wm := range wms {
		ey += wm.p * wm.cy
		ex += wm.p * wm.cx
	}
	ey /= mass
	ex /= mass

	variance := 0.0
	for _, wm := range wms {
		dy := wm.cy - ey
		dx := wm.cx - ex
		variance += (wm.p / mass) * (dy*dy + dx*dx)
	}
	dispersion := math.Sqrt(math.Max(0, variance))
	containing, dist, ok := SelectContainingOrNearestMark(ey, ex, validMarks)
	if ok && dist == 0 && dispersion <= maxDispersion {
		return ey, ex, containing, true
	}
	best := wms[0]
	for _, wm := range wms[1:] {
		if wm.p > best.p || (wm.p == best.p && wm.area < best.area) {
			best = wm
		}
	}
	return best.cy, best.cx, best.mark, true
}

// PairwiseEdge represents an entity-resolution equivalence probability between NodeA and NodeB.
type PairwiseEdge struct {
	NodeA     string
	NodeB     string
	EquivProb float64
}

// TriangleContradiction records a 3-cycle violating transitivity.
type TriangleContradiction struct {
	Nodes    [3]string
	WeakEdge [2]string
	WeakProb float64
	Severity float64
}

// CheckTransitiveTriangles returns all 3-cycles where two edges >= highThresh and the third <= lowThresh.
func CheckTransitiveTriangles(edges []PairwiseEdge, highThresh, lowThresh float64) []TriangleContradiction {
	if highThresh <= 0 {
		highThresh = 0.80
	}
	if lowThresh <= 0 {
		lowThresh = 0.25
	}
	type pair struct{ u, v string }
	probs := make(map[pair]float64)
	counts := make(map[pair]int)
	nodeSet := make(map[string]bool)
	for _, e := range edges {
		u, v := strings.TrimSpace(e.NodeA), strings.TrimSpace(e.NodeB)
		if u == "" || v == "" || u == v {
			continue
		}
		if u > v {
			u, v = v, u
		}
		nodeSet[u] = true
		nodeSet[v] = true
		probs[pair{u, v}] += e.EquivProb
		counts[pair{u, v}]++
	}
	for k, c := range counts {
		probs[k] /= float64(c)
	}
	adj := make(map[string][]string, len(nodeSet))
	for k := range probs {
		adj[k.u] = append(adj[k.u], k.v)
	}
	for u := range adj {
		sort.Strings(adj[u])
	}
	nodes := make([]string, 0, len(nodeSet))
	for n := range nodeSet {
		nodes = append(nodes, n)
	}
	sort.Strings(nodes)

	var out []TriangleContradiction
	for _, a := range nodes {
		nbrs := adj[a]
		for i := 0; i < len(nbrs); i++ {
			b := nbrs[i]
			for j := i + 1; j < len(nbrs); j++ {
				c := nbrs[j]
				pBC, okBC := probs[pair{b, c}]
				if !okBC {
					continue
				}
				pAB := probs[pair{a, b}]
				pAC := probs[pair{a, c}]
				ps := [3]float64{pAB, pBC, pAC}
				es := [3][2]string{{a, b}, {b, c}, {a, c}}
				highCount, lowIdx, lowCount := 0, -1, 0
				minStrong := 1.0
				for idx, p := range ps {
					if p >= highThresh {
						highCount++
						if p < minStrong {
							minStrong = p
						}
					}
					if p <= lowThresh {
						lowCount++
						lowIdx = idx
					}
				}
				if highCount == 2 && lowCount == 1 {
					out = append(out, TriangleContradiction{
						Nodes:    [3]string{a, b, c},
						WeakEdge: es[lowIdx],
						WeakProb: ps[lowIdx],
						Severity: minStrong - ps[lowIdx],
					})
				}
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Severity != out[j].Severity {
			return out[i].Severity > out[j].Severity
		}
		return strings.Join(out[i].Nodes[:], "/") < strings.Join(out[j].Nodes[:], "/")
	})
	return out
}

// ProbabilisticTriangle records a continuous transitive contradiction score in [0, 1].
type ProbabilisticTriangle struct {
	Nodes              [3]string
	EdgeProbs          [3]float64
	WeakEdge           [2]string
	WeakProb           float64
	ContradictionScore float64
}

// ProbabilisticTriangleContradiction returns max(p_ij * p_jk * (1 - p_ik)) in [0, 1] and the
// contradicted edge index (0 for (a,b), 1 for (b,c), 2 for (a,c)).
func ProbabilisticTriangleContradiction(pAB, pBC, pAC float64) (float64, int) {
	clamp := func(v float64) float64 {
		if math.IsNaN(v) || v < 0 {
			return 0
		}
		if v > 1 {
			return 1
		}
		return v
	}
	p0, p1, p2 := clamp(pAB), clamp(pBC), clamp(pAC)
	cands := [3]float64{
		p1 * p2 * (1.0 - p0),
		p0 * p2 * (1.0 - p1),
		p0 * p1 * (1.0 - p2),
	}
	bestIdx := 0
	for i := 1; i < 3; i++ {
		if cands[i] > cands[bestIdx] {
			bestIdx = i
		}
	}
	return cands[bestIdx], bestIdx
}

// ScoreProbabilisticTriangles ranks 3-cycles by continuous soft contradiction score >= minScore.
func ScoreProbabilisticTriangles(edges []PairwiseEdge, minScore float64) []ProbabilisticTriangle {
	type pair struct{ u, v string }
	probs := make(map[pair]float64)
	counts := make(map[pair]int)
	nodeSet := make(map[string]bool)
	for _, e := range edges {
		u, v := strings.TrimSpace(e.NodeA), strings.TrimSpace(e.NodeB)
		if u == "" || v == "" || u == v {
			continue
		}
		if u > v {
			u, v = v, u
		}
		p := e.EquivProb
		if p < 0 {
			p = 0
		} else if p > 1 {
			p = 1
		}
		nodeSet[u] = true
		nodeSet[v] = true
		probs[pair{u, v}] += p
		counts[pair{u, v}]++
	}
	for k, c := range counts {
		probs[k] /= float64(c)
	}
	adj := make(map[string][]string, len(nodeSet))
	for k := range probs {
		adj[k.u] = append(adj[k.u], k.v)
	}
	for u := range adj {
		sort.Strings(adj[u])
	}
	nodes := make([]string, 0, len(nodeSet))
	for n := range nodeSet {
		nodes = append(nodes, n)
	}
	sort.Strings(nodes)

	var out []ProbabilisticTriangle
	for _, a := range nodes {
		nbrs := adj[a]
		for i := 0; i < len(nbrs); i++ {
			b := nbrs[i]
			for j := i + 1; j < len(nbrs); j++ {
				c := nbrs[j]
				pBC, okBC := probs[pair{b, c}]
				if !okBC {
					continue
				}
				pAB := probs[pair{a, b}]
				pAC := probs[pair{a, c}]
				score, weakIdx := ProbabilisticTriangleContradiction(pAB, pBC, pAC)
				if score < minScore {
					continue
				}
				ps := [3]float64{pAB, pBC, pAC}
				es := [3][2]string{{a, b}, {b, c}, {a, c}}
				out = append(out, ProbabilisticTriangle{
					Nodes:              [3]string{a, b, c},
					EdgeProbs:          ps,
					WeakEdge:           es[weakIdx],
					WeakProb:           ps[weakIdx],
					ContradictionScore: score,
				})
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].ContradictionScore != out[j].ContradictionScore {
			return out[i].ContradictionScore > out[j].ContradictionScore
		}
		return strings.Join(out[i].Nodes[:], "/") < strings.Join(out[j].Nodes[:], "/")
	})
	return out
}

var (
	reMultipackPrefix = regexp.MustCompile(`(?i)\b(\d+)\s*[x×]\s*\d+`)
	rePackOf          = regexp.MustCompile(`(?i)\b(?:pack\s+of|case\s+of|box\s+of)\s+(\d+)\b`)
	reHyphenPack      = regexp.MustCompile(`(?i)\b(\d+)\s*(?:-|\s)?\s*(?:pack|pk|count|ct|cans|bottles|bars|pods|pouches)\b`)
	reUnitMeasure     = regexp.MustCompile(`(?i)(?:\b\d+\s*[x×]\s*)?(\d+(?:\.\d+)?)\s*(ml|milliliters?|l|liters?|litres?|cl|g|grams?|kg|kilograms?|fl\s*oz|oz)\b`)
)

// UnitMeasure represents a normalized per-unit volume or mass from a catalog title.
type UnitMeasure struct {
	Value float64
	Unit  string
}

// ParsePackMultiplier extracts an integer multipack multiplier from a catalog title (default 1).
func ParsePackMultiplier(title string) int {
	for _, re := range []*regexp.Regexp{reMultipackPrefix, rePackOf, reHyphenPack} {
		if m := re.FindStringSubmatch(title); len(m) == 2 {
			if n, err := strconv.Atoi(m[1]); err == nil && n > 0 {
				return n
			}
		}
	}
	return 1
}

// ParseUnitMeasure extracts and normalizes per-unit volume or mass (e.g., 0.33L -> 330 ml, 1kg -> 1000 g).
func ParseUnitMeasure(title string) (UnitMeasure, bool) {
	m := reUnitMeasure.FindStringSubmatch(title)
	if len(m) != 3 {
		return UnitMeasure{}, false
	}
	val, err := strconv.ParseFloat(m[1], 64)
	if err != nil || val <= 0 || math.IsNaN(val) || math.IsInf(val, 0) {
		return UnitMeasure{}, false
	}
	rawUnit := strings.Join(strings.Fields(strings.ToLower(m[2])), " ")
	switch rawUnit {
	case "ml", "milliliter", "milliliters":
		return UnitMeasure{Value: val, Unit: "ml"}, true
	case "cl":
		return UnitMeasure{Value: val * 10.0, Unit: "ml"}, true
	case "l", "liter", "liters", "litre", "litres":
		return UnitMeasure{Value: val * 1000.0, Unit: "ml"}, true
	case "g", "gram", "grams":
		return UnitMeasure{Value: val, Unit: "g"}, true
	case "kg", "kilogram", "kilograms":
		return UnitMeasure{Value: val * 1000.0, Unit: "g"}, true
	case "fl oz", "floz":
		return UnitMeasure{Value: val, Unit: "fl_oz"}, true
	case "oz":
		return UnitMeasure{Value: val, Unit: "oz"}, true
	default:
		return UnitMeasure{}, false
	}
}

// ValidatePackQuantityConsistency returns false when multipack count or normalized unit volume/mass differs
// while the model claims an exact duplicate or identical_pack_and_volume.
func ValidatePackQuantityConsistency(titleA, titleB, predictedRelation, predictedPackMatch string) bool {
	if predictedRelation != "exact_duplicate" && predictedPackMatch != "identical_pack_and_volume" {
		return true
	}
	if ParsePackMultiplier(titleA) != ParsePackMultiplier(titleB) {
		return false
	}
	uA, okA := ParseUnitMeasure(titleA)
	uB, okB := ParseUnitMeasure(titleB)
	if okA && okB {
		if uA.Unit != uB.Unit {
			return false
		}
		denom := math.Max(math.Max(uA.Value, uB.Value), 1e-9)
		if math.Abs(uA.Value-uB.Value)/denom > 0.02 {
			return false
		}
	}
	return true
}

// ReasoningToReadoutDistillationResult holds the breakdown of the arXiv:2610.03935 Eq. (12) objective.
type ReasoningToReadoutDistillationResult struct {
	TotalLoss            float64
	CrossEntropyLoss     float64
	OrdinalScoreLoss     float64
	KLAnchorLoss         float64
	ExpectedStudentScore float64
	ExpectedTeacherScore float64
	DistilledPosterior   map[string]float64
}

// ReasoningToReadoutDistillationLoss evaluates Eq. (12) from Duan et al. (arXiv:2610.03935)
// and computes the closed-form stationary log-linear posterior combining teacher and base anchor readouts.
func ReasoningToReadoutDistillationLoss(
	teacherProbs map[string]float64,
	studentProbs map[string]float64,
	baseAnchorProbs map[string]float64,
	primitiveType string,
	taskWeight float64,
	ordinalLambda float64,
	klAnchorKappa float64,
	teacherTemp float64,
	anchorTemp float64,
) (ReasoningToReadoutDistillationResult, error) {
	if len(teacherProbs) == 0 || len(studentProbs) == 0 {
		return ReasoningToReadoutDistillationResult{}, fmt.Errorf("teacherProbs and studentProbs must be non-empty")
	}
	if len(teacherProbs) != len(studentProbs) {
		return ReasoningToReadoutDistillationResult{}, fmt.Errorf("teacherProbs and studentProbs key count mismatch")
	}
	if taskWeight <= 0 || math.IsNaN(taskWeight) || math.IsInf(taskWeight, 0) {
		return ReasoningToReadoutDistillationResult{}, fmt.Errorf("taskWeight must be finite and > 0")
	}
	if ordinalLambda < 0 || math.IsNaN(ordinalLambda) || math.IsInf(ordinalLambda, 0) {
		return ReasoningToReadoutDistillationResult{}, fmt.Errorf("ordinalLambda must be finite and >= 0")
	}
	if klAnchorKappa < 0 || math.IsNaN(klAnchorKappa) || math.IsInf(klAnchorKappa, 0) {
		return ReasoningToReadoutDistillationResult{}, fmt.Errorf("klAnchorKappa must be finite and >= 0")
	}
	if teacherTemp <= 0 || math.IsNaN(teacherTemp) || math.IsInf(teacherTemp, 0) {
		teacherTemp = 1.0
	}
	if anchorTemp <= 0 || math.IsNaN(anchorTemp) || math.IsInf(anchorTemp, 0) {
		anchorTemp = 1.0
	}

	keys := make([]string, 0, len(teacherProbs))
	for k := range teacherProbs {
		if _, ok := studentProbs[k]; !ok {
			return ReasoningToReadoutDistillationResult{}, fmt.Errorf("option key %q missing from studentProbs", k)
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)

	normMap := func(m map[string]float64, label string) (map[string]float64, error) {
		sum := 0.0
		clean := make(map[string]float64, len(keys))
		for _, k := range keys {
			v, ok := m[k]
			if !ok || math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
				return nil, fmt.Errorf("invalid probability for %q in %s", k, label)
			}
			clean[k] = v
			sum += v
		}
		if sum <= 0 {
			return nil, fmt.Errorf("%s sums to zero", label)
		}
		for _, k := range keys {
			p := clean[k] / sum
			if p < 1e-12 {
				p = 1e-12
			}
			clean[k] = p
		}
		return clean, nil
	}

	qBar, err := normMap(teacherProbs, "teacherProbs")
	if err != nil {
		return ReasoningToReadoutDistillationResult{}, err
	}
	pTheta, err := normMap(studentProbs, "studentProbs")
	if err != nil {
		return ReasoningToReadoutDistillationResult{}, err
	}
	p0 := pTheta
	if len(baseAnchorProbs) > 0 {
		p0, err = normMap(baseAnchorProbs, "baseAnchorProbs")
		if err != nil {
			return ReasoningToReadoutDistillationResult{}, err
		}
	}

	ceLoss := 0.0
	klAnchor := 0.0
	for _, k := range keys {
		ceLoss -= qBar[k] * math.Log(pTheta[k])
		klAnchor += p0[k] * math.Log(p0[k]/pTheta[k])
	}

	cleanType := strings.ToLower(strings.TrimSpace(primitiveType))
	isScore := cleanType == "score" || cleanType == "ordinal" || cleanType == "ordinal_scoring"
	expStudent := 0.0
	expTeacher := 0.0
	ordLoss := 0.0
	if isScore {
		for idx, k := range keys {
			val, parseErr := strconv.ParseFloat(k, 64)
			if parseErr != nil {
				val = float64(idx)
			}
			expStudent += val * pTheta[k]
			expTeacher += val * qBar[k]
		}
		diff := expStudent - expTeacher
		ordLoss = diff * diff
	}

	total := taskWeight*(ceLoss+(func() float64 {
		if isScore {
			return ordinalLambda * ordLoss
		}
		return 0.0
	})()) + klAnchorKappa*klAnchor

	denomW := taskWeight + klAnchorKappa
	alphaAnchor := 0.0
	if denomW > 0 {
		alphaAnchor = klAnchorKappa / denomW
	}
	logits := make(map[string]float64, len(keys))
	maxL := math.Inf(-1)
	for _, k := range keys {
		lg := (1.0-alphaAnchor)*(math.Log(qBar[k])/teacherTemp) + alphaAnchor*(math.Log(p0[k])/anchorTemp)
		logits[k] = lg
		if lg > maxL {
			maxL = lg
		}
	}
	sumExp := 0.0
	post := make(map[string]float64, len(keys))
	for _, k := range keys {
		e := math.Exp(logits[k] - maxL)
		post[k] = e
		sumExp += e
	}
	for _, k := range keys {
		post[k] /= sumExp
	}

	return ReasoningToReadoutDistillationResult{
		TotalLoss:            total,
		CrossEntropyLoss:     ceLoss,
		OrdinalScoreLoss:     ordLoss,
		KLAnchorLoss:         klAnchor,
		ExpectedStudentScore: expStudent,
		ExpectedTeacherScore: expTeacher,
		DistilledPosterior:   post,
	}, nil
}

// TrajectoryDelegationCalibrationResult holds Proposition 1 trajectory-calibrated deferral thresholds (arXiv:2610.21412v1).
type TrajectoryDelegationCalibrationResult struct {
	TauProp1Marginal      float64
	TauEmpiricalBin       float64
	TauIrreversibleAction float64
	TauReversibleAction   float64
	CostMarginK           float64
	MeanSystem1Accuracy   float64
	MeanLLMAccuracy       float64
	DelegatedCoverage     float64
}

// FitTrajectoryDelegationThreshold fits trajectory-level confidence deferral thresholds (arXiv:2610.21412v1 Prop. 1).
func FitTrajectoryDelegationThreshold(
	calibConfidences []float64,
	calibSystem1Correct []bool,
	calibLLMCorrect []bool,
	deltaCostUSD float64,
	errorStakesV float64,
	irreversibleStakesV float64,
	reversibleStakesV float64,
	minBinSize int,
	numBins int,
) (TrajectoryDelegationCalibrationResult, error) {
	n := len(calibConfidences)
	if n == 0 || len(calibSystem1Correct) != n {
		return TrajectoryDelegationCalibrationResult{}, fmt.Errorf("calibConfidences and calibSystem1Correct must be non-empty of equal length")
	}
	if len(calibLLMCorrect) > 0 && len(calibLLMCorrect) != n {
		return TrajectoryDelegationCalibrationResult{}, fmt.Errorf("calibLLMCorrect length mismatch")
	}
	if deltaCostUSD < 0 || math.IsNaN(deltaCostUSD) || math.IsInf(deltaCostUSD, 0) {
		return TrajectoryDelegationCalibrationResult{}, fmt.Errorf("deltaCostUSD must be finite and >= 0")
	}
	if errorStakesV <= 0 || irreversibleStakesV <= 0 || reversibleStakesV <= 0 {
		return TrajectoryDelegationCalibrationResult{}, fmt.Errorf("stakes V must be > 0")
	}

	type item struct {
		conf  float64
		s1OK  int
		llmOK int
	}
	paired := make([]item, n)
	sumS1 := 0
	sumLLM := 0
	for i := 0; i < n; i++ {
		c := calibConfidences[i]
		if math.IsNaN(c) || math.IsInf(c, 0) || c < 0 || c > 1.0+1e-6 {
			return TrajectoryDelegationCalibrationResult{}, fmt.Errorf("invalid confidence at %d: %v", i, c)
		}
		if c < 0 {
			c = 0
		} else if c > 1 {
			c = 1
		}
		s1 := 0
		if calibSystem1Correct[i] {
			s1 = 1
		}
		llm := 1
		if len(calibLLMCorrect) > 0 && !calibLLMCorrect[i] {
			llm = 0
		}
		paired[i] = item{conf: c, s1OK: s1, llmOK: llm}
		sumS1 += s1
		sumLLM += llm
	}

	meanS1 := float64(sumS1) / float64(n)
	meanLLM := float64(sumLLM) / float64(n)
	costMargin := deltaCostUSD / errorStakesV
	tauProp1 := math.Min(1.0, math.Max(0.0, meanLLM-costMargin))

	sort.Slice(paired, func(i, j int) bool { return paired[i].conf < paired[j].conf })
	if minBinSize < 1 {
		minBinSize = 1
	}
	if numBins < 1 {
		numBins = 10
	}
	step := n / numBins
	if step < minBinSize {
		step = minBinSize
	}
	var bins [][]item
	for i := 0; i < n; i += step {
		end := i + step
		if end > n {
			end = n
		}
		bins = append(bins, paired[i:end])
	}
	if len(bins) >= 2 && len(bins[len(bins)-1]) < maxInt(1, minBinSize/2) {
		bins[len(bins)-2] = append(bins[len(bins)-2], bins[len(bins)-1]...)
		bins = bins[:len(bins)-1]
	}

	projectThresh := func(stakesVal float64) float64 {
		marginVal := deltaCostUSD / stakesVal
		for bIdx := 0; bIdx < len(bins); bIdx++ {
			allPass := true
			for _, b := range bins[bIdx:] {
				s1C, llmC := 0, 0
				for _, x := range b {
					s1C += x.s1OK
					llmC += x.llmOK
				}
				gBin := float64(s1C) / float64(len(b))
				qBin := float64(llmC) / float64(len(b))
				if (gBin - qBin + marginVal) < -1e-12 {
					allPass = false
					break
				}
			}
			if allPass {
				return bins[bIdx][0].conf
			}
		}
		return 1.000001
	}

	tauEmp := projectThresh(errorStakesV)
	tauIrrev := projectThresh(irreversibleStakesV)
	tauRev := projectThresh(reversibleStakesV)
	delCount := 0
	for _, x := range paired {
		if x.conf >= tauEmp {
			delCount++
		}
	}

	return TrajectoryDelegationCalibrationResult{
		TauProp1Marginal:      tauProp1,
		TauEmpiricalBin:       tauEmp,
		TauIrreversibleAction: tauIrrev,
		TauReversibleAction:   tauRev,
		CostMarginK:           costMargin,
		MeanSystem1Accuracy:   meanS1,
		MeanLLMAccuracy:       meanLLM,
		DelegatedCoverage:     float64(delCount) / float64(n),
	}, nil
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// TriggerGuardUtilityResult holds Proposition 2 Trigger/Guard Youden-J resolution and utility metrics (arXiv:2610.21412v1 Eq. 5).
type TriggerGuardUtilityResult struct {
	PrevalencePi               float64
	TPR                        float64
	FPR                        float64
	YoudenJ                    float64
	Accuracy                   float64
	FiringRate                 float64
	OptimalTriggerThreshold    float64
	DeltaUtilityPerOpportunity float64
	DeltaUtilityTotal          float64
}

// EvaluateTriggerGuardUtility evaluates a binary Trigger/Guard gate via Youden's J (arXiv:2610.21412v1 Prop. 2 Eq. 5).
func EvaluateTriggerGuardUtility(
	needLabels []bool,
	gateDecisions []bool,
	benefitWhenNeeded float64,
	harmWhenUnneeded float64,
	moduleCost float64,
	queryCostKappa float64,
) (TriggerGuardUtilityResult, error) {
	n := len(needLabels)
	if n == 0 || len(gateDecisions) != n {
		return TriggerGuardUtilityResult{}, fmt.Errorf("needLabels and gateDecisions must be non-empty of equal length")
	}
	if (benefitWhenNeeded + harmWhenUnneeded) <= 0 {
		return TriggerGuardUtilityResult{}, fmt.Errorf("benefitWhenNeeded + harmWhenUnneeded must be > 0")
	}
	tp, fp, tn, fn := 0, 0, 0, 0
	for i := 0; i < n; i++ {
		if needLabels[i] && gateDecisions[i] {
			tp++
		} else if needLabels[i] && !gateDecisions[i] {
			fn++
		} else if !needLabels[i] && gateDecisions[i] {
			fp++
		} else {
			tn++
		}
	}
	tpr := float64(tp) / float64(maxInt(1, tp+fn))
	fpr := float64(fp) / float64(maxInt(1, fp+tn))
	youdenJ := tpr - fpr
	pi := float64(tp+fn) / float64(n)
	acc := float64(tp+tn) / float64(n)
	fireRate := float64(tp+fp) / float64(n)
	optTau := math.Min(1.0, math.Max(0.0, (moduleCost+harmWhenUnneeded)/(benefitWhenNeeded+harmWhenUnneeded)))
	duPerOpp := (benefitWhenNeeded+harmWhenUnneeded)*pi*(1.0-pi)*youdenJ - queryCostKappa

	return TriggerGuardUtilityResult{
		PrevalencePi:               pi,
		TPR:                        tpr,
		FPR:                        fpr,
		YoudenJ:                    youdenJ,
		Accuracy:                   acc,
		FiringRate:                 fireRate,
		OptimalTriggerThreshold:    optTau,
		DeltaUtilityPerOpportunity: duPerOpp,
		DeltaUtilityTotal:          float64(n) * duPerOpp,
	}, nil
}

// SelectiveBrierSFTResult holds the breakdown of arXiv:2610.21412v1 Section 5 Eq. (1) SFT loss.
type SelectiveBrierSFTResult struct {
	TotalLoss          float64
	CrossEntropyLoss   float64
	BrierLoss          float64
	TeacherAccepted    bool
	TargetDistribution map[string]float64
}

// SelectiveTeacherBrierSFTLoss computes the Selective Teacher + Brier SFT loss (arXiv:2610.21412v1 Sec. 5 Eq. 1).
func SelectiveTeacherBrierSFTLoss(
	teacherProbs map[string]float64,
	studentProbs map[string]float64,
	goldKey string,
	acceptableKeys []string,
	goldWeightAlpha float64,
	brierWeightMu float64,
) (SelectiveBrierSFTResult, error) {
	if len(teacherProbs) == 0 || len(studentProbs) == 0 || len(teacherProbs) != len(studentProbs) {
		return SelectiveBrierSFTResult{}, fmt.Errorf("teacherProbs and studentProbs must be non-empty with identical keys")
	}
	if _, ok := teacherProbs[goldKey]; !ok {
		return SelectiveBrierSFTResult{}, fmt.Errorf("goldKey %q not in teacherProbs", goldKey)
	}
	if goldWeightAlpha < 0 || goldWeightAlpha > 1 || brierWeightMu < 0 {
		return SelectiveBrierSFTResult{}, fmt.Errorf("invalid alpha or mu")
	}
	keys := make([]string, 0, len(teacherProbs))
	for k := range teacherProbs {
		if _, ok := studentProbs[k]; !ok {
			return SelectiveBrierSFTResult{}, fmt.Errorf("missing key %q in studentProbs", k)
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)

	norm := func(m map[string]float64) (map[string]float64, error) {
		sum := 0.0
		out := make(map[string]float64, len(keys))
		for _, k := range keys {
			v := m[k]
			if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
				return nil, fmt.Errorf("invalid prob for %q", k)
			}
			out[k] = v
			sum += v
		}
		if sum <= 0 {
			return nil, fmt.Errorf("distribution sums to zero")
		}
		for _, k := range keys {
			p := out[k] / sum
			if p < 1e-12 {
				p = 1e-12
			}
			out[k] = p
		}
		return out, nil
	}
	pT, err := norm(teacherProbs)
	if err != nil {
		return SelectiveBrierSFTResult{}, err
	}
	pS, err := norm(studentProbs)
	if err != nil {
		return SelectiveBrierSFTResult{}, err
	}

	accSet := map[string]bool{goldKey: true}
	for _, k := range acceptableKeys {
		accSet[k] = true
	}
	winKey := keys[0]
	for _, k := range keys[1:] {
		if pT[k] > pT[winKey] || (pT[k] == pT[winKey] && k > winKey) {
			winKey = k
		}
	}
	teacherOK := accSet[winKey]

	target := make(map[string]float64, len(keys))
	ceLoss := 0.0
	brierLoss := 0.0
	for _, k := range keys {
		oh := 0.0
		if k == goldKey {
			oh = 1.0
		}
		if teacherOK {
			target[k] = goldWeightAlpha*oh + (1.0-goldWeightAlpha)*pT[k]
		} else {
			target[k] = oh
		}
		ceLoss -= target[k] * math.Log(pS[k])
		diff := pS[k] - target[k]
		brierLoss += diff * diff
	}

	return SelectiveBrierSFTResult{
		TotalLoss:          ceLoss + brierWeightMu*brierLoss,
		CrossEntropyLoss:   ceLoss,
		BrierLoss:          brierLoss,
		TeacherAccepted:    teacherOK,
		TargetDistribution: target,
	}, nil
}

// NonInferiorityTestResult holds Tango's (1998) one-sided paired score non-inferiority test result (arXiv:2610.21412v1 App. E).
type NonInferiorityTestResult struct {
	Wins                  int
	Losses                int
	NTotal                int
	Margin                float64
	DeltaHat              float64
	TangoZ                float64
	PAsymptotic           float64
	PBergerBoos           float64
	ZeroDiscordanceLossUB float64
	NonInferior           bool
}

// TangoPairedNonInferiorityTest evaluates Tango's (1998) one-sided score test with Berger-Boos bound (arXiv:2610.21412v1 App. E).
func TangoPairedNonInferiorityTest(wins, losses, nTotal int, margin, alpha, gamma float64) (NonInferiorityTestResult, error) {
	if nTotal <= 0 || wins < 0 || losses < 0 || (wins+losses) > nTotal {
		return NonInferiorityTestResult{}, fmt.Errorf("invalid paired counts")
	}
	if margin <= 0 || margin >= 1 || alpha <= 0 || alpha >= 1 || gamma <= 0 || gamma >= 0.1 {
		return NonInferiorityTestResult{}, fmt.Errorf("invalid margin, alpha, or gamma")
	}
	d := wins + losses
	deltaHat := float64(wins-losses) / float64(nTotal)
	zeroUB := 1.0 - math.Pow(alpha, 1.0/float64(nTotal))

	if d == 0 {
		pNull := math.Pow(1.0-margin, float64(nTotal))
		pBB := math.Min(1.0, gamma+pNull)
		return NonInferiorityTestResult{
			Wins:                  wins,
			Losses:                losses,
			NTotal:                nTotal,
			Margin:                margin,
			DeltaHat:              0.0,
			TangoZ:                math.Inf(1),
			PAsymptotic:           0.0,
			PBergerBoos:           pBB,
			ZeroDiscordanceLossUB: zeroUB,
			NonInferior:           pBB <= alpha,
		}, nil
	}

	bestLL := -1e30
	bestQ := math.Max(margin+1e-6, math.Min(0.999999, float64(d)/float64(nTotal)))
	for i := 1; i < 2000; i++ {
		q := margin + (1.0-margin)*(float64(i)/2001.0)
		p10 := 0.5 * (q + margin)
		p01 := 0.5 * (q - margin)
		pSame := 1.0 - q
		if p10 <= 0 || p01 <= 0 || pSame <= 0 {
			continue
		}
		ll := float64(losses)*math.Log(p10) + float64(wins)*math.Log(p01) + float64(nTotal-d)*math.Log(pSame)
		if ll > bestLL {
			bestLL = ll
			bestQ = q
		}
	}
	varH0 := float64(nTotal) * math.Max(bestQ-margin*margin, 1e-12)
	zStat := (float64(wins-losses) + float64(nTotal)*margin) / math.Sqrt(varH0)
	pAsym := 0.5 * math.Erfc(zStat/math.Sqrt2)

	zGamma := 3.890592
	phatD := float64(d) / float64(nTotal)
	rad := zGamma*math.Sqrt(math.Max(phatD*(1.0-phatD)/float64(nTotal), 1e-9)) + 1.0/(2.0*float64(nTotal))
	qLo := math.Max(margin, phatD-rad)
	qHi := math.Min(1.0, phatD+rad)

	worstTail := 0.0
	for idx := 0; idx <= 50; idx++ {
		qCand := qLo + (qHi-qLo)*(float64(idx)/50.0)
		varCand := float64(nTotal) * math.Max(qCand-margin*margin, 1e-12)
		zCand := (float64(wins-losses) + float64(nTotal)*margin) / math.Sqrt(varCand)
		tailCand := 0.5 * math.Erfc(zCand/math.Sqrt2)
		if tailCand > worstTail {
			worstTail = tailCand
		}
	}
	pBB := math.Min(1.0, gamma+worstTail)
	return NonInferiorityTestResult{
		Wins:                  wins,
		Losses:                losses,
		NTotal:                nTotal,
		Margin:                margin,
		DeltaHat:              deltaHat,
		TangoZ:                zStat,
		PAsymptotic:           pAsym,
		PBergerBoos:           pBB,
		ZeroDiscordanceLossUB: zeroUB,
		NonInferior:           pBB <= alpha,
	}, nil
}


