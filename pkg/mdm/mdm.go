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
	MaxSlotsPerPass               = 8
	DefaultCosinePriorTemperature = 0.07
)

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
// according to gateMode ("always", "confident", "hesitant", "proportional", or "off").
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
	if gateMode == "off" || len(priorLogits) == 0 || priorWeight <= 0 || len(optKeys) <= 1 {
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
	nodes := make([]string, 0, len(nodeSet))
	for n := range nodeSet {
		nodes = append(nodes, n)
	}
	sort.Strings(nodes)

	var out []TriangleContradiction
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			for k := j + 1; k < len(nodes); k++ {
				a, b, c := nodes[i], nodes[j], nodes[k]
				pAB, ok1 := probs[pair{a, b}]
				pBC, ok2 := probs[pair{b, c}]
				pAC, ok3 := probs[pair{a, c}]
				if !ok1 || !ok2 || !ok3 {
					continue
				}
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
	nodes := make([]string, 0, len(nodeSet))
	for n := range nodeSet {
		nodes = append(nodes, n)
	}
	sort.Strings(nodes)

	var out []ProbabilisticTriangle
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			for k := j + 1; k < len(nodes); k++ {
				a, b, c := nodes[i], nodes[j], nodes[k]
				pAB, ok1 := probs[pair{a, b}]
				pBC, ok2 := probs[pair{b, c}]
				pAC, ok3 := probs[pair{a, c}]
				if !ok1 || !ok2 || !ok3 {
					continue
				}
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
)

// ParsePackMultiplier extracts an integer multipack multiplier from a catalog title (default 1).
func ParsePackMultiplier(title string) int {
	for _, re := range []*regexp.Regexp{reMultipackPrefix, rePackOf} {
		if m := re.FindStringSubmatch(title); len(m) == 2 {
			if n, err := strconv.Atoi(m[1]); err == nil && n > 0 {
				return n
			}
		}
	}
	return 1
}
