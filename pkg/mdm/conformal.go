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
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"time"
)

const conformalTieTol = 1e-12
const maxSeededConformalStreams = 256

var (
	conformalRNGMu      sync.Mutex
	defaultConformalRNG = rand.New(rand.NewSource(time.Now().UnixNano()))
	seededConformalRNGs = make(map[int64]*rand.Rand)
	seededConformalKeys []int64
)

var defaultConformalAbstainKeys = []string{
	"0",
	"none",
	"none_of_the_above",
	"none_visible",
	"oos",
	"abstain",
}

// ConformalConfig parameterizes Split Conformal Prediction across LAC, APS, and RAPS.
type ConformalConfig struct {
	Method      string     // "lac", "aps", or "raps" (default "raps")
	LamReg      float64    // RAPS rank penalty >= 0 (default 0.01 when zero-value ConformalConfig{} is passed)
	KReg        int        // RAPS unpenalized top-k ranks >= 0 (default 1 when zero-value ConformalConfig{} is passed)
	Randomized  bool       // whether to draw u ~ Uniform(0, 1) instead of deterministic u = 0
	Seed        int64      // deterministic RNG seed when Randomized is true (0 draws from shared stream)
	RNG         *rand.Rand // optional caller-managed RNG stream when Randomized is true
	AbstainKeys []string   // custom abstain/catch-all keys; defaults to defaultConformalAbstainKeys when nil
}

// ConformalSet holds the calibrated prediction set C_alpha(x) and pre-fusion cascade gate flags.
type ConformalSet struct {
	PredictionSet   []string
	QHat            float64
	Singleton       bool
	ContainsAbstain bool
	ShouldEscalate  bool
	Method          string
}

// ContextBlock describes an agent trajectory block candidate for MCKP token-budget pruning.
type ContextBlock struct {
	ID         string
	FullTokens int
	StubTokens int
	FullText   string
	StubText   string
	Pinned     bool
}

// ContextPruneResult reports exact MCKP utility-maximizing retention choices under a token budget.
type ContextPruneResult struct {
	Decisions           map[string]string
	TokensBefore        int
	TokensAfter         int
	TokenReductionRatio float64
	ExpectedUtility     float64
}

func storeSeededConformalRNG(seed int64, r *rand.Rand) {
	conformalRNGMu.Lock()
	defer conformalRNGMu.Unlock()
	if _, exists := seededConformalRNGs[seed]; !exists {
		seededConformalKeys = append(seededConformalKeys, seed)
		if len(seededConformalKeys) > maxSeededConformalStreams {
			evict := seededConformalKeys[0]
			seededConformalKeys = seededConformalKeys[1:]
			delete(seededConformalRNGs, evict)
		}
	}
	seededConformalRNGs[seed] = r
}

func partitionConformalTieGroups(ranked []string, norm map[string]float64) [][]string {
	groups := make([][]string, 0, len(ranked))
	for _, k := range ranked {
		pk := norm[k]
		if len(groups) == 0 || math.Abs(pk-norm[groups[len(groups)-1][0]]) > conformalTieTol {
			groups = append(groups, []string{k})
		} else {
			last := len(groups) - 1
			groups[last] = append(groups[last], k)
		}
	}
	return groups
}

func drawWithRand(r *rand.Rand, groups [][]string) (float64, []string) {
	u := r.Float64()
	flat := make([]string, 0, len(groups)*2)
	for _, grp := range groups {
		if len(grp) > 1 {
			cp := make([]string, len(grp))
			copy(cp, grp)
			r.Shuffle(len(cp), func(i, j int) { cp[i], cp[j] = cp[j], cp[i] })
			flat = append(flat, cp...)
		} else {
			flat = append(flat, grp[0])
		}
	}
	return u, flat
}

func drawRandomizedOrderAndU(groups [][]string, cfg ConformalConfig) (float64, []string) {
	if cfg.RNG != nil {
		return drawWithRand(cfg.RNG, groups)
	}
	conformalRNGMu.Lock()
	defer conformalRNGMu.Unlock()
	if cfg.Seed == 0 {
		return drawWithRand(defaultConformalRNG, groups)
	}
	r, ok := seededConformalRNGs[cfg.Seed]
	if !ok {
		r = rand.New(rand.NewSource(cfg.Seed))
		seededConformalKeys = append(seededConformalKeys, cfg.Seed)
		if len(seededConformalKeys) > maxSeededConformalStreams {
			evict := seededConformalKeys[0]
			seededConformalKeys = seededConformalKeys[1:]
			delete(seededConformalRNGs, evict)
		}
		seededConformalRNGs[cfg.Seed] = r
	}
	return drawWithRand(r, groups)
}

func resolveConformalConfig(cfg ConformalConfig) (ConformalConfig, error) {
	resolved := cfg
	if strings.TrimSpace(resolved.Method) == "" {
		resolved.Method = "raps"
		if resolved.LamReg == 0 && resolved.KReg == 0 {
			resolved.LamReg = 0.01
			resolved.KReg = 1
		}
	} else {
		resolved.Method = strings.ToLower(strings.TrimSpace(resolved.Method))
	}

	switch resolved.Method {
	case "lac", "aps", "raps":
	default:
		return ConformalConfig{}, fmt.Errorf("mdm: unsupported conformal method %q", cfg.Method)
	}
	if math.IsNaN(resolved.LamReg) || math.IsInf(resolved.LamReg, 0) || resolved.LamReg < 0 {
		return ConformalConfig{}, fmt.Errorf("mdm: invalid LamReg %v", resolved.LamReg)
	}
	if resolved.KReg < 0 {
		return ConformalConfig{}, fmt.Errorf("mdm: KReg must be >= 0, got %d", resolved.KReg)
	}
	if resolved.AbstainKeys == nil {
		resolved.AbstainKeys = defaultConformalAbstainKeys
	}
	return resolved, nil
}

func normalizeAndRankConformalProbs(probs map[string]float64) ([]string, map[string]float64, error) {
	if len(probs) == 0 {
		return nil, nil, errors.New("mdm: probability map must not be empty")
	}
	clean := make(map[string]float64, len(probs))
	tot := 0.0
	for k, v := range probs {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			return nil, nil, fmt.Errorf("mdm: invalid probability for %q: %v", k, v)
		}
		clean[k] = v
		tot += v
	}
	if tot <= 0 {
		return nil, nil, errors.New("mdm: probability map sum must be > 0")
	}
	norm := make(map[string]float64, len(clean))
	ranked := make([]string, 0, len(clean))
	for k, v := range clean {
		norm[k] = v / tot
		ranked = append(ranked, k)
	}
	sort.Slice(ranked, func(i, j int) bool {
		pi, pj := norm[ranked[i]], norm[ranked[j]]
		if pi != pj {
			return pi > pj
		}
		return ranked[i] < ranked[j]
	})
	return ranked, norm, nil
}

func computeDeterministicConformalScores(
	groups [][]string,
	norm map[string]float64,
	cfg ConformalConfig,
) map[string]float64 {
	scores := make(map[string]float64)
	cumStrictlyGreater := 0.0
	strictCount := 0
	for _, grp := range groups {
		pGrp := norm[grp[0]]
		strictRank := strictCount + 1
		var grpScore float64
		switch cfg.Method {
		case "lac":
			grpScore = 1.0 - pGrp
		case "aps":
			grpScore = cumStrictlyGreater
		default:
			pen := 0.0
			if strictRank > cfg.KReg {
				pen = cfg.LamReg * float64(strictRank-cfg.KReg)
			}
			grpScore = cumStrictlyGreater + pen
		}
		for _, k := range grp {
			scores[k] = grpScore
		}
		cumStrictlyGreater += pGrp * float64(len(grp))
		strictCount += len(grp)
	}
	return scores
}

func computeRandomizedConformalScores(
	ordered []string,
	norm map[string]float64,
	cfg ConformalConfig,
	uDraw float64,
) map[string]float64 {
	scores := make(map[string]float64, len(ordered))
	cumBefore := 0.0
	for idx, k := range ordered {
		rank1 := idx + 1
		pk := norm[k]
		switch cfg.Method {
		case "lac":
			scores[k] = 1.0 - pk
		case "aps":
			scores[k] = cumBefore + uDraw*pk
		default:
			pen := 0.0
			if rank1 > cfg.KReg {
				pen = cfg.LamReg * float64(rank1-cfg.KReg)
			}
			scores[k] = cumBefore + uDraw*pk + pen
		}
		cumBefore += pk
	}
	return scores
}

// CalibrateConformalQuantile computes the finite-sample split conformal quantile q_hat at
// miscoverage level alpha in (0, 1) using the ceil((n+1)*(1-alpha))/n empirical order statistic.
// When ceil((n+1)*(1-alpha)) > n, +Inf is returned so prediction sets include the full label space.
func CalibrateConformalQuantile(
	calibProbs []map[string]float64,
	calibLabels []string,
	alpha float64,
	cfg ConformalConfig,
) (float64, error) {
	if math.IsNaN(alpha) || math.IsInf(alpha, 0) || alpha <= 0 || alpha >= 1 {
		return 0, fmt.Errorf("mdm: alpha must lie strictly in (0, 1), got %v", alpha)
	}
	if len(calibProbs) == 0 || len(calibProbs) != len(calibLabels) {
		return 0, errors.New("mdm: calibProbs and calibLabels must be non-empty slices of equal length")
	}
	resolved, err := resolveConformalConfig(cfg)
	if err != nil {
		return 0, err
	}

	var calRNG *rand.Rand
	useGlobal := false
	if resolved.Randomized {
		if resolved.RNG != nil {
			calRNG = resolved.RNG
		} else if resolved.Seed != 0 {
			calRNG = rand.New(rand.NewSource(resolved.Seed))
		} else {
			useGlobal = true
		}
	}

	scores := make([]float64, len(calibProbs))
	for i, dist := range calibProbs {
		ranked, norm, nErr := normalizeAndRankConformalProbs(dist)
		if nErr != nil {
			return 0, fmt.Errorf("mdm: calibration item %d: %w", i, nErr)
		}
		gold := calibLabels[i]
		if _, ok := norm[gold]; !ok {
			return 0, fmt.Errorf("mdm: calibration label %q at index %d missing from probabilities", gold, i)
		}
		groups := partitionConformalTieGroups(ranked, norm)
		var itemScores map[string]float64
		if !resolved.Randomized {
			itemScores = computeDeterministicConformalScores(groups, norm, resolved)
		} else if useGlobal {
			conformalRNGMu.Lock()
			uVal, ordered := drawWithRand(defaultConformalRNG, groups)
			conformalRNGMu.Unlock()
			itemScores = computeRandomizedConformalScores(ordered, norm, resolved, uVal)
		} else {
			uVal, ordered := drawWithRand(calRNG, groups)
			itemScores = computeRandomizedConformalScores(ordered, norm, resolved, uVal)
		}
		scores[i] = itemScores[gold]
	}

	if resolved.Randomized && resolved.RNG == nil && resolved.Seed != 0 && calRNG != nil {
		storeSeededConformalRNG(resolved.Seed, calRNG)
	}

	sort.Float64s(scores)
	n := len(scores)
	orderIdx := int(math.Ceil(float64(n+1)*(1.0-alpha) - 1e-12))
	if orderIdx > n {
		return math.Inf(1), nil
	}
	if orderIdx < 1 {
		orderIdx = 1
	}
	return scores[orderIdx-1], nil
}

// ConformalPredictionSet constructs C_alpha(x) = {y : s(x, y) <= qHat} ordered by descending probability
// and sets ShouldEscalate whenever |C_alpha(x)| != 1 or any abstain key belongs to C_alpha(x).
func ConformalPredictionSet(
	probs map[string]float64,
	qHat float64,
	cfg ConformalConfig,
) (ConformalSet, error) {
	if math.IsNaN(qHat) || qHat < 0 {
		return ConformalSet{}, fmt.Errorf("mdm: qHat must be >= 0 and non-NaN, got %v", qHat)
	}
	resolved, err := resolveConformalConfig(cfg)
	if err != nil {
		return ConformalSet{}, err
	}
	ranked, norm, err := normalizeAndRankConformalProbs(probs)
	if err != nil {
		return ConformalSet{}, err
	}

	groups := partitionConformalTieGroups(ranked, norm)
	var scores map[string]float64
	if !resolved.Randomized {
		scores = computeDeterministicConformalScores(groups, norm, resolved)
	} else {
		uVal, ordered := drawRandomizedOrderAndU(groups, resolved)
		scores = computeRandomizedConformalScores(ordered, norm, resolved, uVal)
	}

	predSet := make([]string, 0, len(ranked))
	if math.IsInf(qHat, 1) {
		predSet = append(predSet, ranked...)
	} else {
		for _, k := range ranked {
			if scores[k] <= qHat+1e-12 {
				predSet = append(predSet, k)
			}
		}
	}

	abstainLookup := make(map[string]bool, len(resolved.AbstainKeys))
	for _, ak := range resolved.AbstainKeys {
		abstainLookup[ak] = true
	}
	containsAbstain := false
	for _, k := range predSet {
		if abstainLookup[k] {
			containsAbstain = true
			break
		}
	}
	singleton := len(predSet) == 1
	return ConformalSet{
		PredictionSet:   predSet,
		QHat:            qHat,
		Singleton:       singleton,
		ContainsAbstain: containsAbstain,
		ShouldEscalate:  (!singleton) || containsAbstain,
		Method:          resolved.Method,
	}, nil
}

type pruneDPState struct {
	tokens  int
	utility float64
	choices []string
}

func compareChoiceSlices(a, b []string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] < b[i] {
			return -1
		}
		if a[i] > b[i] {
			return 1
		}
	}
	if len(a) < len(b) {
		return -1
	}
	if len(a) > len(b) {
		return 1
	}
	return 0
}

func paretoFilterStates(states []pruneDPState, maxStates int) []pruneDPState {
	sort.Slice(states, func(i, j int) bool {
		if states[i].tokens != states[j].tokens {
			return states[i].tokens < states[j].tokens
		}
		if states[i].utility != states[j].utility {
			return states[i].utility > states[j].utility
		}
		return compareChoiceSlices(states[i].choices, states[j].choices) < 0
	})

	pareto := make([]pruneDPState, 0, len(states))
	bestU := -1.0
	for _, st := range states {
		if st.utility > bestU+1e-12 {
			pareto = append(pareto, st)
			bestU = st.utility
		}
	}
	if len(pareto) <= maxStates {
		return pareto
	}

	minT := pareto[0].tokens
	maxT := pareto[len(pareto)-1].tokens
	if maxT <= minT {
		return pareto[:1]
	}
	span := float64(maxT - minT)
	buckets := make([]pruneDPState, maxStates)
	hasBucket := make([]bool, maxStates)
	for _, st := range pareto {
		bIdx := int((float64(st.tokens-minT) / span) * float64(maxStates-1))
		if bIdx >= maxStates {
			bIdx = maxStates - 1
		}
		if !hasBucket[bIdx] || st.utility > buckets[bIdx].utility || (st.utility == buckets[bIdx].utility && st.tokens < buckets[bIdx].tokens) {
			buckets[bIdx] = st
			hasBucket[bIdx] = true
		}
	}
	compacted := make([]pruneDPState, 0, maxStates)
	for i := 0; i < maxStates; i++ {
		if hasBucket[i] {
			compacted = append(compacted, buckets[i])
		}
	}
	return compacted
}

// PruneAgentContextBlocks solves the Multiple-Choice Knapsack Problem (MCKP) over {"full", "stub", "drop"}
// for agent context blocks under tokenBudget while preserving pinned blocks (minimum action = "stub").
func PruneAgentContextBlocks(
	blocks []ContextBlock,
	slotProbs map[string]map[string]float64,
	tokenBudget int,
) (ContextPruneResult, error) {
	if tokenBudget < 0 {
		return ContextPruneResult{}, fmt.Errorf("mdm: tokenBudget must be >= 0, got %d", tokenBudget)
	}
	if len(blocks) == 0 {
		return ContextPruneResult{
			Decisions:           map[string]string{},
			TokensBefore:        0,
			TokensAfter:         0,
			TokenReductionRatio: 0,
			ExpectedUtility:     0,
		}, nil
	}

	seen := make(map[string]bool, len(blocks))
	minPinned := 0
	tokensBefore := 0
	for _, b := range blocks {
		if b.ID == "" {
			return ContextPruneResult{}, errors.New("mdm: ContextBlock.ID must be non-empty")
		}
		if seen[b.ID] {
			return ContextPruneResult{}, fmt.Errorf("mdm: duplicate ContextBlock.ID %q", b.ID)
		}
		seen[b.ID] = true
		if b.StubTokens < 0 || b.FullTokens < b.StubTokens {
			return ContextPruneResult{}, fmt.Errorf("mdm: block %q requires FullTokens >= StubTokens >= 0", b.ID)
		}
		if _, ok := slotProbs[b.ID]; !ok {
			return ContextPruneResult{}, fmt.Errorf("mdm: missing slotProbs for block %q", b.ID)
		}
		if b.Pinned {
			minPinned += b.StubTokens
		}
		tokensBefore += b.FullTokens
	}
	if tokenBudget < minPinned {
		return ContextPruneResult{}, fmt.Errorf(
			"mdm: tokenBudget (%d) is below minimum pinned stub token sum (%d)",
			tokenBudget,
			minPinned,
		)
	}

	frontier := []pruneDPState{{tokens: 0, utility: 0, choices: nil}}
	actions := [3]string{"full", "stub", "drop"}

	for _, b := range blocks {
		dist := slotProbs[b.ID]
		var vals [3]float64
		tot := 0.0
		for i, act := range actions {
			v := dist[act]
			if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
				return ContextPruneResult{}, fmt.Errorf("mdm: invalid probability for block %q action %q: %v", b.ID, act, v)
			}
			vals[i] = v
			tot += v
		}
		if tot <= 0 {
			return ContextPruneResult{}, fmt.Errorf("mdm: probabilities for block %q sum to zero", b.ID)
		}
		pFull := vals[0] / tot
		pStub := vals[1] / tot
		uFull := 1.0*pFull + 0.5*pStub
		uStub := 0.6*pFull + 1.0*pStub

		type actOpt struct {
			name string
			cost int
			util float64
		}
		opts := []actOpt{
			{name: "full", cost: b.FullTokens, util: uFull},
			{name: "stub", cost: b.StubTokens, util: uStub},
		}
		if !b.Pinned {
			opts = append(opts, actOpt{name: "drop", cost: 0, util: 0.0})
		}

		expanded := make([]pruneDPState, 0, len(frontier)*len(opts))
		for _, st := range frontier {
			for _, op := range opts {
				nt := st.tokens + op.cost
				if nt <= tokenBudget {
					nextChoices := make([]string, len(st.choices)+1)
					copy(nextChoices, st.choices)
					nextChoices[len(st.choices)] = op.name
					expanded = append(expanded, pruneDPState{
						tokens:  nt,
						utility: st.utility + op.util,
						choices: nextChoices,
					})
				}
			}
		}
		if len(expanded) == 0 {
			return ContextPruneResult{}, fmt.Errorf("mdm: no feasible pruning assignment within tokenBudget=%d", tokenBudget)
		}
		frontier = paretoFilterStates(expanded, 4096)
	}

	best := frontier[0]
	for _, st := range frontier[1:] {
		uDiff := math.Round(st.utility*1e12) - math.Round(best.utility*1e12)
		if uDiff > 0 || (uDiff == 0 && st.tokens < best.tokens) || (uDiff == 0 && st.tokens == best.tokens && compareChoiceSlices(st.choices, best.choices) > 0) {
			best = st
		}
	}

	decisions := make(map[string]string, len(blocks))
	for i, b := range blocks {
		decisions[b.ID] = best.choices[i]
	}
	reduction := 0.0
	if tokensBefore > 0 {
		reduction = float64(tokensBefore-best.tokens) / float64(tokensBefore)
	}
	return ContextPruneResult{
		Decisions:           decisions,
		TokensBefore:        tokensBefore,
		TokensAfter:         best.tokens,
		TokenReductionRatio: reduction,
		ExpectedUtility:     best.utility,
	}, nil
}
