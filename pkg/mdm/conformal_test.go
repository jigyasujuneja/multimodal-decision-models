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
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestCalibrateConformalAndExchangeableCoverage(t *testing.T) {
	rng := rand.New(rand.NewSource(20261025))
	const (
		nCalib  = 400
		nTest   = 1200
		numOpts = 8
	)
	keys := make([]string, numOpts)
	for k := 0; k < numOpts; k++ {
		keys[k] = fmt.Sprintf("opt_%d", k)
	}

	drawSample := func() (map[string]float64, string) {
		logits := make([]float64, numOpts)
		maxL := math.Inf(-1)
		for k := 0; k < numOpts; k++ {
			logits[k] = rng.NormFloat64() * 1.35
			if logits[k] > maxL {
				maxL = logits[k]
			}
		}
		trueProbs := make([]float64, numOpts)
		sumExp := 0.0
		for k := 0; k < numOpts; k++ {
			trueProbs[k] = math.Exp(logits[k] - maxL)
			sumExp += trueProbs[k]
		}
		u := rng.Float64() * sumExp
		cum := 0.0
		goldIdx := numOpts - 1
		for k := 0; k < numOpts; k++ {
			cum += trueProbs[k]
			if u <= cum {
				goldIdx = k
				break
			}
		}
		// Miscalibrated noisy model scores over the same exchangeable generating process
		modelProbs := make(map[string]float64, numOpts)
		mSum := 0.0
		for k := 0; k < numOpts; k++ {
			noisy := math.Exp((logits[k]*1.6 + rng.NormFloat64()*0.35) - maxL)
			modelProbs[keys[k]] = noisy
			mSum += noisy
		}
		for k := range modelProbs {
			modelProbs[k] /= mSum
		}
		return modelProbs, keys[goldIdx]
	}

	calibProbs := make([]map[string]float64, nCalib)
	calibLabels := make([]string, nCalib)
	for i := 0; i < nCalib; i++ {
		calibProbs[i], calibLabels[i] = drawSample()
	}
	testProbs := make([]map[string]float64, nTest)
	testLabels := make([]string, nTest)
	for i := 0; i < nTest; i++ {
		testProbs[i], testLabels[i] = drawSample()
	}

	for _, method := range []string{"lac", "aps", "raps"} {
		for _, alpha := range []float64{0.05, 0.10} {
			cfg := ConformalConfig{Method: method, LamReg: 0.02, KReg: 1}
			qHat, err := CalibrateConformalQuantile(calibProbs, calibLabels, alpha, cfg)
			if err != nil {
				t.Fatalf("CalibrateConformalQuantile(%s, alpha=%v) failed: %v", method, alpha, err)
			}
			hits := 0
			for i := 0; i < nTest; i++ {
				res, sErr := ConformalPredictionSet(testProbs[i], qHat, cfg)
				if sErr != nil {
					t.Fatalf("ConformalPredictionSet failed: %v", sErr)
				}
				for _, pred := range res.PredictionSet {
					if pred == testLabels[i] {
						hits++
						break
					}
				}
			}
			cov := float64(hits) / float64(nTest)
			minTarget := 1.0 - alpha - 0.025
			if cov < minTarget {
				t.Fatalf("method=%s alpha=%v empirical coverage %.4f < %.4f", method, alpha, cov, minTarget)
			}
		}
	}
}

func TestConformalEdgeCasesAndAbstainEscalation(t *testing.T) {
	// Small n where ceil((n+1)*(1-alpha)) > n returns +Inf and full label set
	smallProbs := []map[string]float64{
		{"a": 0.7, "b": 0.2, "none_of_the_above": 0.1},
		{"a": 0.6, "b": 0.3, "none_of_the_above": 0.1},
	}
	smallLabels := []string{"a", "b"}
	qInf, err := CalibrateConformalQuantile(smallProbs, smallLabels, 0.05, ConformalConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !math.IsInf(qInf, 1) {
		t.Fatalf("expected +Inf when ceil((n+1)*(1-alpha)) > n, got %v", qInf)
	}
	fullSet, err := ConformalPredictionSet(smallProbs[0], qInf, ConformalConfig{})
	if err != nil {
		t.Fatalf("unexpected error on +Inf qHat: %v", err)
	}
	if len(fullSet.PredictionSet) != 3 || !fullSet.ContainsAbstain || !fullSet.ShouldEscalate {
		t.Fatalf("expected full set of 3 with abstain escalation, got %+v", fullSet)
	}

	// Confident singleton non-abstain does NOT escalate under deterministic RAPS
	confident, err := ConformalPredictionSet(
		map[string]float64{"sku_1": 0.96, "sku_2": 0.03, "none_of_the_above": 0.01},
		0.20,
		ConformalConfig{Method: "raps", LamReg: 0.01, KReg: 1},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !confident.Singleton || confident.ContainsAbstain || confident.ShouldEscalate {
		t.Fatalf("expected confident singleton without escalation, got %+v", confident)
	}
	if confident.PredictionSet[0] != "sku_1" {
		t.Fatalf("expected sku_1, got %v", confident.PredictionSet)
	}

	// Singleton abstain key DOES trigger escalation
	abstainSing, err := ConformalPredictionSet(
		map[string]float64{"none_of_the_above": 0.94, "sku_1": 0.04, "sku_2": 0.02},
		0.20,
		ConformalConfig{Method: "raps", LamReg: 0.01, KReg: 1},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !abstainSing.Singleton || !abstainSing.ContainsAbstain || !abstainSing.ShouldEscalate {
		t.Fatalf("expected singleton abstain to escalate, got %+v", abstainSing)
	}

	// Input contract rejections
	if _, err := CalibrateConformalQuantile(smallProbs, smallLabels, 0.0, ConformalConfig{}); err == nil {
		t.Fatal("expected error for alpha=0")
	}
	if _, err := ConformalPredictionSet(map[string]float64{"a": math.NaN()}, 0.2, ConformalConfig{}); err == nil {
		t.Fatal("expected error for NaN probability")
	}
}

func TestPruneAgentContextBlocksExactMCKP(t *testing.T) {
	blocks := []ContextBlock{
		{ID: "sys_rules", FullTokens: 200, StubTokens: 80, Pinned: true},
		{ID: "active_diff", FullTokens: 300, StubTokens: 90, Pinned: false},
		{ID: "stale_grep", FullTokens: 450, StubTokens: 60, Pinned: false},
		{ID: "failing_test", FullTokens: 250, StubTokens: 70, Pinned: true},
	}
	slotProbs := map[string]map[string]float64{
		"sys_rules":    {"full": 0.25, "stub": 0.70, "drop": 0.05},
		"active_diff":  {"full": 0.92, "stub": 0.06, "drop": 0.02},
		"stale_grep":   {"full": 0.03, "stub": 0.07, "drop": 0.90},
		"failing_test": {"full": 0.88, "stub": 0.10, "drop": 0.02},
	}

	res, err := PruneAgentContextBlocks(blocks, slotProbs, 650)
	if err != nil {
		t.Fatalf("PruneAgentContextBlocks failed: %v", err)
	}
	if res.TokensAfter > 650 {
		t.Fatalf("exceeded budget: %d > 650", res.TokensAfter)
	}
	if res.Decisions["sys_rules"] == "drop" || res.Decisions["failing_test"] == "drop" {
		t.Fatalf("pinned block was dropped: %+v", res.Decisions)
	}
	if res.Decisions["active_diff"] != "full" || res.Decisions["failing_test"] != "full" || res.Decisions["sys_rules"] != "stub" || res.Decisions["stale_grep"] != "drop" {
		t.Fatalf("unexpected MCKP optimum: %+v", res.Decisions)
	}
	if res.TokensBefore != 1200 || res.TokensAfter != 630 {
		t.Fatalf("unexpected token accounting before=%d after=%d", res.TokensBefore, res.TokensAfter)
	}

	// Budget below minimum pinned stub sum (80 + 70 = 150) fails closed
	if _, err := PruneAgentContextBlocks(blocks, slotProbs, 140); err == nil {
		t.Fatal("expected error when tokenBudget < pinned stub sum")
	}
}

func TestConformalTieSymmetryAndStatefulRandomizedCoverage(t *testing.T) {
	// 1. Exact 4-way uniform tie at qHat=0 must include all 4 tied classes symmetrically (never collapse to 1)
	unifA := map[string]float64{"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
	unifZ := map[string]float64{"z_3": 0.25, "z_1": 0.25, "z_2": 0.25, "z_0": 0.25}
	for _, method := range []string{"aps", "raps"} {
		cfg := ConformalConfig{Method: method, LamReg: 0.05, KReg: 1}
		resA, err := ConformalPredictionSet(unifA, 0.0, cfg)
		if err != nil {
			t.Fatalf("ConformalPredictionSet(%s) failed: %v", method, err)
		}
		resZ, err := ConformalPredictionSet(unifZ, 0.0, cfg)
		if err != nil {
			t.Fatalf("ConformalPredictionSet(%s) failed: %v", method, err)
		}
		if len(resA.PredictionSet) != 4 || !resA.ShouldEscalate || resA.Singleton {
			t.Fatalf("expected all 4 tied options with ShouldEscalate=true for %s, got %+v", method, resA)
		}
		if len(resZ.PredictionSet) != 4 || !resZ.ShouldEscalate {
			t.Fatalf("expected label-rename invariant 4-element set for %s, got %+v", method, resZ)
		}

		// Point mass [1, 0, 0] at qHat=0 must still yield singleton ["A"] without zero-prob tail inflation
		ptRes, err := ConformalPredictionSet(map[string]float64{"A": 1.0, "B": 0.0, "C": 0.0}, 0.0, cfg)
		if err != nil {
			t.Fatalf("ConformalPredictionSet point mass failed: %v", err)
		}
		if len(ptRes.PredictionSet) != 1 || ptRes.PredictionSet[0] != "A" || !ptRes.Singleton || ptRes.ShouldEscalate {
			t.Fatalf("expected confident singleton [A] at qHat=0, got %+v", ptRes)
		}
	}

	// 2. Repeated calls with fixed Seed: 2 advance stateful stream rather than freezing u or tie order
	calibTie := make([]map[string]float64, 100)
	labelsTie := make([]string, 100)
	for i := 0; i < 100; i++ {
		calibTie[i] = map[string]float64{"A": 0.5, "B": 0.5}
		if i%2 == 0 {
			labelsTie[i] = "A"
		} else {
			labelsTie[i] = "B"
		}
	}
	seedCfg := ConformalConfig{Method: "aps", Randomized: true, Seed: 2}
	qTie, err := CalibrateConformalQuantile(calibTie, labelsTie, 0.10, seedCfg)
	if err != nil {
		t.Fatalf("CalibrateConformalQuantile failed: %v", err)
	}
	hitsSeed := 0
	for i := 0; i < 200; i++ {
		gold := "A"
		if i%2 == 1 {
			gold = "B"
		}
		cset, err := ConformalPredictionSet(map[string]float64{"A": 0.5, "B": 0.5}, qTie, seedCfg)
		if err != nil {
			t.Fatalf("ConformalPredictionSet failed: %v", err)
		}
		for _, k := range cset.PredictionSet {
			if k == gold {
				hitsSeed++
				break
			}
		}
	}
	covSeed := float64(hitsSeed) / 200.0
	if covSeed < 0.85 {
		t.Fatalf("expected stateful Seed=2 randomized coverage >= 0.85 on exact ties, got %.4f", covSeed)
	}

	// 3. Seed: 0 (default randomized stream) smooths point-mass discrete jump toward 1 - alpha = 0.75
	calibPt := make([]map[string]float64, 600)
	labelsPt := make([]string, 600)
	for i := 0; i < 600; i++ {
		calibPt[i] = map[string]float64{"A": 1.0, "B": 0.0}
		labelsPt[i] = "A"
	}
	defRandCfg := ConformalConfig{Method: "aps", Randomized: true}
	qPt, err := CalibrateConformalQuantile(calibPt, labelsPt, 0.25, defRandCfg)
	if err != nil {
		t.Fatalf("CalibrateConformalQuantile point mass failed: %v", err)
	}
	hitsPt := 0
	for i := 0; i < 2000; i++ {
		cset, err := ConformalPredictionSet(map[string]float64{"A": 1.0, "B": 0.0}, qPt, defRandCfg)
		if err != nil {
			t.Fatalf("ConformalPredictionSet point mass failed: %v", err)
		}
		for _, k := range cset.PredictionSet {
			if k == "A" {
				hitsPt++
				break
			}
		}
	}
	covPt := float64(hitsPt) / 2000.0
	if covPt < 0.65 || covPt > 0.85 {
		t.Fatalf("expected smoothed randomized coverage near 0.75 in [0.65, 0.85], got %.4f", covPt)
	}

	// 4. Concurrent goroutine race-detector verification across shared and seeded streams
	done := make(chan error, 8)
	for g := 0; g < 8; g++ {
		go func(gid int) {
			c := ConformalConfig{Method: "raps", LamReg: 0.02, KReg: 1, Randomized: true, Seed: int64(gid % 3)}
			for j := 0; j < 25; j++ {
				if _, e := ConformalPredictionSet(unifA, 0.50, c); e != nil {
					done <- e
					return
				}
			}
			done <- nil
		}(g)
	}
	for g := 0; g < 8; g++ {
		if e := <-done; e != nil {
			t.Fatalf("concurrent ConformalPredictionSet failed: %v", e)
		}
	}
}

func TestDefaultConformalConfigRapsZeroBoundariesAndNineKeyAbstainUnion(t *testing.T) {
	defCfg := DefaultConformalConfig()
	if defCfg.Method != "raps" || defCfg.Alpha != 0.10 || defCfg.LamReg != 0.01 || defCfg.KReg != 1 || len(defCfg.AbstainKeys) != 9 {
		t.Fatalf("unexpected DefaultConformalConfig(): %+v", defCfg)
	}
	defCfg.AbstainKeys[0] = "mutated_caller_copy"
	if DefaultConformalConfig().AbstainKeys[0] != "0" {
		t.Fatal("DefaultConformalConfig() must return an isolated AbstainKeys slice copy")
	}

	sixOpts := map[string]float64{
		"o1": 0.30, "o2": 0.25, "o3": 0.20, "o4": 0.12, "o5": 0.08, "o6": 0.05,
	}
	// DefaultConformalConfig() and zero-value ConformalConfig{} both apply RAPS(KReg=1, LamReg=0.01):
	// rank 5 score = 0.87 + 0.04 = 0.91; rank 6 score = 0.95 + 0.05 = 1.00.
	for _, cfgUse := range []ConformalConfig{DefaultConformalConfig(), {}} {
		res5, err := ConformalPredictionSet(sixOpts, 0.91, cfgUse)
		if err != nil || len(res5.PredictionSet) != 5 {
			t.Fatalf("expected 5 items at qHat=0.91 for %+v, got res=%+v err=%v", cfgUse, res5, err)
		}
		res4, err := ConformalPredictionSet(sixOpts, 0.909, cfgUse)
		if err != nil || len(res4.PredictionSet) != 4 {
			t.Fatalf("expected 4 items at qHat=0.909 for %+v, got res=%+v err=%v", cfgUse, res4, err)
		}
	}

	// Explicit LamReg=0.0 reduces RAPS identically to APS across all calibration and prediction thresholds.
	calibP := []map[string]float64{sixOpts, {"o1": 0.10, "o2": 0.40, "o3": 0.20, "o4": 0.15, "o5": 0.10, "o6": 0.05}}
	calibL := []string{"o5", "o3"}
	apsCfg := ConformalConfig{Method: "aps"}
	qAPS, err := CalibrateConformalQuantile(calibP, calibL, 0.40, apsCfg)
	if err != nil {
		t.Fatalf("APS calibration failed: %v", err)
	}
	for _, kZero := range []int{0, 1, 3} {
		rapsZero := ConformalConfig{Method: "raps", LamReg: 0.0, KReg: kZero}
		qRAPS, err := CalibrateConformalQuantile(calibP, calibL, 0.40, rapsZero)
		if err != nil || math.Abs(qRAPS-qAPS) > 1e-12 {
			t.Fatalf("expected RAPS(LamReg=0, KReg=%d) qHat=%v to equal APS qHat=%v (err=%v)", kZero, qRAPS, qAPS, err)
		}
		setR, err1 := ConformalPredictionSet(sixOpts, 0.87, rapsZero)
		setA, err2 := ConformalPredictionSet(sixOpts, 0.87, apsCfg)
		if err1 != nil || err2 != nil || len(setR.PredictionSet) != len(setA.PredictionSet) {
			t.Fatalf("expected identical prediction sets at LamReg=0, got %+v vs %+v", setR, setA)
		}
	}

	// Explicit KReg=0 with LamReg=0.20 penalizes rank 1 (rank 1 score = 0.0 + 0.20 = 0.20).
	k0Cfg := ConformalConfig{Method: "raps", LamReg: 0.20, KReg: 0}
	emptySet, err := ConformalPredictionSet(sixOpts, 0.19, k0Cfg)
	if err != nil || len(emptySet.PredictionSet) != 0 || !emptySet.ShouldEscalate {
		t.Fatalf("expected empty prediction set at qHat=0.19 under KReg=0 LamReg=0.20, got %+v err=%v", emptySet, err)
	}
	rank1Set, err := ConformalPredictionSet(sixOpts, 0.20, k0Cfg)
	if err != nil || len(rank1Set.PredictionSet) != 1 || rank1Set.PredictionSet[0] != "o1" {
		t.Fatalf("expected singleton [o1] at qHat=0.20 under KReg=0 LamReg=0.20, got %+v err=%v", rank1Set, err)
	}

	// All 9 default abstain keys (case-insensitive and whitespace-trimmed) escalate when AbstainKeys is nil,
	// while explicit empty non-nil AbstainKeys: []string{} opts out cleanly.
	nineKeys := []string{
		"0",
		"abstain",
		"insufficient_evidence",
		"none",
		"none_of_the_above",
		"none_visible",
		"oos",
		"uncertain",
		"unknown",
		" OOS ",
		"ABSTAIN",
		" Uncertain ",
	}
	for _, ak := range nineKeys {
		probs := map[string]float64{ak: 0.96, "valid_sku": 0.04}
		resNil, err := ConformalPredictionSet(probs, 0.20, ConformalConfig{Method: "lac"})
		if err != nil || !resNil.Singleton || !resNil.ContainsAbstain || !resNil.ShouldEscalate {
			t.Fatalf("expected default abstain escalation for %q, got res=%+v err=%v", ak, resNil, err)
		}
		resOptOut, err := ConformalPredictionSet(probs, 0.20, ConformalConfig{Method: "lac", AbstainKeys: []string{}})
		if err != nil || !resOptOut.Singleton || resOptOut.ContainsAbstain || resOptOut.ShouldEscalate {
			t.Fatalf("expected empty AbstainKeys slice to opt out for %q, got res=%+v err=%v", ak, resOptOut, err)
		}
	}
}
