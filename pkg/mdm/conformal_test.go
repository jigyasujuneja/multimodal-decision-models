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
