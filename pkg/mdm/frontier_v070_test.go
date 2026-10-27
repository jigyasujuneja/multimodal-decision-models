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
	"math"
	"testing"
)

func TestOnlineBCACIDriftStreamGo(t *testing.T) {
	streamProbs := make([]map[string]float64, 0, 120)
	streamLabels := make([]string, 0, 120)
	for i := 0; i < 60; i++ {
		streamProbs = append(streamProbs, map[string]float64{"a": 0.92, "b": 0.05, "c": 0.03})
		streamLabels = append(streamLabels, "a")
	}
	// Abrupt shift where top-1 is wrong and gold is rank-2/3
	for i := 0; i < 60; i++ {
		streamProbs = append(streamProbs, map[string]float64{"a": 0.70, "b": 0.20, "c": 0.10})
		if i%2 == 0 {
			streamLabels = append(streamLabels, "b")
		} else {
			streamLabels = append(streamLabels, "c")
		}
	}

	res, err := EvaluateOnlineBCACIStream(
		streamProbs,
		streamLabels,
		0.10,
		0.05,
		0.025,
		0.15,
		40,
		30,
		DefaultConformalConfig(),
	)
	if err != nil {
		t.Fatalf("EvaluateOnlineBCACIStream failed: %v", err)
	}
	if res.EmpiricalCoverage < 0.82 {
		t.Fatalf("expected BC-ACI empirical coverage >= 0.82 under abrupt drift, got %.4f", res.EmpiricalCoverage)
	}
	if res.RollingWindowWorstCoverage < 0.75 {
		t.Fatalf("expected rolling window worst coverage >= 0.75 under abrupt drift, got %.4f", res.RollingWindowWorstCoverage)
	}
	// Right after the regime shift at step 60..75, alpha_t tightens downward to expand sets
	minPostShiftAlpha := 1.0
	for i := 60; i < 80; i++ {
		if res.AlphaTrajectory[i] < minPostShiftAlpha {
			minPostShiftAlpha = res.AlphaTrajectory[i]
		}
	}
	if minPostShiftAlpha >= res.AlphaTrajectory[59] {
		t.Fatalf("expected alpha_t to tighten immediately after shift at step 60, got minPostShiftAlpha=%.4f vs preShift=%.4f", minPostShiftAlpha, res.AlphaTrajectory[59])
	}
}

func TestPrimalDualParetoBanditRouterGo(t *testing.T) {
	arms := []ModelArmSpec{
		{ArmID: "fast_4b", CostUSD: 0.0004, LatencyMs: 35.0},
		{ArmID: "mid_7b", CostUSD: 0.0020, LatencyMs: 110.0},
		{ArmID: "heavy_27b", CostUSD: 0.0150, LatencyMs: 620.0},
	}
	router, err := NewPrimalDualParetoBanditRouter(arms, 3, 0.0030, 0.35, 0.98, 1.0, 4.0)
	if err != nil {
		t.Fatalf("NewPrimalDualParetoBanditRouter failed: %v", err)
	}
	totalCost := 0.0
	for step := 0; step < 80; step++ {
		ctx := []float64{1.0, float64(step%5) * 0.2, 0.5}
		dec, sErr := router.SelectArm(ctx)
		if sErr != nil {
			t.Fatalf("SelectArm failed: %v", sErr)
		}
		reward := 0.85
		if dec.ChosenArm == "heavy_27b" {
			reward = 0.94
		} else if dec.ChosenArm == "mid_7b" {
			reward = 0.90
		}
		totalCost += router.Arms[dec.ChosenArm].CostUSD
		if _, uErr := router.Update(dec.ChosenArm, ctx, reward); uErr != nil {
			t.Fatalf("Update failed: %v", uErr)
		}
		if dErr := router.UpdateDuelingPreference("mid_7b", "fast_4b", ctx); dErr != nil {
			t.Fatalf("UpdateDuelingPreference failed: %v", dErr)
		}
	}
	meanCost := totalCost / 80.0
	if meanCost > 0.0045 {
		t.Fatalf("expected primal-dual budget pacing to keep meanCost <= 0.0045, got %.6f", meanCost)
	}
}

func TestCalibratedPRMPUCTSearchGo(t *testing.T) {
	nodes := []PRMBranchNode{
		{NodeID: "r1", ParentID: "", Depth: 1, PriorProb: 0.65, RubricProbs: map[string]float64{"0": 0.02, "1": 0.03, "2": 0.15, "3": 0.80}},
		{NodeID: "r1_c1", ParentID: "r1", Depth: 2, PriorProb: 0.80, RubricProbs: map[string]float64{"0": 0.01, "1": 0.04, "2": 0.15, "3": 0.80}, IsTerminal: true, GroundTruthCorrect: true},
		{NodeID: "r2", ParentID: "", Depth: 1, PriorProb: 0.35, RubricProbs: map[string]float64{"0": 0.85, "1": 0.10, "2": 0.03, "3": 0.02}},
		{NodeID: "r2_c1", ParentID: "r2", Depth: 2, PriorProb: 0.50, RubricProbs: map[string]float64{"0": 0.80, "1": 0.10, "2": 0.05, "3": 0.05}, IsTerminal: true, GroundTruthCorrect: false},
		{NodeID: "r2_c2", ParentID: "r2", Depth: 2, PriorProb: 0.50, RubricProbs: map[string]float64{"0": 0.75, "1": 0.15, "2": 0.05, "3": 0.05}, IsTerminal: true, GroundTruthCorrect: false},
	}
	res, err := RunCalibratedPRMPUCTSearch(nodes, 1.28, 0.75, map[int]float64{1: 1.0, 2: 0.95}, true)
	if err != nil {
		t.Fatalf("RunCalibratedPRMPUCTSearch failed: %v", err)
	}
	if !res.LeafCorrect || res.SelectedLeafID != "r1_c1" {
		t.Fatalf("expected r1_c1 correct leaf, got %+v", res)
	}
	if res.PrunedNodes < 2 {
		t.Fatalf("expected >= 2 pruned nodes on low-value r2 subtree, got %d", res.PrunedNodes)
	}
}

func TestCovarianceDiscountedLogitPoolCloneInvarianceGo(t *testing.T) {
	calibLabels := []string{"a", "b", "a", "b", "a", "b"}
	expert := []map[string]float64{
		{"a": 0.88, "b": 0.12},
		{"a": 0.15, "b": 0.85},
		{"a": 0.90, "b": 0.10},
		{"a": 0.12, "b": 0.88},
		{"a": 0.86, "b": 0.14},
		{"a": 0.14, "b": 0.86},
	}
	clone := []map[string]float64{
		{"a": 0.25, "b": 0.75},
		{"a": 0.70, "b": 0.30},
		{"a": 0.30, "b": 0.70},
		{"a": 0.75, "b": 0.25},
		{"a": 0.28, "b": 0.72},
		{"a": 0.72, "b": 0.28},
	}
	calibMap := map[string][]map[string]float64{
		"expert":  expert,
		"clone_1": clone,
		"clone_2": clone,
		"clone_3": clone,
		"clone_4": clone,
	}
	cal, err := CalibrateCorrelatedEnsembleWeights(calibMap, calibLabels, 0.10)
	if err != nil {
		t.Fatalf("CalibrateCorrelatedEnsembleWeights failed: %v", err)
	}
	if cal.Weights["expert"] <= cal.Weights["clone_1"]*2.0 {
		t.Fatalf("expected expert weight (%.4f) >> single clone weight (%.4f)", cal.Weights["expert"], cal.Weights["clone_1"])
	}
	testProbs := map[string]map[string]float64{
		"expert":  {"a": 0.90, "b": 0.10},
		"clone_1": {"a": 0.35, "b": 0.65},
		"clone_2": {"a": 0.35, "b": 0.65},
		"clone_3": {"a": 0.35, "b": 0.65},
		"clone_4": {"a": 0.35, "b": 0.65},
	}
	pooled, pErr := CovarianceDiscountedLogitPool(testProbs, &cal, 1.0)
	if pErr != nil {
		t.Fatalf("CovarianceDiscountedLogitPool failed: %v", pErr)
	}
	if pooled["a"] <= pooled["b"] || math.IsNaN(pooled["a"]) {
		t.Fatalf("expected covariance-discounted pool to select 'a' over 4 correlated clones, got %+v", pooled)
	}
}
