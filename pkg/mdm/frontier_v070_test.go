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

func TestOnlineBCACIDriftStream(t *testing.T) {
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
		t.Fatalf("EvaluateOnlineBCACIStream().EmpiricalCoverage = %.4f, want >= 0.82", res.EmpiricalCoverage)
	}
	if res.RollingWindowWorstCoverage < 0.75 {
		t.Fatalf("EvaluateOnlineBCACIStream().RollingWindowWorstCoverage = %.4f, want >= 0.75", res.RollingWindowWorstCoverage)
	}
	// Right after the regime shift at step 60..75, alpha_t tightens downward to expand sets
	minPostShiftAlpha := 1.0
	for i := 60; i < 80; i++ {
		if res.AlphaTrajectory[i] < minPostShiftAlpha {
			minPostShiftAlpha = res.AlphaTrajectory[i]
		}
	}
	if minPostShiftAlpha >= res.AlphaTrajectory[59] {
		t.Fatalf("minPostShiftAlpha = %.4f, want < preShift %.4f", minPostShiftAlpha, res.AlphaTrajectory[59])
	}
}

func TestPrimalDualParetoBanditRouter(t *testing.T) {
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
		t.Fatalf("meanCost = %.6f, want <= 0.0045", meanCost)
	}
}

func TestCalibratedPRMPUCTSearch(t *testing.T) {
	nodes := []PRMBranchNode{
		{NodeID: "r1", ParentID: "", Depth: 1, PriorProb: 0.65, RubricProbs: map[string]float64{"0": 0.02, "1": 0.03, "2": 0.15, "3": 0.80}},
		{NodeID: "r1_c1", ParentID: "r1", Depth: 2, PriorProb: 0.80, RubricProbs: map[string]float64{"0": 0.01, "1": 0.04, "2": 0.15, "3": 0.80}, IsTerminal: true, GroundTruthCorrect: true},
		{NodeID: "r2", ParentID: "", Depth: 1, PriorProb: 0.35, RubricProbs: map[string]float64{"0": 0.85, "1": 0.10, "2": 0.03, "3": 0.02}},
		{NodeID: "r2_c1", ParentID: "r2", Depth: 2, PriorProb: 0.50, RubricProbs: map[string]float64{"0": 0.80, "1": 0.10, "2": 0.05, "3": 0.05}, IsTerminal: true, GroundTruthCorrect: false},
		{NodeID: "r2_c2", ParentID: "r2", Depth: 2, PriorProb: 0.50, RubricProbs: map[string]float64{"0": 0.75, "1": 0.15, "2": 0.05, "3": 0.05}, IsTerminal: true, GroundTruthCorrect: false},
	}
	res, err := RunCalibratedPRMPUCTSearch(nodes, 1.28, 0.75, map[int]float64{1: 1.0, 2: 0.95}, true)
	if err != nil {
		t.Fatalf("RunCalibratedPRMPUCTSearch() unexpected err = %v", err)
	}
	if !res.LeafCorrect || res.SelectedLeafID != "r1_c1" {
		t.Fatalf("RunCalibratedPRMPUCTSearch() = %+v, want LeafCorrect=true SelectedLeafID=r1_c1", res)
	}
	if res.PrunedNodes < 2 {
		t.Fatalf("RunCalibratedPRMPUCTSearch().PrunedNodes = %d, want >= 2", res.PrunedNodes)
	}
}

func TestCovarianceDiscountedLogitPoolCloneInvariance(t *testing.T) {
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
		t.Fatalf("CalibrateCorrelatedEnsembleWeights() unexpected err = %v", err)
	}
	if cal.Weights["expert"] <= cal.Weights["clone_1"]*2.0 {
		t.Fatalf("cal.Weights[expert] = %.4f, want > 2 * clone_1 (%.4f)", cal.Weights["expert"], cal.Weights["clone_1"])
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
		t.Fatalf("CovarianceDiscountedLogitPool() unexpected err = %v", pErr)
	}
	if pooled["a"] <= pooled["b"] || math.IsNaN(pooled["a"]) {
		t.Fatalf("CovarianceDiscountedLogitPool() = %+v, want a > b", pooled)
	}
}

func TestPRMTreeSearchValidation(t *testing.T) {
	cyclic := []PRMBranchNode{
		{NodeID: "root", ParentID: "", Depth: 1, PriorProb: 0.6, RubricProbs: map[string]float64{"0": 0.1, "3": 0.9}},
		{NodeID: "a", ParentID: "b", Depth: 2, PriorProb: 0.5, RubricProbs: map[string]float64{"0": 0.1, "3": 0.9}},
		{NodeID: "b", ParentID: "a", Depth: 3, PriorProb: 0.5, RubricProbs: map[string]float64{"0": 0.1, "3": 0.9}},
	}
	if _, err := RunCalibratedPRMPUCTSearch(cyclic, 1.0, 0.5, nil, true); err == nil {
		t.Fatal("RunCalibratedPRMPUCTSearch(cyclic) = nil err, want error")
	}

	dup := []PRMBranchNode{
		{NodeID: "r1", ParentID: "", Depth: 1, PriorProb: 0.5, RubricProbs: map[string]float64{"3": 1.0}},
		{NodeID: "r1", ParentID: "", Depth: 1, PriorProb: 0.5, RubricProbs: map[string]float64{"3": 1.0}},
	}
	if _, err := RunCalibratedPRMPUCTSearch(dup, 1.0, 0.5, nil, true); err == nil {
		t.Fatal("RunCalibratedPRMPUCTSearch(duplicate NodeID) = nil err, want error")
	}
}

func TestPRMTreeSearchDepth5Soundness(t *testing.T) {
	// Incumbent depth-2 path averages 0.80 (0.80 -> 0.80, zero variance -> LCB 0.80).
	// Challenger depth-5 path starts at 0.60 and finishes with four 1.0 steps (mean 0.92 > 0.80).
	nodes := []PRMBranchNode{
		{NodeID: "shallow_1", ParentID: "", Depth: 1, PriorProb: 0.9, RubricProbs: map[string]float64{"0": 0.0, "4": 1.0}},
		{NodeID: "shallow_2", ParentID: "shallow_1", Depth: 2, PriorProb: 0.9, RubricProbs: map[string]float64{"0": 0.0, "4": 1.0}, IsTerminal: true, GroundTruthCorrect: false},
		{NodeID: "deep_1", ParentID: "", Depth: 1, PriorProb: 0.1, RubricProbs: map[string]float64{"0": 0.0, "3": 1.0, "5": 0.0}},
		{NodeID: "deep_2", ParentID: "deep_1", Depth: 2, PriorProb: 0.9, RubricProbs: map[string]float64{"5": 1.0}},
		{NodeID: "deep_3", ParentID: "deep_2", Depth: 3, PriorProb: 0.9, RubricProbs: map[string]float64{"5": 1.0}},
		{NodeID: "deep_4", ParentID: "deep_3", Depth: 4, PriorProb: 0.9, RubricProbs: map[string]float64{"5": 1.0}},
		{NodeID: "deep_5", ParentID: "deep_4", Depth: 5, PriorProb: 0.9, RubricProbs: map[string]float64{"5": 1.0}, IsTerminal: true, GroundTruthCorrect: true},
	}
	// Make shallow_1 and shallow_2 explicitly grade 4 out of max grade 5 (= 0.80).
	nodes[0].RubricProbs["5"] = 0.0
	nodes[1].RubricProbs["5"] = 0.0

	res, err := RunCalibratedPRMPUCTSearch(nodes, 1.0, 0.25, nil, true)
	if err != nil {
		t.Fatalf("RunCalibratedPRMPUCTSearch(depth5) unexpected err = %v", err)
	}
	if res.SelectedLeafID != "deep_5" {
		t.Fatalf("RunCalibratedPRMPUCTSearch(depth5).SelectedLeafID = %q (mean=%.4f), want %q", res.SelectedLeafID, res.SelectedValueMean, "deep_5")
	}
	if math.Abs(res.SelectedValueMean-0.92) > 1e-9 {
		t.Fatalf("RunCalibratedPRMPUCTSearch(depth5).SelectedValueMean = %.6f, want 0.920000", res.SelectedValueMean)
	}
}

func TestBanditRouterValidationAndNonFiniteGuards(t *testing.T) {
	dupArms := []ModelArmSpec{
		{ArmID: "a1", CostUSD: 0.001, LatencyMs: 10},
		{ArmID: "a1", CostUSD: 0.002, LatencyMs: 20},
	}
	if _, err := NewPrimalDualParetoBanditRouter(dupArms, 2, 0.01, 0.2, 0.95, 1.0, 1.0); err == nil {
		t.Fatal("NewPrimalDualParetoBanditRouter(duplicate ArmID) = nil err, want error")
	}
	validArms := []ModelArmSpec{
		{ArmID: "a1", CostUSD: 0.001, LatencyMs: 10},
		{ArmID: "a2", CostUSD: 0.002, LatencyMs: 20},
	}
	if _, err := NewPrimalDualParetoBanditRouter(validArms, 2, math.Inf(1), 0.2, 0.95, 1.0, 1.0); err == nil {
		t.Fatal("NewPrimalDualParetoBanditRouter(+Inf budget) = nil err, want error")
	}
	if _, err := NewPrimalDualParetoBanditRouter(validArms, 2, 0.01, math.NaN(), 0.95, 1.0, 1.0); err == nil {
		t.Fatal("NewPrimalDualParetoBanditRouter(NaN alphaUCB) = nil err, want error")
	}
	if _, err := NewPrimalDualParetoBanditRouter([]ModelArmSpec{{ArmID: "a1", CostUSD: -0.1, LatencyMs: 10}}, 2, 0.01, 0.2, 0.95, 1.0, 1.0); err == nil {
		t.Fatal("NewPrimalDualParetoBanditRouter(negative CostUSD) = nil err, want error")
	}

	router, err := NewPrimalDualParetoBanditRouter(validArms, 2, 0.01, 0.2, 0.95, 1.0, 1.0)
	if err != nil {
		t.Fatalf("NewPrimalDualParetoBanditRouter() unexpected err = %v", err)
	}
	if _, err := router.SelectArm([]float64{1.0, math.NaN()}); err == nil {
		t.Fatal("router.SelectArm(NaN context) = nil err, want error")
	}
	if _, err := router.Update("a1", []float64{math.Inf(1), 0.5}, 0.8); err == nil {
		t.Fatal("router.Update(+Inf context) = nil err, want error")
	}
	if _, err := router.Update("a1", []float64{1.0, 0.5}, math.NaN()); err == nil {
		t.Fatal("router.Update(NaN reward) = nil err, want error")
	}
	if err := router.UpdateDuelingPreference("a1", "a2", []float64{0.2, math.Inf(-1)}); err == nil {
		t.Fatal("router.UpdateDuelingPreference(-Inf context) = nil err, want error")
	}
	if got := router.PullCounts["a1"]; got != 0 {
		t.Fatalf("router.PullCounts[a1] after rejected updates = %d, want 0", got)
	}
	dec, err := router.SelectArm([]float64{1.0, 0.5})
	if err != nil {
		t.Fatalf("router.SelectArm(valid after rejected updates) unexpected err = %v", err)
	}
	if !allFinite(dec.UCBQualityScore, dec.PrimalDualAcquisition) {
		t.Fatalf("router.SelectArm() scores = (%v, %v), want finite values", dec.UCBQualityScore, dec.PrimalDualAcquisition)
	}
}

func TestTangoNonInferiorityUsesGammaQuantile(t *testing.T) {
	resTight, err := TangoPairedNonInferiorityTest(12, 8, 200, 0.05, 0.05, 1e-4)
	if err != nil {
		t.Fatalf("TangoPairedNonInferiorityTest(gamma=1e-4) unexpected err = %v", err)
	}
	resLoose, err := TangoPairedNonInferiorityTest(12, 8, 200, 0.05, 0.05, 0.05)
	if err != nil {
		t.Fatalf("TangoPairedNonInferiorityTest(gamma=0.05) unexpected err = %v", err)
	}
	if math.Abs(resTight.PBergerBoos-resLoose.PBergerBoos) < 1e-4 {
		t.Fatalf("PBergerBoos diff = |%v - %v|, want >= 1e-4", resTight.PBergerBoos, resLoose.PBergerBoos)
	}
	if _, err := TangoPairedNonInferiorityTest(12, 8, 200, math.NaN(), 0.05, 1e-4); err == nil {
		t.Fatal("TangoPairedNonInferiorityTest(NaN margin) = nil err, want error")
	}
}

func TestConformalLabelShiftInfAndMissingBonferroniSlot(t *testing.T) {
	cset, err := LabelShiftWeightedConformalPredictionSet(
		[]map[string]float64{{"a": 0.9, "b": 0.1}},
		[]string{"a"},
		map[string]float64{"a": 1.0, "b": 1.0},
		map[string]float64{"a": 0.9, "b": 0.1},
		[]float64{1.0},
		1000.0,
		0.10,
		DefaultConformalConfig(),
	)
	if err != nil {
		t.Fatalf("LabelShiftWeightedConformalPredictionSet() unexpected err = %v", err)
	}
	if !math.IsInf(cset.QHat, 1) {
		t.Fatalf("LabelShiftWeightedConformalPredictionSet().QHat = %v, want +Inf", cset.QHat)
	}

	jointCal := MultiSlotJointConformalResult{
		FamilyAlpha:   0.10,
		SlotNames:     []string{"slot_a"},
		GlobalMaxQHat: 0.95,
		BonferroniQs:  map[string]float64{},
		Method:        "raps",
		NCalibration:  10,
	}
	sets, _, err := MultiSlotJointConformalSets(
		map[string]map[string]float64{"slot_a": {"x": 0.6, "y": 0.4}},
		jointCal,
		true,
		DefaultConformalConfig(),
	)
	if err != nil {
		t.Fatalf("MultiSlotJointConformalSets() unexpected err = %v", err)
	}
	if got := sets["slot_a"].QHat; got != 0.95 {
		t.Fatalf("MultiSlotJointConformalSets()[slot_a].QHat = %v, want 0.95", got)
	}
}

func TestHMACUppercaseInitialismMatchesAlias(t *testing.T) {
	if got, want := HMACUint64("k", "item", "u"), HmacUint64("k", "item", "u"); got != want {
		t.Fatalf("HMACUint64 = %d, want %d", got, want)
	}
	if got, want := HMACUniformFloat64("k", "item", "u"), HmacUniformFloat64("k", "item", "u"); got != want {
		t.Fatalf("HMACUniformFloat64 = %v, want %v", got, want)
	}
}

func TestAudioVideoRejectsNonFiniteInputs(t *testing.T) {
	if _, err := EvaluateStreamingEarlyExit(
		[]map[string]float64{{"a": 0.8, "b": 0.2}},
		[]float64{0.1, 0.2},
		math.NaN(),
		0.2,
		1,
		1,
		0.05,
		0.9,
	); err == nil {
		t.Fatal("EvaluateStreamingEarlyExit(NaN conformalQHat) = nil err, want error")
	}
	if _, err := SelectSubmodularSpatioTemporalFrames(
		[]SpatioTemporalFrameCandidate{
			{FrameIndex: 0, TimestampSec: 0, HesitationScore: 0.5, FeatureVector: []float64{math.NaN()}},
		},
		1, 0.5, 0.2, 0.5,
	); err == nil {
		t.Fatal("SelectSubmodularSpatioTemporalFrames(NaN FeatureVector) = nil err, want error")
	}
}
