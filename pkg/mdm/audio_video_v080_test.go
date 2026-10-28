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

func TestEvaluateDuplexTurnTakingDecision(t *testing.T) {
	cfg := DefaultVoiceActivityProjectionConfig()

	// Case 1: Mid-thought cognitive pause (320ms, flat pitch, low semantic completion) -> must wait_user_thinking
	stage1MidThought := map[string]float64{
		"hold_floor":         0.10,
		"yield_turn":         0.38,
		"backchannel":        0.10,
		"barge_in_interrupt": 0.14,
		"wait_user_thinking": 0.28,
	}
	res1, err := EvaluateDuplexTurnTakingDecision(stage1MidThought, 320.0, -0.5, 2.0, 0.25, false, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res1.SelectedAction != "wait_user_thinking" {
		t.Fatalf("expected wait_user_thinking on mid-thought pause, got %q", res1.SelectedAction)
	}

	// Case 2: Clear turn yield (880ms pause, falling F0 and RMS, high semantic completion)
	stage1Yield := map[string]float64{
		"hold_floor":         0.04,
		"yield_turn":         0.76,
		"backchannel":        0.06,
		"barge_in_interrupt": 0.04,
		"wait_user_thinking": 0.10,
	}
	res2, err := EvaluateDuplexTurnTakingDecision(stage1Yield, 880.0, -4.5, -22.0, 0.92, false, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res2.SelectedAction != "yield_turn" {
		t.Fatalf("expected yield_turn on clear terminal pause, got %q", res2.SelectedAction)
	}
}

func TestConformalMartingaleAndStreamingEarlyExit(t *testing.T) {
	calib := []float64{0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.25, 0.30, 0.35}
	stream := []map[string]float64{
		{"action_a": 0.88, "action_b": 0.08, "action_c": 0.04},
		{"action_a": 0.92, "action_b": 0.05, "action_c": 0.03},
		{"action_a": 0.95, "action_b": 0.03, "action_c": 0.02},
		{"action_a": 0.96, "action_b": 0.02, "action_c": 0.02},
		{"action_a": 0.97, "action_b": 0.02, "action_c": 0.01},
		{"action_a": 0.97, "action_b": 0.02, "action_c": 0.01},
	}
	res, err := EvaluateStreamingEarlyExit(stream, calib, 0.82, 0.25, 2, 2, 0.05, 0.92)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.CertifiedEarlyExit {
		t.Fatalf("expected certified early exit on high-confidence stream")
	}
	if res.ExitStep > 3 {
		t.Fatalf("expected early exit by step 3, got step %d", res.ExitStep)
	}
	if res.SelectedClass != "action_a" {
		t.Fatalf("expected action_a, got %q", res.SelectedClass)
	}
	if res.ComputeSavedFraction < 0.45 {
		t.Fatalf("expected >= 0.45 compute saved, got %v", res.ComputeSavedFraction)
	}
}

func TestEvaluateAudioVisualEventAgreement(t *testing.T) {
	// 1. Off-screen audio event: glass_break loud on audio, background on visual
	audOff := map[string]float64{"glass_break": 0.88, "dog_bark": 0.06, "background": 0.06}
	visOff := map[string]float64{"glass_break": 0.05, "dog_bark": 0.05, "background": 0.90}
	resOff, err := EvaluateAudioVisualEventAgreement(audOff, visOff, 0.40, 0.36, 0.30, 0.42, 0.52)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resOff.Regime != "audible_offscreen_event" || resOff.SelectedEvent != "glass_break" {
		t.Fatalf("expected audible_offscreen_event/glass_break, got %s/%s", resOff.Regime, resOff.SelectedEvent)
	}

	// 2. Spoof / conflict: audio says glass_break, video says dog_bark with high sync claim
	audConf := map[string]float64{"glass_break": 0.90, "dog_bark": 0.05, "background": 0.05}
	visConf := map[string]float64{"glass_break": 0.05, "dog_bark": 0.90, "background": 0.05}
	resConf, err := EvaluateAudioVisualEventAgreement(audConf, visConf, 0.90, 0.36, 0.30, 0.42, 0.52)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resConf.Regime != "cross_modal_spoof_or_conflict" || !resConf.EscalateToStage2 {
		t.Fatalf("expected cross_modal_spoof_or_conflict with escalation, got %+v", resConf)
	}
}

func TestSelectSubmodularSpatioTemporalFrames(t *testing.T) {
	cands := []SpatioTemporalFrameCandidate{
		{FrameIndex: 0, TimestampSec: 0.0, HesitationScore: 0.85, AcousticTransientScore: 0.80, VisualMotionScore: 0.70, FeatureVector: []float64{1, 0, 0}},
		{FrameIndex: 1, TimestampSec: 0.2, HesitationScore: 0.84, AcousticTransientScore: 0.79, VisualMotionScore: 0.68, FeatureVector: []float64{0.99, 0.05, 0}},
		{FrameIndex: 2, TimestampSec: 2.5, HesitationScore: 0.75, AcousticTransientScore: 0.70, VisualMotionScore: 0.80, FeatureVector: []float64{0, 1, 0}},
		{FrameIndex: 3, TimestampSec: 5.0, HesitationScore: 0.78, AcousticTransientScore: 0.85, VisualMotionScore: 0.72, FeatureVector: []float64{0, 0, 1}},
	}
	res, err := SelectSubmodularSpatioTemporalFrames(cands, 3, 0.55, 0.45, 0.65)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(res.SelectedIndices) != 3 {
		t.Fatalf("expected 3 selected frames, got %v", res.SelectedIndices)
	}
	// Should avoid picking both near-duplicate frames 0 and 1 when distinct clusters 2 and 3 exist
	has2 := false
	has3 := false
	for _, idx := range res.SelectedIndices {
		if idx == 2 {
			has2 = true
		}
		if idx == 3 {
			has3 = true
		}
	}
	if !has2 || !has3 || math.IsNaN(res.ObjectiveValue) {
		t.Fatalf("expected diverse frames 2 and 3 in selection, got %v", res.SelectedIndices)
	}

	// Zero-sum probability distribution must fail closed (no uniform 1/K fallback)
	if _, err := JensenShannonDivergenceBits(map[string]float64{"a": 0.0, "b": 0.0}, map[string]float64{"a": 0.5, "b": 0.5}); err == nil {
		t.Fatalf("expected error on zero-sum distribution")
	}
}
