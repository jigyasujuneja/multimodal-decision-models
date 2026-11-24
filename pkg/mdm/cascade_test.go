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
	"context"
	"errors"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestPruneChoiceTopKRetainsOptionDescriptionsWithoutAliasing(t *testing.T) {
	prof := DomainCompetenceProfile{
		Stage1Accuracy:        0.90,
		Stage2Accuracy:        0.93,
		NumOptions:            4,
		PairedEscalationDelta: 2,
	}
	orig := ChoiceQuestion{
		ID:           "route_q",
		Kind:         QuestionKindChoice,
		Instructions: "Choose destination queue",
		Options:      []string{"opt_a", "opt_b", "opt_c", "opt_d"},
		OptionDescriptions: map[string]string{
			"opt_a": "Alpha queue",
			"opt_b": "Bravo queue",
			"opt_c": "Charlie queue",
			"opt_d": "Delta queue",
		},
		PriorLogits: map[string]float64{
			"opt_a": -1.0,
			"opt_b": 0.9,
			"opt_c": 0.4,
			"opt_d": -2.0,
		},
		DomainCompetence: &prof,
	}

	pruned, err := PruneChoiceTopK(orig, map[string]float64{
		"opt_a": 0.11,
		"opt_b": 0.51,
		"opt_c": 0.30,
		"opt_d": 0.08,
	}, 2)
	if err != nil {
		t.Fatalf("PruneChoiceTopK failed: %v", err)
	}
	if len(pruned.Options) != 2 || pruned.Options[0] != "opt_b" || pruned.Options[1] != "opt_c" {
		t.Fatalf("expected kept options [opt_b, opt_c], got %v", pruned.Options)
	}
	if len(pruned.OptionDescriptions) != 2 ||
		pruned.OptionDescriptions["opt_b"] != "Bravo queue" ||
		pruned.OptionDescriptions["opt_c"] != "Charlie queue" {
		t.Fatalf("unexpected pruned OptionDescriptions: %+v", pruned.OptionDescriptions)
	}
	if _, hasA := pruned.OptionDescriptions["opt_a"]; hasA {
		t.Fatal("expected dropped opt_a to be omitted from OptionDescriptions")
	}
	if _, hasD := pruned.OptionDescriptions["opt_d"]; hasD {
		t.Fatal("expected dropped opt_d to be omitted from OptionDescriptions")
	}
	pruned.OptionDescriptions["opt_b"] = "Mutated"
	if orig.OptionDescriptions["opt_b"] != "Bravo queue" {
		t.Fatal("PruneChoiceTopK must deep-copy OptionDescriptions without mutating caller map")
	}
	if len(pruned.PriorLogits) != 2 || pruned.PriorLogits["opt_b"] != 0.9 || pruned.PriorLogits["opt_c"] != 0.4 {
		t.Fatalf("unexpected pruned PriorLogits: %+v", pruned.PriorLogits)
	}

	if _, badErr := PruneChoiceTopK(orig, map[string]float64{"opt_a": 0.5, "opt_b": 0.5}, 1); badErr == nil {
		t.Fatal("expected topK=1 to fail validation")
	}
}

func TestSpeculativeCircuitBreakerAndUniformRescueAndPreservation(t *testing.T) {
	negProf := DomainCompetenceProfile{
		Stage1Accuracy:        0.75,
		Stage2Accuracy:        0.62,
		NumOptions:            2,
		PairedEscalationDelta: -3,
	}
	posProf := DomainCompetenceProfile{
		Stage1Accuracy:        0.90,
		Stage2Accuracy:        0.88,
		NumOptions:            2,
		PairedEscalationDelta: 2,
	}
	baseQ := ChoiceQuestion{
		ID:           "q1",
		Kind:         QuestionKindChoice,
		Instructions: "Select route",
		Options:      []string{"opt_a", "opt_b"},
		OptionDescriptions: map[string]string{
			"opt_a": "Route A",
			"opt_b": "Route B",
		},
	}

	// 1. All choice questions trip IsNegativeTransfer -> 0 speculative Stage-2 calls at t=0
	var specCalls int32
	cfgBlocked := DefaultCascadeConfig()
	cfgBlocked.SpeculativeStage2 = true
	cfgBlocked.DomainCompetence = &negProf
	orchBlocked, err := NewCascadeOrchestrator(
		CascadeBackendFunc(func(_ context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
			return map[string]map[string]float64{"q1": {"opt_a": 0.60, "opt_b": 0.40}}, nil
		}),
		CascadeBackendFunc(func(_ context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
			atomic.AddInt32(&specCalls, 1)
			return map[string]map[string]float64{"q1": {"opt_a": 0.10, "opt_b": 0.90}}, nil
		}),
		cfgBlocked,
	)
	if err != nil {
		t.Fatalf("NewCascadeOrchestrator failed: %v", err)
	}
	resBlocked, err := orchBlocked.Evaluate(context.Background(), "req1", []ChoiceQuestion{baseQ})
	if err != nil {
		t.Fatalf("Evaluate failed: %v", err)
	}
	if atomic.LoadInt32(&specCalls) != 0 {
		t.Fatalf("expected 0 Stage-2 calls when all choice questions trip negative transfer, got %d", specCalls)
	}
	if !resBlocked.SpeculativeSkippedByCircuitBreaker || resBlocked.SpeculativeLaunched || resBlocked.Escalated {
		t.Fatalf("unexpected speculative flags: %+v", resBlocked)
	}
	if resBlocked.Slots["q1"].Predicted != "opt_a" {
		t.Fatalf("expected opt_a, got %q", resBlocked.Slots["q1"].Predicted)
	}

	// 2. Mixed request (1 blocked choice + 1 score question) launches speculative Stage 2 only for non-blocked slot
	var seenMu sync.Mutex
	var seenSpecQuestionIDs []string
	cfgMixed := DefaultCascadeConfig()
	cfgMixed.SpeculativeStage2 = true
	orchMixed, err := NewCascadeOrchestrator(
		CascadeBackendFunc(func(_ context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
			return map[string]map[string]float64{
				"blocked_choice": {"opt_a": 0.60, "opt_b": 0.40},
				"score_slot":     {"1": 0.55, "2": 0.45},
			}, nil
		}),
		CascadeBackendFunc(func(_ context.Context, _ any, qs []ChoiceQuestion) (map[string]map[string]float64, error) {
			seenMu.Lock()
			for _, q := range qs {
				seenSpecQuestionIDs = append(seenSpecQuestionIDs, q.ID)
			}
			seenMu.Unlock()
			return map[string]map[string]float64{
				"score_slot": {"1": 0.15, "2": 0.85},
			}, nil
		}),
		cfgMixed,
	)
	if err != nil {
		t.Fatalf("NewCascadeOrchestrator mixed failed: %v", err)
	}
	resMixed, err := orchMixed.Evaluate(context.Background(), "req_mixed", []ChoiceQuestion{
		{
			ID:               "blocked_choice",
			Kind:             QuestionKindChoice,
			Options:          []string{"opt_a", "opt_b"},
			DomainCompetence: &negProf,
		},
		{
			ID:               "score_slot",
			Kind:             QuestionKindScore,
			Options:          []string{"1", "2"},
			DomainCompetence: &negProf,
		},
	})
	if err != nil {
		t.Fatalf("Evaluate mixed failed: %v", err)
	}
	if len(seenSpecQuestionIDs) != 1 || seenSpecQuestionIDs[0] != "score_slot" {
		t.Fatalf("expected speculative Stage 2 only on score_slot, got %v", seenSpecQuestionIDs)
	}
	if resMixed.Slots["blocked_choice"].Escalated || resMixed.Slots["blocked_choice"].Predicted != "opt_a" {
		t.Fatalf("expected blocked_choice to remain at Stage 1 opt_a, got %+v", resMixed.Slots["blocked_choice"])
	}
	if !resMixed.Slots["score_slot"].Escalated || resMixed.Slots["score_slot"].Predicted != "2" {
		t.Fatalf("expected score_slot to escalate to '2', got %+v", resMixed.Slots["score_slot"])
	}

	// 3. Uniform Stage-1 distribution (|p_i - 1/K| <= 1e-6) rescues via two-phase (0.0, 1.0)
	// even when IsNegativeTransfer is tripped and UseDawidSkenePoEWeights is enabled
	var rescueCalls int32
	cfgRescue := DefaultCascadeConfig()
	cfgRescue.SpeculativeStage2 = true
	cfgRescue.DomainCompetence = &negProf
	cfgRescue.UseDawidSkenePoEWeights = true
	orchRescue, err := NewCascadeOrchestrator(
		CascadeBackendFunc(func(_ context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
			return map[string]map[string]float64{"q1": {"opt_a": 0.5000004, "opt_b": 0.4999996}}, nil
		}),
		CascadeBackendFunc(func(_ context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
			atomic.AddInt32(&rescueCalls, 1)
			return map[string]map[string]float64{"q1": {"opt_a": 0.15, "opt_b": 0.85}}, nil
		}),
		cfgRescue,
	)
	if err != nil {
		t.Fatalf("NewCascadeOrchestrator rescue failed: %v", err)
	}
	resRescue, err := orchRescue.Evaluate(context.Background(), "req_rescue", []ChoiceQuestion{baseQ})
	if err != nil {
		t.Fatalf("Evaluate rescue failed: %v", err)
	}
	if atomic.LoadInt32(&rescueCalls) != 1 || !resRescue.Escalated {
		t.Fatalf("expected 1 two-phase rescue Stage-2 call, got calls=%d escalated=%v", rescueCalls, resRescue.Escalated)
	}
	if math.Abs(resRescue.Slots["q1"].Probabilities["opt_b"]-0.85) > 1e-12 {
		t.Fatalf("expected exact (0.0, 1.0) Stage-2 rescue prob 0.85, got %v", resRescue.Slots["q1"].Probabilities["opt_b"])
	}

	// Just outside 1e-6 (0.500002 / 0.499998) does NOT trigger uniform rescue under negative transfer
	var nearCalls int32
	orchNear, err := NewCascadeOrchestrator(
		CascadeBackendFunc(func(_ context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
			return map[string]map[string]float64{"q1": {"opt_a": 0.500002, "opt_b": 0.499998}}, nil
		}),
		CascadeBackendFunc(func(_ context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
			atomic.AddInt32(&nearCalls, 1)
			return map[string]map[string]float64{"q1": {"opt_a": 0.10, "opt_b": 0.90}}, nil
		}),
		cfgBlocked,
	)
	if err != nil {
		t.Fatalf("NewCascadeOrchestrator near failed: %v", err)
	}
	resNear, err := orchNear.Evaluate(context.Background(), "req_near", []ChoiceQuestion{baseQ})
	if err != nil || atomic.LoadInt32(&nearCalls) != 0 || resNear.Escalated {
		t.Fatalf("expected 0 calls outside 1e-6 tolerance, got calls=%d escalated=%v err=%v", nearCalls, resNear.Escalated, err)
	}

	// 4. Uniform Stage-2 fallback while Stage 1 is non-uniform preserves Stage 1 (1.0, 0.0)
	for _, usePoE := range []bool{false, true} {
		cfgPreserve := DefaultCascadeConfig()
		cfgPreserve.DomainCompetence = &posProf
		cfgPreserve.UseDawidSkenePoEWeights = usePoE
		orchPreserve, pErr := NewCascadeOrchestrator(
			CascadeBackendFunc(func(_ context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
				return map[string]map[string]float64{"q1": {"opt_a": 0.62, "opt_b": 0.38}}, nil
			}),
			CascadeBackendFunc(func(_ context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
				return map[string]map[string]float64{"q1": {"opt_a": 0.50, "opt_b": 0.50}}, nil
			}),
			cfgPreserve,
		)
		if pErr != nil {
			t.Fatalf("NewCascadeOrchestrator preserve failed: %v", pErr)
		}
		resPreserve, pErr := orchPreserve.Evaluate(context.Background(), "req_preserve", []ChoiceQuestion{baseQ})
		if pErr != nil {
			t.Fatalf("Evaluate preserve failed: %v", pErr)
		}
		if resPreserve.Slots["q1"].Predicted != "opt_a" ||
			math.Abs(resPreserve.Slots["q1"].Probabilities["opt_a"]-0.62) > 1e-12 ||
			math.Abs(resPreserve.Slots["q1"].Probabilities["opt_b"]-0.38) > 1e-12 {
			t.Fatalf("usePoE=%v: expected preserved Stage-1 {0.62, 0.38}, got %+v", usePoE, resPreserve.Slots["q1"].Probabilities)
		}
	}
}

func TestCascadeContextCancellationAndVoIConfigBounds(t *testing.T) {
	baseQ := ChoiceQuestion{
		ID:      "act",
		Kind:    QuestionKindChoice,
		Options: []string{"allow", "deny"},
	}

	// 1. Speculative fast-exit cancels child stage2Ctx without mutating caller ctx
	stage2Started := make(chan struct{})
	stage2Canceled := make(chan struct{})
	cfgSpec := DefaultCascadeConfig()
	cfgSpec.SpeculativeStage2 = true
	orchFast, err := NewCascadeOrchestrator(
		CascadeBackendFunc(func(_ context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
			<-stage2Started
			return map[string]map[string]float64{"act": {"allow": 0.99, "deny": 0.01}}, nil
		}),
		CascadeBackendFunc(func(s2Ctx context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
			close(stage2Started)
			<-s2Ctx.Done()
			close(stage2Canceled)
			return nil, s2Ctx.Err()
		}),
		cfgSpec,
	)
	if err != nil {
		t.Fatalf("NewCascadeOrchestrator fast failed: %v", err)
	}
	callerCtx, callerCancel := context.WithCancel(context.Background())
	defer callerCancel()

	fastRes, err := orchFast.Evaluate(callerCtx, "fast_exit", []ChoiceQuestion{baseQ})
	if err != nil {
		t.Fatalf("fast-exit Evaluate failed: %v", err)
	}
	if fastRes.Escalated || fastRes.Slots["act"].Predicted != "allow" {
		t.Fatalf("unexpected fast-exit result: %+v", fastRes)
	}
	if callerCtx.Err() != nil {
		t.Fatalf("caller ctx must not be canceled on fast-exit, got %v", callerCtx.Err())
	}
	select {
	case <-stage2Canceled:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("expected speculative Stage-2 child context to be canceled on fast-exit")
	}

	// 2. Pre-canceled and mid-flight canceled ctx propagate ctx.Err() in both two-phase and speculative modes
	preCtx, preCancel := context.WithCancel(context.Background())
	preCancel()
	var s1Count int32
	orchCancel, err := NewCascadeOrchestrator(
		CascadeBackendFunc(func(_ context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
			atomic.AddInt32(&s1Count, 1)
			return map[string]map[string]float64{"act": {"allow": 0.6, "deny": 0.4}}, nil
		}),
		CascadeBackendFunc(func(c context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
			<-c.Done()
			return nil, c.Err()
		}),
		DefaultCascadeConfig(),
	)
	if err != nil {
		t.Fatalf("NewCascadeOrchestrator cancel failed: %v", err)
	}
	if _, cErr := orchCancel.Evaluate(preCtx, "pre", []ChoiceQuestion{baseQ}); !errors.Is(cErr, context.Canceled) {
		t.Fatalf("expected context.Canceled on pre-canceled ctx, got %v", cErr)
	}
	if atomic.LoadInt32(&s1Count) != 0 {
		t.Fatalf("expected 0 Stage-1 calls on pre-canceled ctx, got %d", s1Count)
	}

	for _, speculative := range []bool{false, true} {
		cfgMid := DefaultCascadeConfig()
		cfgMid.SpeculativeStage2 = speculative
		orchMid, mErr := NewCascadeOrchestrator(
			CascadeBackendFunc(func(_ context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
				return map[string]map[string]float64{"act": {"allow": 0.58, "deny": 0.42}}, nil
			}),
			CascadeBackendFunc(func(c context.Context, _ any, _ []ChoiceQuestion) (map[string]map[string]float64, error) {
				select {
				case <-c.Done():
					return nil, c.Err()
				case <-time.After(2 * time.Second):
					return map[string]map[string]float64{"act": {"allow": 0.1, "deny": 0.9}}, nil
				}
			}),
			cfgMid,
		)
		if mErr != nil {
			t.Fatalf("NewCascadeOrchestrator mid failed: %v", mErr)
		}
		midCtx, midCancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
		_, evalErr := orchMid.Evaluate(midCtx, "mid", []ChoiceQuestion{baseQ})
		midCancel()
		if !errors.Is(evalErr, context.DeadlineExceeded) {
			t.Fatalf("speculative=%v: expected context.DeadlineExceeded, got %v", speculative, evalErr)
		}
	}

	// 3. VoIStage2ExpectedAccuracy accepts full [0.0, 1.0] (including < 0.50) and rejects out-of-range
	for _, validAcc := range []float64{0.0, 0.25, 0.49, 0.50, 0.95, 1.0} {
		cfgVoI := DefaultCascadeConfig()
		cfgVoI.EnableVoIGate = true
		cfgVoI.VoIErrorCost = 1.0
		cfgVoI.VoIStage2ExpectedAccuracy = validAcc
		if vErr := cfgVoI.Validate(); vErr != nil {
			t.Fatalf("expected VoIStage2ExpectedAccuracy=%v to be valid, got %v", validAcc, vErr)
		}
	}
	for _, badAcc := range []float64{-0.01, 1.01, math.NaN(), math.Inf(1)} {
		cfgBad := DefaultCascadeConfig()
		cfgBad.VoIStage2ExpectedAccuracy = badAcc
		if vErr := cfgBad.Validate(); vErr == nil {
			t.Fatalf("expected VoIStage2ExpectedAccuracy=%v to fail validation", badAcc)
		}
	}
}
