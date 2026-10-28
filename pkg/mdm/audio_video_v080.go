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
	"sort"
	"strings"
)

// TurnTakingActions enumerates the 5 canonical full-duplex Voice Activity Projection (VAP) actions.
var TurnTakingActions = [5]string{
	"hold_floor",
	"yield_turn",
	"backchannel",
	"barge_in_interrupt",
	"wait_user_thinking",
}

// VoiceActivityProjectionConfig parameterizes cost-sensitive full-duplex turn-taking decisions.
type VoiceActivityProjectionConfig struct {
	FalseBargeInCost         float64
	FalseYieldOnThinkingCost float64
	MissedBargeInCost        float64
	TurnLatencyCost          float64
	BackchannelMismatchCost  float64
	PauseThinkingLowMs       float64
	PauseThinkingHighMs      float64
	PauseYieldThresholdMs    float64
	PriorWeight              float64
	HesitationThreshold      float64
	ConformalQHat            float64
}

// DefaultVoiceActivityProjectionConfig returns the default VAP turn-taking configuration.
func DefaultVoiceActivityProjectionConfig() VoiceActivityProjectionConfig {
	return VoiceActivityProjectionConfig{
		FalseBargeInCost:         6.0,
		FalseYieldOnThinkingCost: 2.8,
		MissedBargeInCost:        2.2,
		TurnLatencyCost:          1.0,
		BackchannelMismatchCost:  1.2,
		PauseThinkingLowMs:       180.0,
		PauseThinkingHighMs:      650.0,
		PauseYieldThresholdMs:    700.0,
		PriorWeight:              0.45,
		HesitationThreshold:      0.28,
		ConformalQHat:            0.85,
	}
}

// TurnTakingDecisionResult holds the calibrated full-duplex turn-taking decision.
type TurnTakingDecisionResult struct {
	SelectedAction           string
	RawArgmaxAction          string
	CalibratedProbabilities  map[string]float64
	ExpectedLosses           map[string]float64
	ConformalSet             []string
	NormalizedHesitation     float64
	BargeInBlockedByGuard    bool
	EscalateToDeliberative   bool
}

func normalizeDistGo(probs map[string]float64) (map[string]float64, error) {
	if len(probs) == 0 {
		return nil, fmt.Errorf("mdm: probability map must be non-empty")
	}
	out := make(map[string]float64, len(probs))
	total := 0.0
	for k, v := range probs {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0.0 {
			return nil, fmt.Errorf("mdm: invalid probability %v for key %q", v, k)
		}
		out[k] = v
		total += v
	}
	if total <= 0.0 {
		return nil, fmt.Errorf("mdm: probability sum must be > 0; uniform 1/K fallbacks are prohibited")
	}
	for k, v := range out {
		out[k] = v / total
	}
	return out, nil
}

func normalizedEntropyGo(probs map[string]float64) float64 {
	if len(probs) <= 1 {
		return 0.0
	}
	h := 0.0
	for _, p := range probs {
		if p > 1e-12 {
			h -= p * math.Log(p)
		}
	}
	return math.Max(0.0, math.Min(1.0, h/math.Log(float64(len(probs)))))
}

// ComputeVAPProsodicPriorLogits computes zero-centered acoustic-prosodic log-prior logits over turn-taking actions.
func ComputeVAPProsodicPriorLogits(
	pauseMs float64,
	rmsSlopeDB float64,
	f0DeltaHz float64,
	semanticCompleteProb float64,
	userOverlapActive bool,
	cfg VoiceActivityProjectionConfig,
) (map[string]float64, error) {
	if math.IsNaN(pauseMs) || math.IsInf(pauseMs, 0) || pauseMs < 0.0 {
		return nil, fmt.Errorf("mdm: pauseMs must be finite and >= 0, got %v", pauseMs)
	}
	if math.IsNaN(rmsSlopeDB) || math.IsInf(rmsSlopeDB, 0) || math.IsNaN(f0DeltaHz) || math.IsInf(f0DeltaHz, 0) {
		return nil, fmt.Errorf("mdm: rmsSlopeDB and f0DeltaHz must be finite")
	}
	if math.IsNaN(semanticCompleteProb) || math.IsInf(semanticCompleteProb, 0) || semanticCompleteProb < 0.0 || semanticCompleteProb > 1.0 {
		return nil, fmt.Errorf("mdm: semanticCompleteProb must lie in [0, 1], got %v", semanticCompleteProb)
	}

	inThinkingPause := pauseMs >= cfg.PauseThinkingLowMs && pauseMs <= cfg.PauseThinkingHighMs
	longPause := pauseMs >= cfg.PauseYieldThresholdMs
	fallingProsody := (f0DeltaHz <= -12.0) && (rmsSlopeDB <= -2.0)
	flatProsody := (math.Abs(f0DeltaHz) <= 10.0) && (math.Abs(rmsSlopeDB) <= 2.0)

	logits := make(map[string]float64, 5)
	yieldScore := 1.8 * (semanticCompleteProb - 0.5)
	if longPause {
		yieldScore += math.Min(1.6, 0.8+(pauseMs-cfg.PauseYieldThresholdMs)/600.0)
	}
	if fallingProsody {
		yieldScore += 0.9
	}
	if userOverlapActive {
		yieldScore -= 1.2
	}
	logits["yield_turn"] = yieldScore

	thinkScore := 1.6 * (0.55 - semanticCompleteProb)
	if inThinkingPause {
		thinkScore += 1.15
	}
	if flatProsody {
		thinkScore += 0.75
	}
	if longPause && semanticCompleteProb >= 0.80 {
		thinkScore -= 1.1
	}
	logits["wait_user_thinking"] = thinkScore

	holdScore := 0.0
	if pauseMs < cfg.PauseThinkingLowMs && !userOverlapActive {
		holdScore += 1.10 + 0.6*(1.0-semanticCompleteProb)
	} else {
		holdScore -= 0.75
	}
	logits["hold_floor"] = holdScore

	bcScore := -0.35
	if pauseMs >= 120.0 && pauseMs <= 450.0 && semanticCompleteProb >= 0.35 && semanticCompleteProb <= 0.72 && !userOverlapActive {
		bcScore += 0.95
	}
	logits["backchannel"] = bcScore

	bargeScore := -1.25
	if userOverlapActive {
		bargeScore += 1.85 + math.Max(0.0, math.Min(1.2, rmsSlopeDB/4.0))
	} else if rmsSlopeDB >= 3.5 && pauseMs <= 100.0 {
		bargeScore += 0.85
	}
	logits["barge_in_interrupt"] = bargeScore

	mean := 0.0
	for _, v := range logits {
		mean += v
	}
	mean /= 5.0
	for k, v := range logits {
		logits[k] = v - mean
	}
	return logits, nil
}

// EvaluateDuplexTurnTakingDecision evaluates a calibrated full-duplex turn-taking action with asymmetric cost and conformal guards.
func EvaluateDuplexTurnTakingDecision(
	stage1Probs map[string]float64,
	pauseMs float64,
	rmsSlopeDB float64,
	f0DeltaHz float64,
	semanticCompleteProb float64,
	userOverlapActive bool,
	cfg VoiceActivityProjectionConfig,
) (TurnTakingDecisionResult, error) {
	normRaw, err := normalizeDistGo(stage1Probs)
	if err != nil {
		return TurnTakingDecisionResult{}, err
	}
	for _, act := range TurnTakingActions {
		if _, ok := normRaw[act]; !ok {
			return TurnTakingDecisionResult{}, fmt.Errorf("mdm: stage1Probs missing required action %q", act)
		}
	}

	rawArgmax := TurnTakingActions[0]
	for _, act := range TurnTakingActions[1:] {
		if normRaw[act] > normRaw[rawArgmax] {
			rawArgmax = act
		}
	}
	hRaw := normalizedEntropyGo(normRaw)

	priorLogits, err := ComputeVAPProsodicPriorLogits(pauseMs, rmsSlopeDB, f0DeltaHz, semanticCompleteProb, userOverlapActive, cfg)
	if err != nil {
		return TurnTakingDecisionResult{}, err
	}
	optKeys := TurnTakingActions[:]
	fusedProbs, _, _, _, _ := FusePriorWithGate(normRaw, priorLogits, optKeys, 1.0, cfg.PriorWeight, cfg.HesitationThreshold, "proportional")

	confCfg := DefaultConformalConfig()
	confCfg.Method = "raps"
	confCfg.LamReg = 0.02
	confCfg.KReg = 1
	confRes, err := ConformalPredictionSet(fusedProbs, cfg.ConformalQHat, confCfg)
	if err != nil {
		return TurnTakingDecisionResult{}, err
	}
	hCal := normalizedEntropyGo(fusedProbs)

	lossMatrix := make(map[string]map[string]float64, 5)
	for _, a := range TurnTakingActions {
		lossMatrix[a] = make(map[string]float64, 5)
		for _, b := range TurnTakingActions {
			if a == b {
				lossMatrix[a][b] = 0.0
			} else {
				lossMatrix[a][b] = 1.0
			}
		}
	}
	lossMatrix["barge_in_interrupt"]["hold_floor"] = cfg.FalseBargeInCost
	lossMatrix["barge_in_interrupt"]["wait_user_thinking"] = cfg.FalseBargeInCost
	lossMatrix["barge_in_interrupt"]["yield_turn"] = cfg.FalseBargeInCost * 0.55
	lossMatrix["barge_in_interrupt"]["backchannel"] = cfg.FalseBargeInCost * 0.65

	lossMatrix["yield_turn"]["wait_user_thinking"] = cfg.FalseYieldOnThinkingCost
	lossMatrix["yield_turn"]["hold_floor"] = cfg.FalseYieldOnThinkingCost
	lossMatrix["yield_turn"]["barge_in_interrupt"] = cfg.MissedBargeInCost

	lossMatrix["wait_user_thinking"]["yield_turn"] = cfg.TurnLatencyCost
	lossMatrix["wait_user_thinking"]["barge_in_interrupt"] = cfg.MissedBargeInCost
	lossMatrix["hold_floor"]["barge_in_interrupt"] = cfg.MissedBargeInCost
	lossMatrix["hold_floor"]["yield_turn"] = cfg.TurnLatencyCost * 1.15

	for _, b := range TurnTakingActions {
		if b != "backchannel" {
			lossMatrix["backchannel"][b] = cfg.BackchannelMismatchCost
		}
	}

	expectedLosses := make(map[string]float64, 5)
	for _, cand := range TurnTakingActions {
		sumLoss := 0.0
		for _, truth := range TurnTakingActions {
			sumLoss += lossMatrix[cand][truth] * fusedProbs[truth]
		}
		expectedLosses[cand] = sumLoss
	}

	minRiskAction := TurnTakingActions[0]
	for _, act := range TurnTakingActions[1:] {
		if expectedLosses[act] < expectedLosses[minRiskAction]-1e-12 ||
			(math.Abs(expectedLosses[act]-expectedLosses[minRiskAction]) <= 1e-12 && fusedProbs[act] > fusedProbs[minRiskAction]) {
			minRiskAction = act
		}
	}

	inConfSet := func(target string) bool {
		for _, s := range confRes.PredictionSet {
			if s == target {
				return true
			}
		}
		return false
	}

	selected := minRiskAction
	bargeBlocked := false
	if selected == "barge_in_interrupt" {
		if len(confRes.PredictionSet) > 1 && (inConfSet("hold_floor") || inConfSet("wait_user_thinking")) {
			bargeBlocked = true
			if fusedProbs["wait_user_thinking"] >= fusedProbs["hold_floor"] {
				selected = "wait_user_thinking"
			} else {
				selected = "hold_floor"
			}
		}
	} else if selected == "yield_turn" &&
		pauseMs >= cfg.PauseThinkingLowMs &&
		pauseMs <= cfg.PauseThinkingHighMs &&
		inConfSet("wait_user_thinking") &&
		fusedProbs["wait_user_thinking"] >= 0.25 {
		bargeBlocked = true
		selected = "wait_user_thinking"
	}

	escalate := len(confRes.PredictionSet) > 2 || hCal >= cfg.HesitationThreshold*1.45 || hRaw >= 0.65
	return TurnTakingDecisionResult{
		SelectedAction:          selected,
		RawArgmaxAction:         rawArgmax,
		CalibratedProbabilities: fusedProbs,
		ExpectedLosses:          expectedLosses,
		ConformalSet:            confRes.PredictionSet,
		NormalizedHesitation:    hCal,
		BargeInBlockedByGuard:   bargeBlocked,
		EscalateToDeliberative:  escalate,
	}, nil
}

// ConformalMartingaleStepResult records one step of a Conformal Test Martingale.
type ConformalMartingaleStepResult struct {
	StepIndex           int
	NonconformityScore  float64
	ConformalPValue     float64
	BettingFraction     float64
	MartingaleValue     float64
	LogMartingaleValue  float64
	DriftOrAnomalyAlarm bool
}

// ConformalMartingaleTracker maintains an anytime-valid test martingale bounded by Ville's inequality.
type ConformalMartingaleTracker struct {
	calibScores           []float64
	AlphaAlarm            float64
	AlarmThreshold        float64
	MaxBettingFraction    float64
	BettingAdaptationRate float64
	stepIndex             int
	logMartingale         float64
	emaDeficit            float64
	alarmTriggered        bool
}

// NewConformalMartingaleTracker creates a validated ConformalMartingaleTracker.
func NewConformalMartingaleTracker(
	calibrationScores []float64,
	alphaAlarm float64,
	maxBettingFraction float64,
	bettingAdaptationRate float64,
) (*ConformalMartingaleTracker, error) {
	if len(calibrationScores) == 0 {
		return nil, fmt.Errorf("mdm: calibrationScores must be non-empty")
	}
	cleaned := make([]float64, 0, len(calibrationScores))
	for _, s := range calibrationScores {
		if math.IsNaN(s) || math.IsInf(s, 0) || s < 0.0 {
			return nil, fmt.Errorf("mdm: invalid calibration score %v", s)
		}
		cleaned = append(cleaned, s)
	}
	sort.Float64s(cleaned)
	if math.IsNaN(alphaAlarm) || math.IsInf(alphaAlarm, 0) || alphaAlarm <= 0.0 || alphaAlarm >= 1.0 {
		return nil, fmt.Errorf("mdm: alphaAlarm must lie in (0, 1), got %v", alphaAlarm)
	}
	if math.IsNaN(maxBettingFraction) || math.IsInf(maxBettingFraction, 0) || maxBettingFraction <= 0.0 || maxBettingFraction >= 2.0 {
		return nil, fmt.Errorf("mdm: maxBettingFraction must lie in (0, 2), got %v", maxBettingFraction)
	}
	if math.IsNaN(bettingAdaptationRate) || math.IsInf(bettingAdaptationRate, 0) || bettingAdaptationRate <= 0.0 || bettingAdaptationRate > 1.0 {
		return nil, fmt.Errorf("mdm: bettingAdaptationRate must lie in (0, 1], got %v", bettingAdaptationRate)
	}
	return &ConformalMartingaleTracker{
		calibScores:           cleaned,
		AlphaAlarm:            alphaAlarm,
		AlarmThreshold:        1.0 / alphaAlarm,
		MaxBettingFraction:    maxBettingFraction,
		BettingAdaptationRate: bettingAdaptationRate,
		emaDeficit:            0.25,
	}, nil
}

// ComputeConformalPValue computes the smoothed conformal p-value against the reference bag.
func (t *ConformalMartingaleTracker) ComputeConformalPValue(score float64, tieBreakU float64) (float64, error) {
	if math.IsNaN(score) || math.IsInf(score, 0) || score < 0.0 {
		return 0, fmt.Errorf("mdm: score must be finite and >= 0, got %v", score)
	}
	if math.IsNaN(tieBreakU) || math.IsInf(tieBreakU, 0) || tieBreakU < 0.0 || tieBreakU > 1.0 {
		return 0, fmt.Errorf("mdm: tieBreakU must lie in [0, 1], got %v", tieBreakU)
	}
	n := len(t.calibScores)
	greater := 0
	equal := 0
	for _, ref := range t.calibScores {
		if ref > score+1e-12 {
			greater++
		} else if math.Abs(ref-score) <= 1e-12 {
			equal++
		}
	}
	pVal := (float64(greater) + tieBreakU*float64(equal+1)) / float64(n+1)
	minP := 1.0 / float64(2*(n+1))
	return math.Max(minP, math.Min(1.0, pVal)), nil
}

// Step processes one streaming nonconformity score and updates the test martingale.
func (t *ConformalMartingaleTracker) Step(score float64, tieBreakU float64) (ConformalMartingaleStepResult, error) {
	pVal, err := t.ComputeConformalPValue(score, tieBreakU)
	if err != nil {
		return ConformalMartingaleStepResult{}, err
	}
	kappaT := math.Max(0.0, math.Min(t.MaxBettingFraction, t.MaxBettingFraction*math.Tanh(2.4*math.Max(0.0, t.emaDeficit))))
	mult := math.Max(1e-6, 1.0+kappaT*(0.5-pVal))
	t.logMartingale += math.Log(mult)
	mVal := math.Exp(math.Min(80.0, t.logMartingale))

	deficit := 0.5 - pVal
	t.emaDeficit = (1.0-t.BettingAdaptationRate)*t.emaDeficit + t.BettingAdaptationRate*deficit
	t.stepIndex++
	if mVal >= t.AlarmThreshold {
		t.alarmTriggered = true
	}
	return ConformalMartingaleStepResult{
		StepIndex:           t.stepIndex,
		NonconformityScore:  score,
		ConformalPValue:     pVal,
		BettingFraction:     kappaT,
		MartingaleValue:     mVal,
		LogMartingaleValue:  t.logMartingale,
		DriftOrAnomalyAlarm: t.alarmTriggered,
	}, nil
}

// StreamingEarlyExitResult summarizes sequential early-exit video/audio evaluation.
type StreamingEarlyExitResult struct {
	SelectedClass          string
	ExitStep               int
	TotalSteps             int
	ComputeSavedFraction   float64
	CertifiedEarlyExit     bool
	AnomalyAlarmTriggered  bool
	PosteriorProbabilities map[string]float64
	ConformalSetAtExit     []string
	MartingaleTrajectory   []float64
	HesitationTrajectory   []float64
}

// EvaluateStreamingEarlyExit evaluates sequential video/audio chunks with Conformal Martingale drift protection and early exit.
func EvaluateStreamingEarlyExit(
	chunkProbStream []map[string]float64,
	calibNonconformityScores []float64,
	conformalQHat float64,
	hesitationExitThreshold float64,
	minChunksBeforeExit int,
	stabilityStreakRequired int,
	martingaleAlarmAlpha float64,
	temporalDecay float64,
) (StreamingEarlyExitResult, error) {
	if len(chunkProbStream) == 0 {
		return StreamingEarlyExitResult{}, fmt.Errorf("mdm: chunkProbStream must be non-empty")
	}
	if minChunksBeforeExit < 1 || stabilityStreakRequired < 1 {
		return StreamingEarlyExitResult{}, fmt.Errorf("mdm: minChunksBeforeExit and stabilityStreakRequired must be >= 1")
	}
	if math.IsNaN(temporalDecay) || temporalDecay <= 0.0 || temporalDecay > 1.0 {
		return StreamingEarlyExitResult{}, fmt.Errorf("mdm: temporalDecay must lie in (0, 1]")
	}

	firstNorm, err := normalizeDistGo(chunkProbStream[0])
	if err != nil {
		return StreamingEarlyExitResult{}, err
	}
	if len(firstNorm) < 2 {
		return StreamingEarlyExitResult{}, fmt.Errorf("mdm: chunk probabilities must contain >= 2 classes")
	}
	classes := make([]string, 0, len(firstNorm))
	for k := range firstNorm {
		classes = append(classes, k)
	}
	sort.Strings(classes)

	tracker, err := NewConformalMartingaleTracker(calibNonconformityScores, martingaleAlarmAlpha, 1.65, 0.45)
	if err != nil {
		return StreamingEarlyExitResult{}, err
	}

	accumLogOdds := make(map[string]float64, len(classes))
	martingaleTraj := make([]float64, 0, len(chunkProbStream))
	hesitationTraj := make([]float64, 0, len(chunkProbStream))

	prevLeader := ""
	leaderStreak := 0
	exitStep := len(chunkProbStream)
	certifiedExit := false
	posteriorAtExit := firstNorm
	confSetAtExit := append([]string(nil), classes...)

	confCfg := DefaultConformalConfig()
	confCfg.Method = "raps"
	confCfg.LamReg = 0.02
	confCfg.KReg = 1

	for idx, rawChunk := range chunkProbStream {
		stepNum := idx + 1
		normChunk, err := normalizeDistGo(rawChunk)
		if err != nil {
			return StreamingEarlyExitResult{}, err
		}
		if len(normChunk) != len(classes) {
			return StreamingEarlyExitResult{}, fmt.Errorf("mdm: mismatched class keys at chunk %d", idx)
		}
		maxP := 0.0
		for _, c := range classes {
			v, ok := normChunk[c]
			if !ok {
				return StreamingEarlyExitResult{}, fmt.Errorf("mdm: missing class %q at chunk %d", c, idx)
			}
			if v > maxP {
				maxP = v
			}
		}
		hChunk := normalizedEntropyGo(normChunk)
		streamNonconf := math.Min(1.0, (1.0-maxP)+0.35*hChunk)
		mStep, err := tracker.Step(streamNonconf, 0.5)
		if err != nil {
			return StreamingEarlyExitResult{}, err
		}
		martingaleTraj = append(martingaleTraj, mStep.MartingaleValue)

		chunkRel := math.Max(0.15, 1.0-0.65*hChunk)
		maxLogit := math.Inf(-1)
		for _, c := range classes {
			pc := math.Max(1e-6, math.Min(1.0-1e-6, normChunk[c]))
			accumLogOdds[c] = temporalDecay*accumLogOdds[c] + chunkRel*math.Log(pc)
			if accumLogOdds[c] > maxLogit {
				maxLogit = accumLogOdds[c]
			}
		}

		posterior := make(map[string]float64, len(classes))
		denom := 0.0
		for _, c := range classes {
			ev := math.Exp(accumLogOdds[c] - maxLogit)
			posterior[c] = ev
			denom += ev
		}
		for _, c := range classes {
			posterior[c] /= denom
		}
		hPost := normalizedEntropyGo(posterior)
		hesitationTraj = append(hesitationTraj, hPost)

		confRes, err := ConformalPredictionSet(posterior, conformalQHat, confCfg)
		if err != nil {
			return StreamingEarlyExitResult{}, err
		}
		leader := classes[0]
		for _, c := range classes[1:] {
			if posterior[c] > posterior[leader] {
				leader = c
			}
		}
		if leader == prevLeader {
			leaderStreak++
		} else {
			prevLeader = leader
			leaderStreak = 1
		}

		posteriorAtExit = posterior
		confSetAtExit = confRes.PredictionSet

		if stepNum >= minChunksBeforeExit &&
			len(confSetAtExit) == 1 &&
			hPost <= hesitationExitThreshold &&
			leaderStreak >= stabilityStreakRequired &&
			!tracker.alarmTriggered {
			exitStep = stepNum
			certifiedExit = true
			break
		}
	}

	totalSteps := len(chunkProbStream)
	savedFrac := float64(totalSteps-exitStep) / float64(totalSteps)
	selectedClass := classes[0]
	for _, c := range classes[1:] {
		if posteriorAtExit[c] > posteriorAtExit[selectedClass] {
			selectedClass = c
		}
	}
	return StreamingEarlyExitResult{
		SelectedClass:          selectedClass,
		ExitStep:               exitStep,
		TotalSteps:             totalSteps,
		ComputeSavedFraction:   savedFrac,
		CertifiedEarlyExit:     certifiedExit,
		AnomalyAlarmTriggered:  tracker.alarmTriggered,
		PosteriorProbabilities: posteriorAtExit,
		ConformalSetAtExit:     confSetAtExit,
		MartingaleTrajectory:   martingaleTraj,
		HesitationTrajectory:   hesitationTraj,
	}, nil
}

// AudioVisualAgreementResult holds the cross-modal AVEL agreement and spoof/off-screen classification.
type AudioVisualAgreementResult struct {
	Regime                  string
	SelectedEvent           string
	FusedProbabilities      map[string]float64
	JensenShannonDivergence float64
	AudioHesitation         float64
	VisualHesitation        float64
	TriangleViolationScore  float64
	EscalateToStage2        bool
}

func isSilentBackgroundKeyGo(k string) bool {
	switch strings.ToLower(strings.TrimSpace(k)) {
	case "background", "none", "silent", "silence", "ambient", "no_event", "off":
		return true
	default:
		return false
	}
}

// JensenShannonDivergenceBits computes normalized Jensen-Shannon divergence in [0, 1] (log2 basis).
func JensenShannonDivergenceBits(audioProbs, visualProbs map[string]float64) (float64, error) {
	pa, err := normalizeDistGo(audioProbs)
	if err != nil {
		return 0, err
	}
	pv, err := normalizeDistGo(visualProbs)
	if err != nil {
		return 0, err
	}
	if len(pa) != len(pv) {
		return 0, fmt.Errorf("mdm: audioProbs and visualProbs must have identical keys")
	}
	jsdNats := 0.0
	for k, a := range pa {
		v, ok := pv[k]
		if !ok {
			return 0, fmt.Errorf("mdm: missing key %q in visualProbs", k)
		}
		m := 0.5 * (a + v)
		if a > 1e-12 && m > 1e-12 {
			jsdNats += 0.5 * a * math.Log(a/m)
		}
		if v > 1e-12 && m > 1e-12 {
			jsdNats += 0.5 * v * math.Log(v/m)
		}
	}
	return math.Max(0.0, math.Min(1.0, jsdNats/math.Log(2.0))), nil
}

// ScoreAudioVisualTriangleConsistency computes the 3-slot spatio-temporal triangle violation mass in [0, 1].
func ScoreAudioVisualTriangleConsistency(
	audioProbs map[string]float64,
	visualProbs map[string]float64,
	syncSameSourceProb float64,
) (float64, error) {
	pa, err := normalizeDistGo(audioProbs)
	if err != nil {
		return 0, err
	}
	pv, err := normalizeDistGo(visualProbs)
	if err != nil {
		return 0, err
	}
	if len(pa) != len(pv) {
		return 0, fmt.Errorf("mdm: audioProbs and visualProbs must have identical keys")
	}
	if math.IsNaN(syncSameSourceProb) || syncSameSourceProb < 0.0 || syncSameSourceProb > 1.0 {
		return 0, fmt.Errorf("mdm: syncSameSourceProb must lie in [0, 1]")
	}

	var activeKeys []string
	for k := range pa {
		if _, ok := pv[k]; !ok {
			return 0, fmt.Errorf("mdm: missing key %q in visualProbs", k)
		}
		if !isSilentBackgroundKeyGo(k) {
			activeKeys = append(activeKeys, k)
		}
	}
	if len(activeKeys) == 0 {
		for k := range pa {
			activeKeys = append(activeKeys, k)
		}
	}

	falseSyncMass := 0.0
	for _, ca := range activeKeys {
		for _, cv := range activeKeys {
			if ca != cv {
				m := pa[ca] * pv[cv] * syncSameSourceProb
				if m > falseSyncMass {
					falseSyncMass = m
				}
			}
		}
	}
	desyncSpoofMass := 0.0
	for _, c := range activeKeys {
		m := pa[c] * pv[c] * (1.0 - syncSameSourceProb)
		if m > desyncSpoofMass {
			desyncSpoofMass = m
		}
	}
	return math.Max(0.0, math.Min(1.0, math.Max(falseSyncMass, desyncSpoofMass))), nil
}

// EvaluateAudioVisualEventAgreement classifies an audio-visual segment into synchronous, off-screen, silent, or spoof/conflict.
func EvaluateAudioVisualEventAgreement(
	audioProbs map[string]float64,
	visualProbs map[string]float64,
	syncSameSourceProb float64,
	jsdConflictThreshold float64,
	triangleViolationThreshold float64,
	confidentHesitationCeiling float64,
	uncertainHesitationFloor float64,
) (AudioVisualAgreementResult, error) {
	pa, err := normalizeDistGo(audioProbs)
	if err != nil {
		return AudioVisualAgreementResult{}, err
	}
	pv, err := normalizeDistGo(visualProbs)
	if err != nil {
		return AudioVisualAgreementResult{}, err
	}
	jsd, err := JensenShannonDivergenceBits(pa, pv)
	if err != nil {
		return AudioVisualAgreementResult{}, err
	}
	sTri, err := ScoreAudioVisualTriangleConsistency(pa, pv, syncSameSourceProb)
	if err != nil {
		return AudioVisualAgreementResult{}, err
	}

	keys := make([]string, 0, len(pa))
	for k := range pa {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	hAud := normalizedEntropyGo(pa)
	hVis := normalizedEntropyGo(pv)

	topAud := keys[0]
	topVis := keys[0]
	for _, k := range keys[1:] {
		if pa[k] > pa[topAud] {
			topAud = k
		}
		if pv[k] > pv[topVis] {
			topVis = k
		}
	}
	audIsBg := isSilentBackgroundKeyGo(topAud)
	visIsBg := isSilentBackgroundKeyGo(topVis)

	if !audIsBg && hAud <= confidentHesitationCeiling && (visIsBg || hVis >= uncertainHesitationFloor) {
		return AudioVisualAgreementResult{
			Regime:                  "audible_offscreen_event",
			SelectedEvent:           topAud,
			FusedProbabilities:      pa,
			JensenShannonDivergence: jsd,
			AudioHesitation:         hAud,
			VisualHesitation:        hVis,
			TriangleViolationScore:  sTri,
			EscalateToStage2:        false,
		}, nil
	}

	if !visIsBg && hVis <= confidentHesitationCeiling && (audIsBg || hAud >= uncertainHesitationFloor) {
		return AudioVisualAgreementResult{
			Regime:                  "visible_silent_event",
			SelectedEvent:           topVis,
			FusedProbabilities:      pv,
			JensenShannonDivergence: jsd,
			AudioHesitation:         hAud,
			VisualHesitation:        hVis,
			TriangleViolationScore:  sTri,
			EscalateToStage2:        false,
		}, nil
	}

	confidentDisagreement := topAud != topVis &&
		!audIsBg &&
		!visIsBg &&
		hAud <= confidentHesitationCeiling &&
		hVis <= confidentHesitationCeiling &&
		jsd >= jsdConflictThreshold

	if sTri >= triangleViolationThreshold || confidentDisagreement {
		wA := math.Max(0.1, 1.0-hAud)
		wV := math.Max(0.1, 1.0-hVis)
		wSum := wA + wV
		pooled := make(map[string]float64, len(keys))
		selected := keys[0]
		for _, k := range keys {
			pooled[k] = (wA*pa[k] + wV*pv[k]) / wSum
		}
		for _, k := range keys[1:] {
			if pooled[k] > pooled[selected] {
				selected = k
			}
		}
		return AudioVisualAgreementResult{
			Regime:                  "cross_modal_spoof_or_conflict",
			SelectedEvent:           selected,
			FusedProbabilities:      pooled,
			JensenShannonDivergence: jsd,
			AudioHesitation:         hAud,
			VisualHesitation:        hVis,
			TriangleViolationScore:  sTri,
			EscalateToStage2:        true,
		}, nil
	}

	wA := math.Max(0.15, 1.0-hAud)
	wV := math.Max(0.15, 1.0-hVis)
	logits := make(map[string]float64, len(keys))
	maxL := math.Inf(-1)
	for _, k := range keys {
		l := wA*math.Log(math.Max(1e-6, pa[k])) + wV*math.Log(math.Max(1e-6, pv[k]))
		logits[k] = l
		if l > maxL {
			maxL = l
		}
	}
	fused := make(map[string]float64, len(keys))
	denom := 0.0
	for _, k := range keys {
		ev := math.Exp(logits[k] - maxL)
		fused[k] = ev
		denom += ev
	}
	selected := keys[0]
	for _, k := range keys {
		fused[k] /= denom
		if fused[k] > fused[selected] {
			selected = k
		}
	}
	hFused := normalizedEntropyGo(fused)
	return AudioVisualAgreementResult{
		Regime:                  "synchronous_av_event",
		SelectedEvent:           selected,
		FusedProbabilities:      fused,
		JensenShannonDivergence: jsd,
		AudioHesitation:         hAud,
		VisualHesitation:        hVis,
		TriangleViolationScore:  sTri,
		EscalateToStage2:        hFused >= uncertainHesitationFloor,
	}, nil
}

// SpatioTemporalFrameCandidate represents a candidate video/audio frame slice for submodular selection.
type SpatioTemporalFrameCandidate struct {
	FrameIndex             int
	TimestampSec           float64
	HesitationScore        float64
	AcousticTransientScore float64
	VisualMotionScore      float64
	FeatureVector          []float64
}

// SpatioTemporalSelectionResult holds the chronologically ordered submodular keyframe selection.
type SpatioTemporalSelectionResult struct {
	SelectedIndices       []int
	ObjectiveValue        float64
	MarginalGains         []float64
	TemporalCoverageRatio float64
}

func cosineSimSliceGo(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}
	dot, na, nb := 0.0, 0.0, 0.0
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na <= 1e-12 || nb <= 1e-12 {
		return 0.0
	}
	return math.Max(0.0, math.Min(1.0, dot/(math.Sqrt(na)*math.Sqrt(nb))))
}

// SelectSubmodularSpatioTemporalFrames selects up to budgetB keyframes maximizing utility + facility coverage - redundancy.
func SelectSubmodularSpatioTemporalFrames(
	candidates []SpatioTemporalFrameCandidate,
	budgetB int,
	facilityWeight float64,
	redundancyPenalty float64,
	temporalKernelDecay float64,
) (SpatioTemporalSelectionResult, error) {
	if len(candidates) == 0 {
		return SpatioTemporalSelectionResult{}, fmt.Errorf("mdm: candidates must be non-empty")
	}
	if budgetB < 1 {
		return SpatioTemporalSelectionResult{}, fmt.Errorf("mdm: budgetB must be >= 1, got %d", budgetB)
	}
	n := len(candidates)
	targetK := budgetB
	if n < targetK {
		targetK = n
	}

	minT := candidates[0].TimestampSec
	maxT := candidates[0].TimestampSec
	for _, c := range candidates[1:] {
		if c.TimestampSec < minT {
			minT = c.TimestampSec
		}
		if c.TimestampSec > maxT {
			maxT = c.TimestampSec
		}
	}
	timeSpan := math.Max(1e-6, maxT-minT)

	utilities := make([]float64, n)
	for i, c := range candidates {
		h := math.Max(0.0, math.Min(1.0, c.HesitationScore))
		a := math.Max(0.0, math.Min(1.0, c.AcousticTransientScore))
		m := math.Max(0.0, math.Min(1.0, c.VisualMotionScore))
		utilities[i] = 0.45*h + 0.30*a + 0.25*m
	}

	simMatrix := make([][]float64, n)
	for i := range simMatrix {
		simMatrix[i] = make([]float64, n)
		simMatrix[i][i] = 1.0
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			fSim := cosineSimSliceGo(candidates[i].FeatureVector, candidates[j].FeatureVector)
			dtNorm := math.Abs(candidates[i].TimestampSec-candidates[j].TimestampSec) / timeSpan
			timeK := math.Exp(-temporalKernelDecay * dtNorm * float64(n))
			val := fSim * timeK
			simMatrix[i][j] = val
			simMatrix[j][i] = val
		}
	}

	selectedPos := make([]int, 0, targetK)
	selectedSet := make(map[int]bool, targetK)
	currentCov := make([]float64, n)
	marginalGains := make([]float64, 0, targetK)
	totalObj := 0.0

	for step := 0; step < targetK; step++ {
		bestIdx := -1
		bestGain := -1e18
		for candPos := 0; candPos < n; candPos++ {
			if selectedSet[candPos] {
				continue
			}
			covGain := 0.0
			for j := 0; j < n; j++ {
				if simMatrix[candPos][j] > currentCov[j] {
					covGain += simMatrix[candPos][j] - currentCov[j]
				}
			}
			covGain /= float64(n)
			redLoss := 0.0
			for _, s := range selectedPos {
				redLoss += simMatrix[candPos][s]
			}
			gain := utilities[candPos] + facilityWeight*covGain - redundancyPenalty*redLoss
			if gain > bestGain+1e-12 {
				bestGain = gain
				bestIdx = candPos
			}
		}
		if bestIdx < 0 {
			break
		}
		selectedPos = append(selectedPos, bestIdx)
		selectedSet[bestIdx] = true
		marginalGains = append(marginalGains, bestGain)
		totalObj += bestGain
		for j := 0; j < n; j++ {
			if simMatrix[bestIdx][j] > currentCov[j] {
				currentCov[j] = simMatrix[bestIdx][j]
			}
		}
	}

	type posGain struct {
		frameIdx int
		gain     float64
	}
	pairs := make([]posGain, len(selectedPos))
	for i, pos := range selectedPos {
		pairs[i] = posGain{frameIdx: candidates[pos].FrameIndex, gain: marginalGains[i]}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].frameIdx < pairs[j].frameIdx
	})
	chronoIndices := make([]int, len(pairs))
	chronoGains := make([]float64, len(pairs))
	for i, p := range pairs {
		chronoIndices[i] = p.frameIdx
		chronoGains[i] = p.gain
	}
	covSum := 0.0
	for _, c := range currentCov {
		covSum += c
	}
	return SpatioTemporalSelectionResult{
		SelectedIndices:       chronoIndices,
		ObjectiveValue:        totalObj,
		MarginalGains:         chronoGains,
		TemporalCoverageRatio: covSum / float64(n),
	}, nil
}
