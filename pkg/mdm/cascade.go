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
	"fmt"
	"math"
	"sort"
	"strings"
)

// DefaultUniformDistributionTolerance is the per-option |p_i - 1/K| tolerance for uniform fallback detection.
const DefaultUniformDistributionTolerance = 1e-6

// DefaultCascadeHesitationThreshold is the canonical normalized-entropy escalation threshold.
const DefaultCascadeHesitationThreshold = 0.16

// QuestionKind identifies the decision primitive of a cascade slot.
type QuestionKind string

const (
	// QuestionKindChoice denotes a categorical multi-option decision slot.
	QuestionKindChoice QuestionKind = "choice"
	// QuestionKindNoul denotes a binary Yes/No log-likelihood decision slot.
	QuestionKindNoul QuestionKind = "noul"
	// QuestionKindScore denotes an ordinal bucketed rubric score slot.
	QuestionKindScore QuestionKind = "score"
)

// ChoiceQuestion defines a decision slot for two-phase and speculative cascade orchestration.
type ChoiceQuestion struct {
	ID                 string                   `json:"id,omitempty"`
	Kind               QuestionKind             `json:"kind,omitempty"`
	Prompt             string                   `json:"prompt,omitempty"`
	Instructions       string                   `json:"instructions,omitempty"`
	Options            []string                 `json:"options"`
	OptionDescriptions map[string]string        `json:"option_descriptions,omitempty"`
	PriorLogits        map[string]float64       `json:"prior_logits,omitempty"`
	IncludeNoneOfAbove bool                     `json:"include_none_of_above,omitempty"`
	CatchAllKey        string                   `json:"catch_all_key,omitempty"`
	DomainCompetence   *DomainCompetenceProfile `json:"domain_competence,omitempty"`
}

func (q ChoiceQuestion) effectiveKind() QuestionKind {
	if q.Kind == "" {
		return QuestionKindChoice
	}
	return q.Kind
}

func cloneChoiceQuestion(q ChoiceQuestion) ChoiceQuestion {
	cloned := q
	if q.Options != nil {
		cloned.Options = append([]string(nil), q.Options...)
	}
	if q.OptionDescriptions != nil {
		cloned.OptionDescriptions = make(map[string]string, len(q.OptionDescriptions))
		for k, v := range q.OptionDescriptions {
			cloned.OptionDescriptions[k] = v
		}
	}
	if q.PriorLogits != nil {
		cloned.PriorLogits = make(map[string]float64, len(q.PriorLogits))
		for k, v := range q.PriorLogits {
			cloned.PriorLogits[k] = v
		}
	}
	if q.DomainCompetence != nil {
		profCopy := *q.DomainCompetence
		cloned.DomainCompetence = &profCopy
	}
	return cloned
}

// PruneChoiceTopK prunes a ChoiceQuestion to its topK highest-probability Stage-1 options
// while retaining OptionDescriptions and PriorLogits filtered strictly to the kept options.
func PruneChoiceTopK(q ChoiceQuestion, stage1Probs map[string]float64, topK int) (ChoiceQuestion, error) {
	if q.effectiveKind() != QuestionKindChoice {
		return ChoiceQuestion{}, fmt.Errorf("mdm: PruneChoiceTopK requires a choice question, got %q", q.Kind)
	}
	if len(q.Options) < 2 {
		return ChoiceQuestion{}, fmt.Errorf("mdm: ChoiceQuestion requires >= 2 options, got %d", len(q.Options))
	}
	if topK < 2 {
		return ChoiceQuestion{}, fmt.Errorf("mdm: topK must be >= 2, got %d", topK)
	}
	if len(stage1Probs) == 0 {
		return ChoiceQuestion{}, errors.New("mdm: stage1Probs must be non-empty")
	}

	seen := make(map[string]bool, len(q.Options))
	posByOpt := make(map[string]int, len(q.Options))
	scores := make(map[string]float64, len(q.Options))
	totalMass := 0.0
	for idx, opt := range q.Options {
		if strings.TrimSpace(opt) == "" {
			return ChoiceQuestion{}, errors.New("mdm: ChoiceQuestion.Options must not contain empty strings")
		}
		if seen[opt] {
			return ChoiceQuestion{}, fmt.Errorf("mdm: duplicate option %q in ChoiceQuestion", opt)
		}
		seen[opt] = true
		posByOpt[opt] = idx
		val := stage1Probs[opt]
		if math.IsNaN(val) || math.IsInf(val, 0) || val < 0 {
			return ChoiceQuestion{}, fmt.Errorf("mdm: invalid Stage-1 probability for option %q: %v", opt, val)
		}
		scores[opt] = val
		totalMass += val
	}
	if totalMass <= 0 || math.IsNaN(totalMass) || math.IsInf(totalMass, 0) {
		return ChoiceQuestion{}, errors.New("mdm: stage1Probs must have positive total mass over question options")
	}

	ranked := append([]string(nil), q.Options...)
	sort.Slice(ranked, func(i, j int) bool {
		si, sj := scores[ranked[i]], scores[ranked[j]]
		if si != sj {
			return si > sj
		}
		return posByOpt[ranked[i]] < posByOpt[ranked[j]]
	})

	limit := topK
	if limit > len(ranked) {
		limit = len(ranked)
	}
	kept := append([]string(nil), ranked[:limit]...)

	pruned := ChoiceQuestion{
		ID:                 q.ID,
		Kind:               q.Kind,
		Prompt:             q.Prompt,
		Instructions:       q.Instructions,
		Options:            kept,
		IncludeNoneOfAbove: q.IncludeNoneOfAbove,
		CatchAllKey:        q.CatchAllKey,
	}
	if q.OptionDescriptions != nil {
		keptDescs := make(map[string]string, len(kept))
		for _, opt := range kept {
			if desc, ok := q.OptionDescriptions[opt]; ok {
				keptDescs[opt] = desc
			}
		}
		pruned.OptionDescriptions = keptDescs
	}
	if q.PriorLogits != nil {
		keptPriors := make(map[string]float64, len(kept))
		for _, opt := range kept {
			if logitVal, ok := q.PriorLogits[opt]; ok {
				keptPriors[opt] = logitVal
			}
		}
		pruned.PriorLogits = keptPriors
	}
	if q.DomainCompetence != nil {
		profCopy := *q.DomainCompetence
		pruned.DomainCompetence = &profCopy
	}
	return pruned, nil
}

// IsUniformDistribution returns true when K >= 2 normalized option probabilities lie within
// DefaultUniformDistributionTolerance (1e-6) of 1/K.
func IsUniformDistribution(probs map[string]float64) bool {
	return IsUniformDistributionWithTolerance(probs, DefaultUniformDistributionTolerance)
}

// IsUniformDistributionWithTolerance returns true when K >= 2 normalized option probabilities
// lie within tolerance of 1/K.
func IsUniformDistributionWithTolerance(probs map[string]float64, tolerance float64) bool {
	if len(probs) < 2 || math.IsNaN(tolerance) || math.IsInf(tolerance, 0) || tolerance < 0 {
		return false
	}
	total := 0.0
	for _, val := range probs {
		if math.IsNaN(val) || math.IsInf(val, 0) || val < 0 {
			return false
		}
		total += val
	}
	if total <= 0 || math.IsNaN(total) || math.IsInf(total, 0) {
		return false
	}
	target := 1.0 / float64(len(probs))
	for _, val := range probs {
		if math.Abs((val/total)-target) > tolerance {
			return false
		}
	}
	return true
}

// CascadeConfig configures two-phase and speculative Stage-1 -> Stage-2 cascade orchestration.
type CascadeConfig struct {
	HesitationThreshold                 float64
	UseZeroHesitationThreshold          bool
	Temperature                         float64
	PriorWeight                         float64
	PriorGateMode                       string
	SpeculativeStage2                   bool
	Stage2TopK                          int
	DomainCompetence                    *DomainCompetenceProfile
	HasCustomNegativeTransferThresholds bool
	NegativeTransferMargin              float64
	NegativeTransferMinDeltaHurt        int
	NegativeTransferBrierRatio          float64
	UseDawidSkenePoEWeights             bool
	EnableVoIGate                       bool
	VoIErrorCost                        float64
	VoIStage2ExpectedAccuracy           float64
	VoIStage2RPCCostUSD                 float64
	VoILatencyCostUSD                   float64
}

// DefaultCascadeConfig returns canonical defaults for two-phase cascade orchestration.
func DefaultCascadeConfig() CascadeConfig {
	return CascadeConfig{
		HesitationThreshold:                 DefaultCascadeHesitationThreshold,
		Temperature:                         1.0,
		PriorGateMode:                       "hesitant",
		HasCustomNegativeTransferThresholds: true,
		NegativeTransferMargin:              0.05,
		NegativeTransferMinDeltaHurt:        -2,
		NegativeTransferBrierRatio:          2.5,
		VoIStage2ExpectedAccuracy:           0.95,
		VoIStage2RPCCostUSD:                 0.005,
	}
}

func (c CascadeConfig) effectiveNegativeTransferThresholds() (float64, int, float64) {
	if c.HasCustomNegativeTransferThresholds {
		return c.NegativeTransferMargin, c.NegativeTransferMinDeltaHurt, c.NegativeTransferBrierRatio
	}
	return 0.05, -2, 2.5
}

// Validate checks CascadeConfig bounds, including VoIStage2ExpectedAccuracy in [0.0, 1.0]
// and NegativeTransferMinDeltaHurt <= 0 when custom negative-transfer thresholds are enabled.
func (c CascadeConfig) Validate() error {
	if math.IsNaN(c.HesitationThreshold) || math.IsInf(c.HesitationThreshold, 0) || c.HesitationThreshold < 0 {
		return fmt.Errorf("mdm: HesitationThreshold must be finite and >= 0, got %v", c.HesitationThreshold)
	}
	if math.IsNaN(c.Temperature) || math.IsInf(c.Temperature, 0) || c.Temperature < 0 {
		return fmt.Errorf("mdm: Temperature must be finite and > 0, got %v", c.Temperature)
	}
	if math.IsNaN(c.PriorWeight) || math.IsInf(c.PriorWeight, 0) || c.PriorWeight < 0 {
		return fmt.Errorf("mdm: PriorWeight must be finite and >= 0, got %v", c.PriorWeight)
	}
	if c.Stage2TopK < 0 || c.Stage2TopK == 1 {
		return fmt.Errorf("mdm: Stage2TopK must be 0 or >= 2, got %d", c.Stage2TopK)
	}
	if c.DomainCompetence != nil {
		if err := c.DomainCompetence.Validate(); err != nil {
			return err
		}
	}
	if c.HasCustomNegativeTransferThresholds {
		if math.IsNaN(c.NegativeTransferMargin) || math.IsInf(c.NegativeTransferMargin, 0) || c.NegativeTransferMargin < 0 {
			return fmt.Errorf("mdm: NegativeTransferMargin must be finite and >= 0, got %v", c.NegativeTransferMargin)
		}
		if c.NegativeTransferMinDeltaHurt > 0 {
			return fmt.Errorf("mdm: NegativeTransferMinDeltaHurt must be <= 0, got %d", c.NegativeTransferMinDeltaHurt)
		}
		if math.IsNaN(c.NegativeTransferBrierRatio) || math.IsInf(c.NegativeTransferBrierRatio, 0) || c.NegativeTransferBrierRatio <= 1.0 {
			return fmt.Errorf("mdm: NegativeTransferBrierRatio must be finite and > 1.0, got %v", c.NegativeTransferBrierRatio)
		}
	}
	if math.IsNaN(c.VoIStage2ExpectedAccuracy) || math.IsInf(c.VoIStage2ExpectedAccuracy, 0) ||
		c.VoIStage2ExpectedAccuracy < 0.0 || c.VoIStage2ExpectedAccuracy > 1.0 {
		return fmt.Errorf("mdm: VoIStage2ExpectedAccuracy must lie in [0.0, 1.0], got %v", c.VoIStage2ExpectedAccuracy)
	}
	if math.IsNaN(c.VoIErrorCost) || math.IsInf(c.VoIErrorCost, 0) || c.VoIErrorCost < 0 {
		return fmt.Errorf("mdm: VoIErrorCost must be finite and >= 0, got %v", c.VoIErrorCost)
	}
	if math.IsNaN(c.VoIStage2RPCCostUSD) || math.IsInf(c.VoIStage2RPCCostUSD, 0) || c.VoIStage2RPCCostUSD < 0 {
		return fmt.Errorf("mdm: VoIStage2RPCCostUSD must be finite and >= 0, got %v", c.VoIStage2RPCCostUSD)
	}
	if math.IsNaN(c.VoILatencyCostUSD) || math.IsInf(c.VoILatencyCostUSD, 0) || c.VoILatencyCostUSD < 0 {
		return fmt.Errorf("mdm: VoILatencyCostUSD must be finite and >= 0, got %v", c.VoILatencyCostUSD)
	}
	return nil
}

// CascadeStageBackend scores a batch of ChoiceQuestions under context cancellation.
type CascadeStageBackend interface {
	ScoreQuestions(ctx context.Context, state any, questions []ChoiceQuestion) (map[string]map[string]float64, error)
}

// CascadeBackendFunc adapts an ordinary function into a CascadeStageBackend.
type CascadeBackendFunc func(ctx context.Context, state any, questions []ChoiceQuestion) (map[string]map[string]float64, error)

// ScoreQuestions invokes f(ctx, state, questions).
func (f CascadeBackendFunc) ScoreQuestions(ctx context.Context, state any, questions []ChoiceQuestion) (map[string]map[string]float64, error) {
	return f(ctx, state, questions)
}

// CascadeSlotResult holds the calibrated probabilities and escalation telemetry for one slot.
type CascadeSlotResult struct {
	SlotID           string
	Predicted        string
	Probabilities    map[string]float64
	RawProbabilities map[string]float64
	HRaw             float64
	Confidence       float64
	Escalated        bool
	PassesUsed       int
}

// CascadeResult holds the multi-slot cascade readout and speculative/circuit-breaker flags.
type CascadeResult struct {
	Slots                              map[string]CascadeSlotResult
	Escalated                          bool
	EscalationReasons                  []string
	SpeculativeLaunched                bool
	SpeculativeSkippedByCircuitBreaker bool
}

// CascadeOrchestrator coordinates two-phase and speculative Stage-1 -> Stage-2 cascade evaluation.
type CascadeOrchestrator struct {
	stage1 CascadeStageBackend
	stage2 CascadeStageBackend
	config CascadeConfig
}

// NewCascadeOrchestrator validates cfg and constructs a CascadeOrchestrator.
func NewCascadeOrchestrator(stage1, stage2 CascadeStageBackend, cfg CascadeConfig) (*CascadeOrchestrator, error) {
	if stage1 == nil {
		return nil, errors.New("mdm: stage1 backend must not be nil")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.UseDawidSkenePoEWeights && cfg.DomainCompetence == nil {
		return nil, errors.New("mdm: UseDawidSkenePoEWeights requires DomainCompetence to be set")
	}
	return &CascadeOrchestrator{
		stage1: stage1,
		stage2: stage2,
		config: cfg,
	}, nil
}

func (o *CascadeOrchestrator) resolveSlotCompetence(q ChoiceQuestion) *DomainCompetenceProfile {
	if q.DomainCompetence != nil {
		return q.DomainCompetence
	}
	return o.config.DomainCompetence
}

func (o *CascadeOrchestrator) isSlotNegativeTransfer(profile *DomainCompetenceProfile) bool {
	if profile == nil {
		return false
	}
	margin, minHurt, brierRatio := o.config.effectiveNegativeTransferThresholds()
	return profile.IsNegativeTransfer(margin, minHurt, brierRatio)
}

func validateStrictSlotDistribution(dist map[string]float64, options []string, slotID string) (map[string]float64, error) {
	if len(dist) == 0 {
		return nil, fmt.Errorf("mdm: missing probabilities for slot %q", slotID)
	}
	clean := make(map[string]float64, len(options))
	total := 0.0
	for _, opt := range options {
		val, ok := dist[opt]
		if !ok || math.IsNaN(val) || math.IsInf(val, 0) || val < 0 {
			return nil, fmt.Errorf("mdm: invalid probability for slot %q option %q: %v", slotID, opt, val)
		}
		clean[opt] = val
		total += val
	}
	if total <= 0 || math.IsNaN(total) || math.IsInf(total, 0) {
		return nil, fmt.Errorf("mdm: probabilities for slot %q sum to zero", slotID)
	}
	for _, opt := range options {
		clean[opt] /= total
	}
	return clean, nil
}

func argmaxByOptionOrder(probs map[string]float64, options []string) string {
	bestKey := options[0]
	bestProb := -1.0
	for _, opt := range options {
		if p := probs[opt]; p > bestProb {
			bestProb = p
			bestKey = opt
		}
	}
	return bestKey
}

// Evaluate executes two-phase or speculative cascade routing while honoring ctx cancellation
// without mutating caller contexts on fast-exit.
func (o *CascadeOrchestrator) Evaluate(
	ctx context.Context,
	state any,
	questions []ChoiceQuestion,
) (CascadeResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return CascadeResult{}, err
	}
	if len(questions) == 0 {
		return CascadeResult{}, errors.New("mdm: questions must be non-empty")
	}

	seenIDs := make(map[string]bool, len(questions))
	byID := make(map[string]ChoiceQuestion, len(questions))
	orderedIDs := make([]string, 0, len(questions))
	for idx, rawQ := range questions {
		q := cloneChoiceQuestion(rawQ)
		if strings.TrimSpace(q.ID) == "" {
			q.ID = fmt.Sprintf("q_%d", idx)
		}
		if seenIDs[q.ID] {
			return CascadeResult{}, fmt.Errorf("mdm: duplicate ChoiceQuestion.ID %q", q.ID)
		}
		if len(q.Options) < 2 {
			return CascadeResult{}, fmt.Errorf("mdm: slot %q requires >= 2 options", q.ID)
		}
		if q.DomainCompetence != nil {
			if err := q.DomainCompetence.Validate(); err != nil {
				return CascadeResult{}, err
			}
		}
		seenIDs[q.ID] = true
		byID[q.ID] = q
		orderedIDs = append(orderedIDs, q.ID)
	}

	effTemp := o.config.Temperature
	if effTemp <= 0 {
		effTemp = 1.0
	}
	effHes := o.config.HesitationThreshold
	if effHes == 0 && !o.config.UseZeroHesitationThreshold {
		effHes = DefaultCascadeHesitationThreshold
	}
	gateMode := o.config.PriorGateMode
	if strings.TrimSpace(gateMode) == "" {
		gateMode = "hesitant"
	}

	type specOutcome struct {
		dists map[string]map[string]float64
		err   error
	}

	stage2Ctx, cancelStage2 := context.WithCancel(ctx)
	defer cancelStage2()

	var specCh chan specOutcome
	specCovered := make(map[string]bool)
	specLaunched := false
	specSkippedByCB := false

	if o.config.SpeculativeStage2 && o.stage2 != nil {
		allChoiceNegTransfer := true
		specQuestions := make([]ChoiceQuestion, 0, len(orderedIDs))
		for _, qid := range orderedIDs {
			q := byID[qid]
			isChoice := q.effectiveKind() == QuestionKindChoice
			isBlocked := o.isSlotNegativeTransfer(o.resolveSlotCompetence(q))
			if !(isChoice && isBlocked) {
				allChoiceNegTransfer = false
				specQuestions = append(specQuestions, cloneChoiceQuestion(q))
				specCovered[qid] = true
			}
		}
		if allChoiceNegTransfer {
			specSkippedByCB = true
		} else if len(specQuestions) > 0 {
			specLaunched = true
			specCh = make(chan specOutcome, 1)
			go func(runQuestions []ChoiceQuestion) {
				dists, s2Err := o.stage2.ScoreQuestions(stage2Ctx, state, runQuestions)
				specCh <- specOutcome{dists: dists, err: s2Err}
			}(specQuestions)
		}
	}

	stage1Questions := make([]ChoiceQuestion, 0, len(orderedIDs))
	for _, qid := range orderedIDs {
		stage1Questions = append(stage1Questions, cloneChoiceQuestion(byID[qid]))
	}
	rawStage1, err := o.stage1.ScoreQuestions(ctx, state, stage1Questions)
	if err != nil {
		cancelStage2()
		if ctxErr := ctx.Err(); ctxErr != nil {
			return CascadeResult{}, ctxErr
		}
		return CascadeResult{}, err
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		cancelStage2()
		return CascadeResult{}, ctxErr
	}

	slotsOut := make(map[string]CascadeSlotResult, len(orderedIDs))
	candidateEscalated := make([]string, 0)
	escalationReasons := make([]string, 0)
	stage1UniformByID := make(map[string]bool, len(orderedIDs))

	for _, qid := range orderedIDs {
		q := byID[qid]
		normRaw, vErr := validateStrictSlotDistribution(rawStage1[qid], q.Options, qid)
		if vErr != nil {
			cancelStage2()
			return CascadeResult{}, vErr
		}
		s1Uniform := IsUniformDistribution(normRaw)
		stage1UniformByID[qid] = s1Uniform

		fusedProbs, winner, _, rawNormH, _ := FusePriorWithGate(
			normRaw,
			q.PriorLogits,
			q.Options,
			effTemp,
			o.config.PriorWeight,
			effHes,
			gateMode,
		)
		if s1Uniform && (len(q.PriorLogits) == 0 || o.config.PriorWeight <= 0) {
			// Preserve exact normalized uniform probabilities without 1e-6 floor drift.
			fusedProbs = make(map[string]float64, len(normRaw))
			for k, v := range normRaw {
				fusedProbs[k] = v
			}
			winner = argmaxByOptionOrder(fusedProbs, q.Options)
			rawNormH = 1.0
		}

		hesEsc := rawNormH >= effHes
		voiEsc := false
		if o.config.EnableVoIGate {
			maxP := fusedProbs[winner]
			riskStage1 := (1.0 - maxP) * o.config.VoIErrorCost
			riskStage2 := (1.0 - o.config.VoIStage2ExpectedAccuracy) * o.config.VoIErrorCost
			netVoI := riskStage1 - (riskStage2 + o.config.VoIStage2RPCCostUSD + o.config.VoILatencyCostUSD)
			voiEsc = netVoI > 0.0
		}
		uniformRescue := s1Uniform && o.stage2 != nil

		if hesEsc || voiEsc || uniformRescue {
			candidateEscalated = append(candidateEscalated, qid)
			if hesEsc {
				escalationReasons = append(escalationReasons, fmt.Sprintf("%s:h_raw=%.4f>=%.4f", qid, rawNormH, effHes))
			}
			if voiEsc {
				escalationReasons = append(escalationReasons, fmt.Sprintf("%s:voi_positive", qid))
			}
		}

		slotsOut[qid] = CascadeSlotResult{
			SlotID:           qid,
			Predicted:        winner,
			Probabilities:    fusedProbs,
			RawProbabilities: normRaw,
			HRaw:             rawNormH,
			Confidence:       1.0 - rawNormH,
			Escalated:        false,
			PassesUsed:       1,
		}
	}

	didEscalate := false
	if len(candidateEscalated) > 0 && o.stage2 != nil {
		executableQIDs := make([]string, 0, len(candidateEscalated))
		hadNegTransferBlock := false
		for _, qid := range candidateEscalated {
			q := byID[qid]
			blocked := q.effectiveKind() == QuestionKindChoice && o.isSlotNegativeTransfer(o.resolveSlotCompetence(q))
			if blocked && !stage1UniformByID[qid] {
				hadNegTransferBlock = true
			} else {
				if blocked && stage1UniformByID[qid] {
					escalationReasons = append(escalationReasons, fmt.Sprintf("%s:stage1_uniform_abstention_rescue", qid))
				}
				executableQIDs = append(executableQIDs, qid)
			}
		}
		if hadNegTransferBlock {
			escalationReasons = append(escalationReasons, "domain_competence:negative_transfer_short_circuit")
		}

		if len(executableQIDs) == 0 {
			cancelStage2()
		} else {
			if ctxErr := ctx.Err(); ctxErr != nil {
				cancelStage2()
				return CascadeResult{}, ctxErr
			}
			didEscalate = true
			stage2RawByID := make(map[string]map[string]float64, len(executableQIDs))
			evaluatedOptsByID := make(map[string][]string, len(executableQIDs))

			needSpec := false
			for _, qid := range executableQIDs {
				if specCovered[qid] {
					needSpec = true
					break
				}
			}
			if specCh != nil && needSpec {
				select {
				case outcome := <-specCh:
					if ctxErr := ctx.Err(); ctxErr != nil {
						return CascadeResult{}, ctxErr
					}
					if outcome.err != nil {
						return CascadeResult{}, outcome.err
					}
					for _, qid := range executableQIDs {
						if specCovered[qid] {
							if d, ok := outcome.dists[qid]; ok {
								stage2RawByID[qid] = d
								evaluatedOptsByID[qid] = byID[qid].Options
							}
						}
					}
				case <-ctx.Done():
					cancelStage2()
					return CascadeResult{}, ctx.Err()
				}
			} else {
				cancelStage2()
			}

			twoPhaseQuestions := make([]ChoiceQuestion, 0)
			for _, qid := range executableQIDs {
				if _, already := stage2RawByID[qid]; already {
					continue
				}
				baseQ := byID[qid]
				if o.config.Stage2TopK >= 2 && baseQ.effectiveKind() == QuestionKindChoice && len(baseQ.Options) > o.config.Stage2TopK {
					prunedQ, pErr := PruneChoiceTopK(baseQ, slotsOut[qid].Probabilities, o.config.Stage2TopK)
					if pErr != nil {
						return CascadeResult{}, pErr
					}
					twoPhaseQuestions = append(twoPhaseQuestions, prunedQ)
					evaluatedOptsByID[qid] = prunedQ.Options
				} else {
					twoPhaseQuestions = append(twoPhaseQuestions, cloneChoiceQuestion(baseQ))
					evaluatedOptsByID[qid] = baseQ.Options
				}
			}

			if len(twoPhaseQuestions) > 0 {
				if ctxErr := ctx.Err(); ctxErr != nil {
					return CascadeResult{}, ctxErr
				}
				twoPhaseDists, tpErr := o.stage2.ScoreQuestions(ctx, state, twoPhaseQuestions)
				if tpErr != nil {
					if ctxErr := ctx.Err(); ctxErr != nil {
						return CascadeResult{}, ctxErr
					}
					return CascadeResult{}, tpErr
				}
				if ctxErr := ctx.Err(); ctxErr != nil {
					return CascadeResult{}, ctxErr
				}
				for _, tq := range twoPhaseQuestions {
					stage2RawByID[tq.ID] = twoPhaseDists[tq.ID]
				}
			}

			for _, qid := range executableQIDs {
				q := byID[qid]
				evalOpts := evaluatedOptsByID[qid]
				normSubS2, vErr := validateStrictSlotDistribution(stage2RawByID[qid], evalOpts, qid)
				if vErr != nil {
					return CascadeResult{}, vErr
				}
				s2Uniform := IsUniformDistribution(normSubS2)
				expandedS2 := make(map[string]float64, len(q.Options))
				for _, opt := range q.Options {
					expandedS2[opt] = normSubS2[opt]
				}

				s1Uniform := stage1UniformByID[qid]
				profile := o.resolveSlotCompetence(q)
				var finalProbs map[string]float64

				switch {
				case s1Uniform && !s2Uniform:
					// Weight (0.0, 1.0): rescue uniform-blocked Stage 1 using non-uniform Stage 2.
					finalProbs = expandedS2
				case !s1Uniform && s2Uniform:
					// Weight (1.0, 0.0): preserve calibrated Stage 1 when Stage 2 returns uniform fallback.
					finalProbs = make(map[string]float64, len(slotsOut[qid].Probabilities))
					for k, v := range slotsOut[qid].Probabilities {
						finalProbs[k] = v
					}
					escalationReasons = append(escalationReasons, fmt.Sprintf("%s:stage2_uniform_fallback_preserved_stage1", qid))
				case s1Uniform && s2Uniform:
					finalProbs = expandedS2
				case o.config.UseDawidSkenePoEWeights && profile != nil:
					numOpts := len(q.Options)
					if profile.NumOptions > numOpts {
						numOpts = profile.NumOptions
					}
					w1, w2, wErr := ComputeDawidSkenePoEWeights(
						profile.Stage1Accuracy,
						profile.Stage2Accuracy,
						numOpts,
						DefaultDawidSkeneFloorAccuracy,
					)
					if wErr != nil {
						return CascadeResult{}, wErr
					}
					wSum := w1 + w2
					alpha1 := w1 / wSum
					alpha2 := w2 / wSum
					const epsPoE = 1e-12
					s1Probs := slotsOut[qid].Probabilities
					logits := make(map[string]float64, len(q.Options))
					maxL := math.Inf(-1)
					for _, opt := range q.Options {
						p1 := math.Max(epsPoE, s1Probs[opt])
						p2 := math.Max(epsPoE, expandedS2[opt])
						lv := alpha1*math.Log(p1) + alpha2*math.Log(p2)
						logits[opt] = lv
						if lv > maxL {
							maxL = lv
						}
					}
					finalProbs = make(map[string]float64, len(q.Options))
					denom := 0.0
					for _, opt := range q.Options {
						ev := math.Exp(logits[opt] - maxL)
						finalProbs[opt] = ev
						denom += ev
					}
					for _, opt := range q.Options {
						finalProbs[opt] /= denom
					}
				default:
					finalProbs = expandedS2
				}

				pred := argmaxByOptionOrder(finalProbs, q.Options)
				hS2 := 0.0
				for _, opt := range q.Options {
					p := finalProbs[opt]
					if p > 0 {
						hS2 -= p * math.Log(p)
					}
				}
				normHS2 := 0.0
				if len(q.Options) > 1 {
					normHS2 = math.Min(1.0, math.Max(0.0, hS2/math.Log(float64(len(q.Options)))))
				}

				slotsOut[qid] = CascadeSlotResult{
					SlotID:           qid,
					Predicted:        pred,
					Probabilities:    finalProbs,
					RawProbabilities: expandedS2,
					HRaw:             normHS2,
					Confidence:       1.0 - normHS2,
					Escalated:        true,
					PassesUsed:       2,
				}
			}
		}
	} else {
		cancelStage2()
	}

	return CascadeResult{
		Slots:                              slotsOut,
		Escalated:                          didEscalate,
		EscalationReasons:                  escalationReasons,
		SpeculativeLaunched:                specLaunched,
		SpeculativeSkippedByCircuitBreaker: specSkippedByCB,
	}, nil
}
