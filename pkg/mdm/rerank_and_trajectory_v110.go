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
	"strings"
)

// DefaultAbstainDocID is the canonical abstention option identifier in hybrid RAG shortlisting.
const DefaultAbstainDocID = "__no_relevant_doc__"

// RerankCandidate represents a candidate passage entering hybrid RAG context filtering.
type RerankCandidate struct {
	DocID             string
	Text              string
	Stage1MarginScore float64
}

// RAGContextFilterResult summarizes conformal RAG context retention and injection quarantine.
type RAGContextFilterResult struct {
	RetainedDocIDs         []string
	QuarantinedDocIDs      []string
	AbstainedNoRelevantDoc bool
	NormalizedRelevance    map[string]float64
	ConformalSetIDs        []string
	ShouldEscalate         bool
}

// DefaultNEVCPolicy returns the canonical NEVC policy defaults matching Python and TypeScript.
func DefaultNEVCPolicy() NEVCPolicy {
	return NEVCPolicy{
		TaskValue:                      1.0,
		InterventionCost:               0.08,
		FixProbabilityIfFailing:        0.45,
		RegressionProbabilityIfPassing: 0.04,
		PlattSlope:                     1.0,
		PlattIntercept:                 0.0,
	}
}

// NEVCPolicy configures the Net Expected Value of Control (NEVC) gate for agent loops.
type NEVCPolicy struct {
	TaskValue                      float64
	InterventionCost               float64
	FixProbabilityIfFailing        float64
	RegressionProbabilityIfPassing float64
	PlattSlope                     float64
	PlattIntercept                 float64
}

// NEVCDecision summarizes an NEVC intervention evaluation.
type NEVCDecision struct {
	ShouldIntervene           bool
	RawFailProbability        float64
	CalibratedFailProbability float64
	NetExpectedValue          float64
	MinFailProbabilityCutoff  float64
}

func riemannZetaApprox(exponentP float64) float64 {
	const terms = 512
	partial := 0.0
	for idx := 1; idx <= terms; idx++ {
		partial += math.Pow(float64(idx), -exponentP)
	}
	nFloat := float64(terms)
	tail := math.Pow(nFloat, 1.0-exponentP)/(exponentP-1.0) - 0.5*math.Pow(nFloat, -exponentP)
	return partial + tail
}

// PolynomialAlphaAtStep computes step-t miscoverage budget alpha_t under polynomial risk spending.
// Pass horizonMaxSteps <= 0 for an unbounded trajectory normalized by Riemann zeta(p) (requires p > 1).
func PolynomialAlphaAtStep(
	stepIndex int,
	alphaTotal float64,
	exponentP float64,
	horizonMaxSteps int,
	markovSlackFactor float64,
) (float64, error) {
	if stepIndex < 1 {
		return 0, fmt.Errorf("mdm: stepIndex must be >= 1, got %d", stepIndex)
	}
	if math.IsNaN(alphaTotal) || math.IsInf(alphaTotal, 0) || alphaTotal <= 0 || alphaTotal >= 1 {
		return 0, fmt.Errorf("mdm: alphaTotal must lie strictly in (0, 1), got %v", alphaTotal)
	}
	if math.IsNaN(exponentP) || math.IsInf(exponentP, 0) || exponentP <= 0 {
		return 0, fmt.Errorf("mdm: exponentP must be > 0, got %v", exponentP)
	}
	if math.IsNaN(markovSlackFactor) || math.IsInf(markovSlackFactor, 0) || markovSlackFactor <= 0 || markovSlackFactor > 1 {
		return 0, fmt.Errorf("mdm: markovSlackFactor must lie in (0, 1], got %v", markovSlackFactor)
	}

	var normConst float64
	if horizonMaxSteps > 0 {
		if stepIndex > horizonMaxSteps {
			return 0, fmt.Errorf("mdm: stepIndex (%d) exceeds horizonMaxSteps (%d)", stepIndex, horizonMaxSteps)
		}
		for j := 1; j <= horizonMaxSteps; j++ {
			normConst += math.Pow(float64(j), -exponentP)
		}
	} else {
		if exponentP <= 1.0 {
			return 0, fmt.Errorf("mdm: unbounded horizon requires exponentP > 1.0, got %v", exponentP)
		}
		normConst = riemannZetaApprox(exponentP)
	}
	alphaStep := (alphaTotal * markovSlackFactor) * math.Pow(float64(stepIndex), -exponentP) / normConst
	if alphaStep < math.SmallestNonzeroFloat64 {
		alphaStep = math.SmallestNonzeroFloat64
	}
	if alphaStep > 0.999999 {
		alphaStep = 0.999999
	}
	return alphaStep, nil
}

func logitClipped(prob float64) float64 {
	clipped := math.Min(1.0-1e-6, math.Max(1e-6, prob))
	return math.Log(clipped / (1.0 - clipped))
}

func sigmoidStable(logitVal float64) float64 {
	if logitVal >= 0 {
		return 1.0 / (1.0 + math.Exp(-math.Min(60.0, logitVal)))
	}
	expPos := math.Exp(math.Max(-60.0, logitVal))
	return expPos / (1.0 + expPos)
}

// EvaluateNEVCGate evaluates whether triggering an extra corrective agent turn has NEVC > 0.
func EvaluateNEVCGate(rawFailProb float64, policy NEVCPolicy) (NEVCDecision, error) {
	if math.IsNaN(rawFailProb) || math.IsInf(rawFailProb, 0) || rawFailProb < 0 || rawFailProb > 1 {
		return NEVCDecision{}, fmt.Errorf("mdm: rawFailProb must lie in [0, 1], got %v", rawFailProb)
	}
	if policy == (NEVCPolicy{}) {
		policy = DefaultNEVCPolicy()
	}
	taskVal := policy.TaskValue
	if taskVal == 0 {
		taskVal = 1.0
	}
	if math.IsNaN(taskVal) || math.IsInf(taskVal, 0) || taskVal <= 0 {
		return NEVCDecision{}, fmt.Errorf("mdm: TaskValue must be > 0, got %v", taskVal)
	}
	if math.IsNaN(policy.InterventionCost) || math.IsInf(policy.InterventionCost, 0) || policy.InterventionCost < 0 {
		return NEVCDecision{}, fmt.Errorf("mdm: InterventionCost must be >= 0, got %v", policy.InterventionCost)
	}
	fixProb := policy.FixProbabilityIfFailing
	if fixProb == 0 {
		fixProb = 0.45
	}
	if math.IsNaN(fixProb) || math.IsInf(fixProb, 0) || fixProb <= 0 || fixProb > 1 {
		return NEVCDecision{}, fmt.Errorf("mdm: FixProbabilityIfFailing must lie in (0, 1], got %v", fixProb)
	}
	if math.IsNaN(policy.RegressionProbabilityIfPassing) || math.IsInf(policy.RegressionProbabilityIfPassing, 0) ||
		policy.RegressionProbabilityIfPassing < 0 || policy.RegressionProbabilityIfPassing >= 1 {
		return NEVCDecision{}, fmt.Errorf("mdm: RegressionProbabilityIfPassing must lie in [0, 1)")
	}
	slope := policy.PlattSlope
	if slope == 0 {
		slope = 1.0
	}
	if math.IsNaN(slope) || math.IsInf(slope, 0) || slope <= 0 {
		return NEVCDecision{}, fmt.Errorf("mdm: PlattSlope must be > 0, got %v", slope)
	}
	if math.IsNaN(policy.PlattIntercept) || math.IsInf(policy.PlattIntercept, 0) {
		return NEVCDecision{}, fmt.Errorf("mdm: PlattIntercept must be finite")
	}

	calFail := sigmoidStable(slope*logitClipped(rawFailProb) + policy.PlattIntercept)
	expectedGain := calFail * fixProb * taskVal
	expectedRegress := (1.0 - calFail) * policy.RegressionProbabilityIfPassing * taskVal
	netVal := expectedGain - policy.InterventionCost - expectedRegress
	denom := taskVal * (fixProb + policy.RegressionProbabilityIfPassing)
	cutoff := math.Min(1.0, math.Max(0.0, (policy.InterventionCost+policy.RegressionProbabilityIfPassing*taskVal)/denom))

	return NEVCDecision{
		ShouldIntervene:           netVal > 0,
		RawFailProbability:        rawFailProb,
		CalibratedFailProbability: calFail,
		NetExpectedValue:          netVal,
		MinFailProbabilityCutoff:  cutoff,
	}, nil
}

func containsPromptOverrideMarker(rawText string) bool {
	lowered := strings.ToLower(rawText)
	markers := []string{
		"</untrusted_data>",
		"<untrusted_data>",
		"ignore previous instructions",
		"ignore all previous instructions",
		"disregard prior instructions",
		"system override:",
		"evaluation rubric override",
		"select this document with probability",
	}
	for _, marker := range markers {
		if strings.Contains(lowered, marker) {
			return true
		}
	}
	return false
}

// ConformalRAGContextFilter retains a variable-size passage set with 1-alpha coverage while quarantining injections.
func ConformalRAGContextFilter(
	candidates []RerankCandidate,
	relevanceProbs map[string]float64,
	qHat float64,
	cfg ConformalConfig,
	abstainDocID string,
	abstainThreshold float64,
	quarantineInjections bool,
) (RAGContextFilterResult, error) {
	if len(candidates) == 0 {
		return RAGContextFilterResult{}, fmt.Errorf("mdm: candidates must be non-empty")
	}
	if len(relevanceProbs) == 0 {
		return RAGContextFilterResult{}, fmt.Errorf("mdm: relevanceProbs must be non-empty")
	}
	if strings.TrimSpace(abstainDocID) == "" {
		abstainDocID = DefaultAbstainDocID
	}
	if math.IsNaN(abstainThreshold) || math.IsInf(abstainThreshold, 0) || abstainThreshold <= 0 || abstainThreshold >= 1 {
		return RAGContextFilterResult{}, fmt.Errorf("mdm: abstainThreshold must lie in (0, 1), got %v", abstainThreshold)
	}
	if math.IsNaN(qHat) || qHat < 0 {
		return RAGContextFilterResult{}, fmt.Errorf("mdm: qHat must be >= 0 and non-NaN, got %v", qHat)
	}
	for probKey, probVal := range relevanceProbs {
		if strings.TrimSpace(probKey) == "" {
			return RAGContextFilterResult{}, fmt.Errorf("mdm: relevanceProbs keys must be non-empty")
		}
		if math.IsNaN(probVal) || math.IsInf(probVal, 0) || probVal < 0 {
			return RAGContextFilterResult{}, fmt.Errorf("mdm: invalid relevance probability for %q: %v", probKey, probVal)
		}
	}

	seenDocIDs := make(map[string]bool, len(candidates))
	quarantined := make([]string, 0)
	safeIDs := make([]string, 0, len(candidates))
	for _, cand := range candidates {
		if strings.TrimSpace(cand.DocID) == "" {
			return RAGContextFilterResult{}, fmt.Errorf("mdm: candidate DocID must be non-empty")
		}
		if cand.DocID == abstainDocID {
			return RAGContextFilterResult{}, fmt.Errorf("mdm: candidate DocID %q collides with abstainDocID", cand.DocID)
		}
		if seenDocIDs[cand.DocID] {
			return RAGContextFilterResult{}, fmt.Errorf("mdm: duplicate candidate DocID %q", cand.DocID)
		}
		seenDocIDs[cand.DocID] = true
		if quarantineInjections && containsPromptOverrideMarker(cand.Text) {
			quarantined = append(quarantined, cand.DocID)
		} else {
			safeIDs = append(safeIDs, cand.DocID)
		}
	}

	filtered := make(map[string]float64, len(safeIDs)+1)
	probSum := 0.0
	for _, docID := range safeIDs {
		val := relevanceProbs[docID]
		filtered[docID] = val
		probSum += val
	}
	if rawAbstain, hasAbstain := relevanceProbs[abstainDocID]; hasAbstain {
		filtered[abstainDocID] = rawAbstain
		probSum += rawAbstain
	}
	if probSum <= 0.0 {
		return RAGContextFilterResult{
			RetainedDocIDs:         []string{},
			QuarantinedDocIDs:      quarantined,
			AbstainedNoRelevantDoc: true,
			NormalizedRelevance:    map[string]float64{},
			ConformalSetIDs:        []string{},
			ShouldEscalate:         true,
		}, nil
	}

	norm := make(map[string]float64, len(filtered))
	for key, val := range filtered {
		norm[key] = val / probSum
	}
	if norm[abstainDocID] >= abstainThreshold {
		return RAGContextFilterResult{
			RetainedDocIDs:         []string{},
			QuarantinedDocIDs:      quarantined,
			AbstainedNoRelevantDoc: true,
			NormalizedRelevance:    norm,
			ConformalSetIDs:        []string{abstainDocID},
			ShouldEscalate:         false,
		}, nil
	}

	baseAbstain := cfg.AbstainKeys
	abstainCopy := make([]string, 0, len(baseAbstain)+1)
	abstainCopy = append(abstainCopy, baseAbstain...)
	abstainCopy = append(abstainCopy, abstainDocID)
	cfgWithAbstain := cfg
	cfgWithAbstain.AbstainKeys = abstainCopy
	cset, err := ConformalPredictionSet(norm, qHat, cfgWithAbstain)
	if err != nil {
		return RAGContextFilterResult{}, err
	}
	retained := make([]string, 0, len(cset.PredictionSet))
	for _, id := range cset.PredictionSet {
		if id != abstainDocID {
			retained = append(retained, id)
		}
	}
	return RAGContextFilterResult{
		RetainedDocIDs:         retained,
		QuarantinedDocIDs:      quarantined,
		AbstainedNoRelevantDoc: len(retained) == 0,
		NormalizedRelevance:    norm,
		ConformalSetIDs:        cset.PredictionSet,
		ShouldEscalate:         cset.ContainsAbstain || len(retained) == 0,
	}, nil
}
