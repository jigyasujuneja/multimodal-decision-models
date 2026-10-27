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
	"sort"
	"strconv"
	"strings"
)

// OnlineBCACIResult summarizes an Online Bias-Corrected Adaptive Conformal Inference (BC-ACI) stream.
type OnlineBCACIResult struct {
	EmpiricalCoverage          float64
	RollingWindowWorstCoverage float64
	MeanSetSize                float64
	SingletonRate              float64
	EscalationRate             float64
	FinalAlphaT                float64
	AlphaTrajectory            []float64
	QHatTrajectory             []float64
	CoveredFlags               []bool
}

// OnlineBCACITracker implements stateful Online Bias-Corrected Adaptive Conformal Inference (BC-ACI).
type OnlineBCACITracker struct {
	AlphaTarget  float64
	Gamma        float64
	Beta         float64
	EtaD         float64
	WindowSize   int
	Config       ConformalConfig
	AlphaMin     float64
	AlphaMax     float64
	AlphaT       float64
	BiasMomentum float64
	scoreWindow  []float64
}

// NewOnlineBCACITracker constructs an OnlineBCACITracker with validated parameters.
func NewOnlineBCACITracker(
	alphaTarget float64,
	gamma float64,
	beta float64,
	etaD float64,
	windowSize int,
	cfg ConformalConfig,
	initialScores []float64,
) (*OnlineBCACITracker, error) {
	if math.IsNaN(alphaTarget) || math.IsInf(alphaTarget, 0) || alphaTarget <= 0 || alphaTarget >= 1 {
		return nil, fmt.Errorf("mdm: alphaTarget must lie strictly in (0, 1), got %v", alphaTarget)
	}
	if math.IsNaN(gamma) || math.IsInf(gamma, 0) || gamma <= 0 {
		return nil, fmt.Errorf("mdm: gamma must be > 0, got %v", gamma)
	}
	if math.IsNaN(beta) || math.IsInf(beta, 0) || beta < 0 {
		return nil, fmt.Errorf("mdm: beta must be >= 0, got %v", beta)
	}
	if math.IsNaN(etaD) || math.IsInf(etaD, 0) || etaD <= 0 || etaD > 1 {
		return nil, fmt.Errorf("mdm: etaD must lie in (0, 1], got %v", etaD)
	}
	if windowSize < 5 {
		return nil, fmt.Errorf("mdm: windowSize must be >= 5, got %d", windowSize)
	}
	resolved, err := resolveConformalConfig(cfg)
	if err != nil {
		return nil, err
	}
	win := make([]float64, 0, windowSize)
	for _, s := range initialScores {
		if math.IsNaN(s) || math.IsInf(s, 0) || s < 0 {
			return nil, fmt.Errorf("mdm: invalid initial score %v", s)
		}
		win = append(win, s)
	}
	if len(win) > windowSize {
		win = append([]float64(nil), win[len(win)-windowSize:]...)
	}
	return &OnlineBCACITracker{
		AlphaTarget: alphaTarget,
		Gamma:       gamma,
		Beta:        beta,
		EtaD:        etaD,
		WindowSize:  windowSize,
		Config:      resolved,
		AlphaMin:    0.001,
		AlphaMax:    0.999,
		AlphaT:      alphaTarget,
		scoreWindow: win,
	}, nil
}

// CurrentQuantile returns the current empirical quantile q_hat_t from the sliding window.
func (t *OnlineBCACITracker) CurrentQuantile() float64 {
	if len(t.scoreWindow) == 0 {
		return 1.0 - t.AlphaT
	}
	effAlpha := math.Min(t.AlphaMax, math.Max(t.AlphaMin, t.AlphaT))
	if effAlpha <= t.AlphaMin+1e-9 {
		return math.Inf(1)
	}
	n := len(t.scoreWindow)
	sorted := make([]float64, n)
	copy(sorted, t.scoreWindow)
	sort.Float64s(sorted)
	orderIdx := int(math.Ceil(float64(n+1)*(1.0-effAlpha) - 1e-12))
	if orderIdx > n {
		return math.Inf(1)
	}
	pos := orderIdx - 1
	if pos < 0 {
		pos = 0
	}
	qEmp := sorted[pos]
	if effAlpha < t.AlphaTarget {
		urgency := math.Min(1.0, (t.AlphaTarget-effAlpha)/math.Max(1e-6, t.AlphaTarget))
		qFloor := (1.0 - effAlpha) * urgency
		if qFloor > qEmp {
			return qFloor
		}
	}
	return qEmp
}

// ObserveAndUpdate predicts C_{alpha_t}(X_t), checks goldLabel coverage, and updates alpha_{t+1}.
func (t *OnlineBCACITracker) ObserveAndUpdate(
	probs map[string]float64,
	goldLabel string,
) (ConformalSet, bool, float64, error) {
	qHatT := t.CurrentQuantile()
	setRes, err := ConformalPredictionSet(probs, qHatT, t.Config)
	if err != nil {
		return ConformalSet{}, false, t.AlphaT, err
	}
	ranked, norm, nErr := normalizeAndRankConformalProbs(probs)
	if nErr != nil {
		return ConformalSet{}, false, t.AlphaT, nErr
	}
	if _, ok := norm[goldLabel]; !ok {
		return ConformalSet{}, false, t.AlphaT, fmt.Errorf("mdm: goldLabel %q not in probabilities", goldLabel)
	}
	groups := partitionConformalTieGroups(ranked, norm)
	scores := computeDeterministicConformalScores(groups, norm, t.Config)
	scoreT := scores[goldLabel]

	covered := false
	for _, k := range setRes.PredictionSet {
		if k == goldLabel {
			covered = true
			break
		}
	}
	errT := 1.0
	if covered {
		errT = 0.0
	}
	t.BiasMomentum = (1.0-t.EtaD)*t.BiasMomentum + t.EtaD*(errT-t.AlphaTarget)
	nextAlpha := t.AlphaT + t.Gamma*(t.AlphaTarget-errT) - t.Beta*t.BiasMomentum
	maxCalmAlpha := math.Min(t.AlphaMax, math.Max(t.AlphaTarget*1.5, t.AlphaTarget+2.0*t.Gamma))
	t.AlphaT = math.Min(maxCalmAlpha, math.Max(t.AlphaMin, nextAlpha))

	t.scoreWindow = append(t.scoreWindow, scoreT)
	if len(t.scoreWindow) > t.WindowSize {
		t.scoreWindow = t.scoreWindow[1:]
	}
	return setRes, covered, t.AlphaT, nil
}

// EvaluateOnlineBCACIStream evaluates Online BC-ACI across a non-stationary probability stream.
func EvaluateOnlineBCACIStream(
	streamProbs []map[string]float64,
	streamLabels []string,
	alphaTarget float64,
	gamma float64,
	beta float64,
	etaD float64,
	windowSize int,
	rollingEvalWindow int,
	cfg ConformalConfig,
) (OnlineBCACIResult, error) {
	if len(streamProbs) == 0 || len(streamProbs) != len(streamLabels) {
		return OnlineBCACIResult{}, errors.New("mdm: streamProbs and streamLabels must be non-empty slices of equal length")
	}
	if rollingEvalWindow < 1 {
		return OnlineBCACIResult{}, fmt.Errorf("mdm: rollingEvalWindow must be >= 1, got %d", rollingEvalWindow)
	}
	tracker, err := NewOnlineBCACITracker(alphaTarget, gamma, beta, etaD, windowSize, cfg, nil)
	if err != nil {
		return OnlineBCACIResult{}, err
	}

	n := len(streamProbs)
	alphaTraj := make([]float64, 0, n)
	qHatTraj := make([]float64, 0, n)
	coveredFlags := make([]bool, 0, n)
	totalSize := 0
	singletonCount := 0
	escalateCount := 0
	coveredCount := 0

	for i := 0; i < n; i++ {
		alphaTraj = append(alphaTraj, tracker.AlphaT)
		setRes, covered, _, stepErr := tracker.ObserveAndUpdate(streamProbs[i], streamLabels[i])
		if stepErr != nil {
			return OnlineBCACIResult{}, stepErr
		}
		qHatTraj = append(qHatTraj, setRes.QHat)
		coveredFlags = append(coveredFlags, covered)
		totalSize += len(setRes.PredictionSet)
		if setRes.Singleton {
			singletonCount++
		}
		if setRes.ShouldEscalate {
			escalateCount++
		}
		if covered {
			coveredCount++
		}
	}

	empCov := float64(coveredCount) / float64(n)
	worstRolling := empCov
	if n > rollingEvalWindow {
		worstRolling = 1.0
		w := rollingEvalWindow
		runSum := 0
		for i := 0; i < w; i++ {
			if coveredFlags[i] {
				runSum++
			}
		}
		worstRolling = math.Min(worstRolling, float64(runSum)/float64(w))
		for i := w; i < n; i++ {
			if coveredFlags[i] {
				runSum++
			}
			if coveredFlags[i-w] {
				runSum--
			}
			worstRolling = math.Min(worstRolling, float64(runSum)/float64(w))
		}
	}

	return OnlineBCACIResult{
		EmpiricalCoverage:          empCov,
		RollingWindowWorstCoverage: worstRolling,
		MeanSetSize:                float64(totalSize) / float64(n),
		SingletonRate:              float64(singletonCount) / float64(n),
		EscalationRate:             float64(escalateCount) / float64(n),
		FinalAlphaT:                tracker.AlphaT,
		AlphaTrajectory:            alphaTraj,
		QHatTrajectory:             qHatTraj,
		CoveredFlags:               coveredFlags,
	}, nil
}

// ModelArmSpec specifies the per-call cost and latency of a candidate decision or reasoning model arm.
type ModelArmSpec struct {
	ArmID     string
	CostUSD   float64
	LatencyMs float64
}

// BanditRoutingDecision holds the selected arm and acquisition diagnostics for a single step.
type BanditRoutingDecision struct {
	ChosenArm             string
	UCBQualityScore       float64
	PrimalDualAcquisition float64
	DualPriceLambda       float64
	DualLatencyMu         float64
	ArmUCBScores          map[string]float64
}

// PrimalDualParetoBanditRouter implements Online Primal-Dual Contextual & Dueling Bandit Routing.
type PrimalDualParetoBanditRouter struct {
	ContextDim        int
	BudgetPerQueryUSD float64
	AlphaUCB          float64
	RhoDiscount       float64
	RidgeLambda       float64
	EtaDualCost       float64
	DuelingLR         float64
	DuelingWeight     float64
	DualLambda        float64
	DualMu            float64
	Arms              map[string]ModelArmSpec
	ArmOrder          []string
	PullCounts        map[string]int
	covA              map[string][][]float64
	vecB              map[string][]float64
	duelW             map[string][]float64
}

func solveSPDLinearSystemGo(matrix [][]float64, rhs []float64) []float64 {
	d := len(rhs)
	aug := make([][]float64, d)
	for i := 0; i < d; i++ {
		row := make([]float64, d+1)
		for j := 0; j < d; j++ {
			row[j] = matrix[i][j]
		}
		row[d] = rhs[i]
		aug[i] = row
	}
	for col := 0; col < d; col++ {
		pivotRow := col
		maxVal := math.Abs(aug[col][col])
		for r := col + 1; r < d; r++ {
			if math.Abs(aug[r][col]) > maxVal {
				maxVal = math.Abs(aug[r][col])
				pivotRow = r
			}
		}
		if maxVal < 1e-12 {
			aug[col][col] += 1e-6
			pivotRow = col
		}
		if pivotRow != col {
			aug[col], aug[pivotRow] = aug[pivotRow], aug[col]
		}
		piv := aug[col][col]
		for j := col; j <= d; j++ {
			aug[col][j] /= piv
		}
		for r := 0; r < d; r++ {
			if r == col {
				continue
			}
			factor := aug[r][col]
			if math.Abs(factor) > 1e-15 {
				for j := col; j <= d; j++ {
					aug[r][j] -= factor * aug[col][j]
				}
			}
		}
	}
	sol := make([]float64, d)
	for i := 0; i < d; i++ {
		sol[i] = aug[i][d]
	}
	return sol
}

// NewPrimalDualParetoBanditRouter constructs a PrimalDualParetoBanditRouter.
func NewPrimalDualParetoBanditRouter(
	arms []ModelArmSpec,
	contextDim int,
	budgetPerQueryUSD float64,
	alphaUCB float64,
	rhoDiscount float64,
	ridgeLambda float64,
	etaDualCost float64,
) (*PrimalDualParetoBanditRouter, error) {
	if len(arms) == 0 {
		return nil, errors.New("mdm: arms must be non-empty")
	}
	if contextDim < 1 {
		return nil, fmt.Errorf("mdm: contextDim must be >= 1, got %d", contextDim)
	}
	if budgetPerQueryUSD <= 0 || math.IsNaN(budgetPerQueryUSD) {
		return nil, fmt.Errorf("mdm: budgetPerQueryUSD must be > 0, got %v", budgetPerQueryUSD)
	}
	if rhoDiscount <= 0 || rhoDiscount > 1 || math.IsNaN(rhoDiscount) {
		return nil, fmt.Errorf("mdm: rhoDiscount must lie in (0, 1], got %v", rhoDiscount)
	}
	if ridgeLambda <= 0 || math.IsNaN(ridgeLambda) {
		return nil, fmt.Errorf("mdm: ridgeLambda must be > 0, got %v", ridgeLambda)
	}

	armMap := make(map[string]ModelArmSpec, len(arms))
	order := make([]string, 0, len(arms))
	for _, a := range arms {
		if strings.TrimSpace(a.ArmID) == "" {
			return nil, errors.New("mdm: ArmID must be non-empty")
		}
		armMap[a.ArmID] = a
		order = append(order, a.ArmID)
	}
	sort.Strings(order)

	covA := make(map[string][][]float64, len(order))
	vecB := make(map[string][]float64, len(order))
	duelW := make(map[string][]float64, len(order))
	pulls := make(map[string]int, len(order))
	for _, id := range order {
		mat := make([][]float64, contextDim)
		for i := 0; i < contextDim; i++ {
			mat[i] = make([]float64, contextDim)
			mat[i][i] = ridgeLambda
		}
		covA[id] = mat
		vecB[id] = make([]float64, contextDim)
		duelW[id] = make([]float64, contextDim)
		pulls[id] = 0
	}

	return &PrimalDualParetoBanditRouter{
		ContextDim:        contextDim,
		BudgetPerQueryUSD: budgetPerQueryUSD,
		AlphaUCB:          alphaUCB,
		RhoDiscount:       rhoDiscount,
		RidgeLambda:       ridgeLambda,
		EtaDualCost:       etaDualCost,
		DuelingLR:         0.15,
		DuelingWeight:     0.20,
		Arms:              armMap,
		ArmOrder:          order,
		PullCounts:        pulls,
		covA:              covA,
		vecB:              vecB,
		duelW:             duelW,
	}, nil
}

// SelectArm chooses the Pareto-optimal model arm for context vector phi.
func (r *PrimalDualParetoBanditRouter) SelectArm(context []float64) (BanditRoutingDecision, error) {
	if len(context) != r.ContextDim {
		return BanditRoutingDecision{}, fmt.Errorf("mdm: expected context len %d, got %d", r.ContextDim, len(context))
	}
	ucbScores := make(map[string]float64, len(r.ArmOrder))
	bestArm := ""
	bestAcq := math.Inf(-1)

	for _, id := range r.ArmOrder {
		spec := r.Arms[id]
		theta := solveSPDLinearSystemGo(r.covA[id], r.vecB[id])
		z := solveSPDLinearSystemGo(r.covA[id], context)
		meanEst := 0.0
		quad := 0.0
		duelTerm := 0.0
		for i := 0; i < r.ContextDim; i++ {
			meanEst += theta[i] * context[i]
			quad += context[i] * z[i]
			duelTerm += r.duelW[id][i] * context[i]
		}
		if quad < 0 {
			quad = 0
		}
		ucb := meanEst + r.AlphaUCB*math.Sqrt(quad) + r.DuelingWeight*duelTerm
		ucbScores[id] = ucb
		acq := ucb - r.DualLambda*spec.CostUSD
		if bestArm == "" || acq > bestAcq+1e-12 {
			bestArm = id
			bestAcq = acq
		}
	}
	return BanditRoutingDecision{
		ChosenArm:             bestArm,
		UCBQualityScore:       ucbScores[bestArm],
		PrimalDualAcquisition: bestAcq,
		DualPriceLambda:       r.DualLambda,
		DualLatencyMu:         r.DualMu,
		ArmUCBScores:          ucbScores,
	}, nil
}

// Update updates discounted LinUCB statistics and primal-dual Lagrangian budget multiplier.
func (r *PrimalDualParetoBanditRouter) Update(chosenArm string, context []float64, reward float64) (float64, error) {
	spec, ok := r.Arms[chosenArm]
	if !ok {
		return r.DualLambda, fmt.Errorf("mdm: unknown chosenArm %q", chosenArm)
	}
	if len(context) != r.ContextDim {
		return r.DualLambda, fmt.Errorf("mdm: expected context len %d, got %d", r.ContextDim, len(context))
	}
	rho := r.RhoDiscount
	priorDiag := (1.0 - rho) * r.RidgeLambda
	for _, id := range r.ArmOrder {
		mat := r.covA[id]
		vec := r.vecB[id]
		for i := 0; i < r.ContextDim; i++ {
			vec[i] *= rho
			for j := 0; j < r.ContextDim; j++ {
				mat[i][j] *= rho
				if i == j {
					mat[i][j] += priorDiag
				}
			}
		}
	}
	cMat := r.covA[chosenArm]
	cVec := r.vecB[chosenArm]
	for i := 0; i < r.ContextDim; i++ {
		cVec[i] += reward * context[i]
		for j := 0; j < r.ContextDim; j++ {
			cMat[i][j] += context[i] * context[j]
		}
	}
	r.PullCounts[chosenArm]++
	r.DualLambda = math.Max(0.0, r.DualLambda+r.EtaDualCost*(spec.CostUSD-r.BudgetPerQueryUSD))
	return r.DualLambda, nil
}

// UpdateDuelingPreference performs a Bradley-Terry contextual preference update between winnerArm and loserArm.
func (r *PrimalDualParetoBanditRouter) UpdateDuelingPreference(winnerArm string, loserArm string, context []float64) error {
	if _, ok := r.Arms[winnerArm]; !ok {
		return fmt.Errorf("mdm: unknown winnerArm %q", winnerArm)
	}
	if _, ok := r.Arms[loserArm]; !ok {
		return fmt.Errorf("mdm: unknown loserArm %q", loserArm)
	}
	if len(context) != r.ContextDim {
		return fmt.Errorf("mdm: expected context len %d, got %d", r.ContextDim, len(context))
	}
	if winnerArm == loserArm {
		return nil
	}
	wWin := r.duelW[winnerArm]
	wLose := r.duelW[loserArm]
	diff := 0.0
	for i := 0; i < r.ContextDim; i++ {
		diff += (wWin[i] - wLose[i]) * context[i]
	}
	diff = math.Min(20.0, math.Max(-20.0, diff))
	probWin := 1.0 / (1.0 + math.Exp(-diff))
	gradScale := r.DuelingLR * (1.0 - probWin)
	for i := 0; i < r.ContextDim; i++ {
		wWin[i] = r.RhoDiscount*wWin[i] + gradScale*context[i]
		wLose[i] = r.RhoDiscount*wLose[i] - gradScale*context[i]
	}
	return nil
}

// PRMBranchNode represents a candidate reasoning step in a test-time compute tree.
type PRMBranchNode struct {
	NodeID             string
	ParentID           string
	Depth              int
	PriorProb          float64
	RubricProbs        map[string]float64
	IsTerminal         bool
	GroundTruthCorrect bool
}

// PRMTreeSearchResult summarizes PUCT + Conformal Branch-and-Bound tree search.
type PRMTreeSearchResult struct {
	SelectedLeafID    string
	SelectedPath      []string
	SelectedValueMean float64
	SelectedLCB       float64
	SelectedUCB       float64
	TotalNodes        int
	EvaluatedNodes    int
	PrunedNodes       int
	PruningRatio      float64
	LeafCorrect       bool
}

// ScorePRMRubricDistribution computes calibrated step value E[V] in [0, 1] and epistemic std sigma[V].
func ScorePRMRubricDistribution(rubricProbs map[string]float64, horizonDiscount float64) (float64, float64, error) {
	if len(rubricProbs) == 0 {
		return 0, 0, errors.New("mdm: rubricProbs must be non-empty")
	}
	if horizonDiscount <= 0 || math.IsNaN(horizonDiscount) {
		return 0, 0, fmt.Errorf("mdm: horizonDiscount must be > 0, got %v", horizonDiscount)
	}
	totalP := 0.0
	maxGrade := 3.0
	type gradeProb struct {
		g float64
		p float64
	}
	parsed := make([]gradeProb, 0, len(rubricProbs))
	for k, v := range rubricProbs {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, 0, fmt.Errorf("mdm: invalid rubric prob %v", v)
		}
		gInt, err := strconv.Atoi(strings.TrimSpace(k))
		if err != nil || gInt < 0 {
			return 0, 0, fmt.Errorf("mdm: rubric key %q must be a non-negative integer string", k)
		}
		g := float64(gInt)
		if g > maxGrade {
			maxGrade = g
		}
		parsed = append(parsed, gradeProb{g: g, p: v})
		totalP += v
	}
	if totalP <= 0 {
		return 0, 0, errors.New("mdm: rubricProbs sum must be > 0")
	}
	meanNorm := 0.0
	secondMom := 0.0
	for _, gp := range parsed {
		pNorm := gp.p / totalP
		gUnit := gp.g / maxGrade
		meanNorm += pNorm * gUnit
		secondMom += pNorm * gUnit * gUnit
	}
	varUnit := math.Max(0.0, secondMom-meanNorm*meanNorm)
	disc := math.Min(1.5, math.Max(0.05, horizonDiscount))
	calMean := math.Min(1.0, math.Max(0.0, meanNorm*disc))
	return calMean, math.Sqrt(varUnit), nil
}

// RunCalibratedPRMPUCTSearch runs PUCT + Conformal Branch-and-Bound tree search over a reasoning tree.
func RunCalibratedPRMPUCTSearch(
	nodes []PRMBranchNode,
	conformalZ float64,
	cPUCT float64,
	horizonDiscounts map[int]float64,
	enableConformalPruning bool,
) (PRMTreeSearchResult, error) {
	if len(nodes) == 0 {
		return PRMTreeSearchResult{}, errors.New("mdm: nodes must be non-empty")
	}
	childrenMap := make(map[string][]PRMBranchNode)
	for _, n := range nodes {
		childrenMap[n.ParentID] = append(childrenMap[n.ParentID], n)
	}
	for k := range childrenMap {
		list := childrenMap[k]
		sort.Slice(list, func(i, j int) bool {
			if list[i].PriorProb != list[j].PriorProb {
				return list[i].PriorProb > list[j].PriorProb
			}
			return list[i].NodeID < list[j].NodeID
		})
		childrenMap[k] = list
	}
	roots := childrenMap[""]
	if len(roots) == 0 {
		return PRMTreeSearchResult{}, errors.New("mdm: tree must have at least one root node with empty ParentID")
	}

	evaluatedIDs := make(map[string]bool)
	prunedIDs := make(map[string]bool)

	var bestLeaf *PRMBranchNode
	var bestPath []string
	bestVal := -1.0
	bestLCB := -1.0
	bestUCB := -1.0
	bestScore := math.Inf(-1)

	var countDescendants func(nid string) []string
	countDescendants = func(nid string) []string {
		desc := []string{nid}
		stack := []string{nid}
		for len(stack) > 0 {
			cur := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			for _, ch := range childrenMap[cur] {
				desc = append(desc, ch.NodeID)
				stack = append(stack, ch.NodeID)
			}
		}
		return desc
	}

	var explore func(node PRMBranchNode, pathSoFar []string, cumVal float64, cumVar float64, rank int, sibCount int) error
	explore = func(node PRMBranchNode, pathSoFar []string, cumVal float64, cumVar float64, rank int, sibCount int) error {
		disc := 1.0
		if horizonDiscounts != nil {
			if d, ok := horizonDiscounts[node.Depth]; ok {
				disc = d
			}
		}
		stepV, stepStd, err := ScorePRMRubricDistribution(node.RubricProbs, disc)
		if err != nil {
			return err
		}
		evaluatedIDs[node.NodeID] = true

		steps := float64(len(pathSoFar) + 1)
		newVal := cumVal + stepV
		newVar := cumVar + stepStd*stepStd
		pathMean := newVal / steps
		pathStd := math.Sqrt(newVar) / steps
		nodeLCB := math.Max(0.0, pathMean-conformalZ*pathStd)
		nodeUCB := math.Min(1.0, pathMean+conformalZ*pathStd)
		puctBonus := cPUCT * node.PriorProb * (math.Sqrt(float64(sibCount+1)) / float64(1+rank))

		chList := childrenMap[node.NodeID]
		isLeaf := node.IsTerminal || len(chList) == 0
		if enableConformalPruning && !isLeaf && bestLeaf != nil {
			if nodeUCB+0.02*puctBonus < bestLCB {
				for _, ch := range chList {
					for _, dID := range countDescendants(ch.NodeID) {
						if !evaluatedIDs[dID] {
							prunedIDs[dID] = true
						}
					}
				}
				return nil
			}
		}

		curPath := append(append([]string(nil), pathSoFar...), node.NodeID)
		if isLeaf {
			leafAcq := pathMean + 0.25*nodeLCB + 0.10*puctBonus
			if leafAcq > bestScore+1e-12 {
				bestScore = leafAcq
				nCopy := node
				bestLeaf = &nCopy
				bestPath = curPath
				bestVal = pathMean
				bestLCB = nodeLCB
				bestUCB = nodeUCB
			}
			return nil
		}

		for rIdx, ch := range chList {
			if enableConformalPruning && bestLeaf != nil {
				optChildUCB := (newVal+math.Min(1.0, 0.65+0.35*ch.PriorProb))/(steps+1.0) + conformalZ*pathStd
				if optChildUCB < bestLCB {
					for _, dID := range countDescendants(ch.NodeID) {
						if !evaluatedIDs[dID] {
							prunedIDs[dID] = true
						}
					}
					continue
				}
			}
			if err := explore(ch, curPath, newVal, newVar, rIdx, len(chList)); err != nil {
				return err
			}
		}
		return nil
	}

	for rIdx, rNode := range roots {
		if err := explore(rNode, nil, 0.0, 0.0, rIdx, len(roots)); err != nil {
			return PRMTreeSearchResult{}, err
		}
	}

	if bestLeaf == nil {
		first := roots[0]
		bestLeaf = &first
		bestPath = []string{first.NodeID}
		bestVal = 0.0
		bestLCB = 0.0
		bestUCB = 0.0
	}

	prunedCount := 0
	for id := range prunedIDs {
		if !evaluatedIDs[id] {
			prunedCount++
		}
	}
	totalN := len(nodes)
	return PRMTreeSearchResult{
		SelectedLeafID:    bestLeaf.NodeID,
		SelectedPath:      bestPath,
		SelectedValueMean: bestVal,
		SelectedLCB:       bestLCB,
		SelectedUCB:       bestUCB,
		TotalNodes:        totalN,
		EvaluatedNodes:    len(evaluatedIDs),
		PrunedNodes:       prunedCount,
		PruningRatio:      float64(prunedCount) / float64(totalN),
		LeafCorrect:       bestLeaf.GroundTruthCorrect,
	}, nil
}

// CorrelatedEnsembleCalibrationResult stores Covariance-Discounted Supra-Bayesian pooling weights.
type CorrelatedEnsembleCalibrationResult struct {
	AgentIDs                []string
	Weights                 map[string]float64
	MurphySkillScores       map[string]float64
	EffectiveSampleSize     float64
	MeanPairwiseCorrelation float64
	ShrinkageDelta          float64
}

func computeMurphySkillGo(seq []map[string]float64, labels []string) (float64, float64) {
	n := len(labels)
	if n == 0 {
		return 0.0, 0.05
	}
	const numBins = 10
	binSumConf := make([]float64, numBins)
	binSumHit := make([]float64, numBins)
	binCount := make([]int, numBins)
	totalHits := 0.0

	for i, dist := range seq {
		bestK := ""
		bestP := -1.0
		for k, v := range dist {
			if v > bestP || (v == bestP && k < bestK) {
				bestP = v
				bestK = k
			}
		}
		if bestP < 0 {
			bestP = 0
		}
		if bestP > 1 {
			bestP = 1
		}
		hit := 0.0
		if bestK == labels[i] {
			hit = 1.0
			totalHits += 1.0
		}
		bIdx := int(bestP * float64(numBins))
		if bIdx >= numBins {
			bIdx = numBins - 1
		}
		if bIdx < 0 {
			bIdx = 0
		}
		binSumConf[bIdx] += bestP
		binSumHit[bIdx] += hit
		binCount[bIdx]++
	}
	baseAcc := totalHits / float64(n)
	rel := 0.0
	res := 0.0
	for b := 0; b < numBins; b++ {
		if binCount[b] == 0 {
			continue
		}
		w := float64(binCount[b]) / float64(n)
		cMean := binSumConf[b] / float64(binCount[b])
		hMean := binSumHit[b] / float64(binCount[b])
		dRel := cMean - hMean
		dRes := hMean - baseAcc
		rel += w * dRel * dRel
		res += w * dRes * dRes
	}
	return res, rel
}

// CalibrateCorrelatedEnsembleWeights computes Covariance-Discounted Supra-Bayesian weights across M agents.
func CalibrateCorrelatedEnsembleWeights(
	calibAgentProbs map[string][]map[string]float64,
	calibLabels []string,
	shrinkageDelta float64,
) (CorrelatedEnsembleCalibrationResult, error) {
	if len(calibAgentProbs) == 0 || len(calibLabels) == 0 {
		return CorrelatedEnsembleCalibrationResult{}, errors.New("mdm: calibAgentProbs and calibLabels must be non-empty")
	}
	if shrinkageDelta <= 0 || shrinkageDelta > 1 || math.IsNaN(shrinkageDelta) {
		return CorrelatedEnsembleCalibrationResult{}, fmt.Errorf("mdm: shrinkageDelta must lie in (0, 1], got %v", shrinkageDelta)
	}
	n := len(calibLabels)
	agentIDs := make([]string, 0, len(calibAgentProbs))
	for aid := range calibAgentProbs {
		agentIDs = append(agentIDs, aid)
	}
	sort.Strings(agentIDs)
	mCount := len(agentIDs)

	residuals := make(map[string][]float64, mCount)
	skills := make(map[string]float64, mCount)

	for _, aid := range agentIDs {
		seq := calibAgentProbs[aid]
		if len(seq) != n {
			return CorrelatedEnsembleCalibrationResult{}, fmt.Errorf("mdm: agent %q length mismatch", aid)
		}
		resTerm, relTerm := computeMurphySkillGo(seq, calibLabels)
		accHits := 0.0
		resList := make([]float64, n)
		for idx, dist := range seq {
			lbl := calibLabels[idx]
			pGold := math.Min(1.0, math.Max(0.0, dist[lbl]))
			resList[idx] = 1.0 - pGold
			bestK := ""
			bestP := -1.0
			for k, v := range dist {
				if v > bestP || (v == bestP && k < bestK) {
					bestP = v
					bestK = k
				}
			}
			if bestK == lbl {
				accHits += 1.0
			}
		}
		acc := accHits / float64(n)
		skillVal := (resTerm + 0.02*acc + 0.005) / (relTerm + 0.04)
		skills[aid] = math.Max(1e-4, skillVal)
		residuals[aid] = resList
	}

	corrMat := make([][]float64, mCount)
	for i := 0; i < mCount; i++ {
		corrMat[i] = make([]float64, mCount)
		corrMat[i][i] = 1.0
	}
	offDiagSum := 0.0
	offDiagPairs := 0

	for i := 0; i < mCount; i++ {
		ri := residuals[agentIDs[i]]
		meanI, varI, normI := 0.0, 0.0, 0.0
		for _, x := range ri {
			meanI += x
			normI += x * x
		}
		meanI /= float64(n)
		normI = math.Sqrt(normI)
		for _, x := range ri {
			d := x - meanI
			varI += d * d
		}
		for j := i + 1; j < mCount; j++ {
			rj := residuals[agentIDs[j]]
			meanJ, varJ, normJ := 0.0, 0.0, 0.0
			for _, x := range rj {
				meanJ += x
				normJ += x * x
			}
			meanJ /= float64(n)
			normJ = math.Sqrt(normJ)
			for _, x := range rj {
				d := x - meanJ
				varJ += d * d
			}
			pearson := 0.0
			if varI > 1e-12 && varJ > 1e-12 {
				cov := 0.0
				for k := 0; k < n; k++ {
					cov += (ri[k] - meanI) * (rj[k] - meanJ)
				}
				pearson = cov / math.Sqrt(varI*varJ)
			}
			cosSim := 0.0
			if normI > 1e-12 && normJ > 1e-12 {
				dot := 0.0
				for k := 0; k < n; k++ {
					dot += ri[k] * rj[k]
				}
				cosSim = dot / (normI * normJ)
			}
			rhoIJ := math.Max(0.0, math.Min(1.0, 0.65*math.Max(0.0, pearson)+0.35*math.Max(0.0, cosSim)))
			corrMat[i][j] = rhoIJ
			corrMat[j][i] = rhoIJ
			offDiagSum += rhoIJ
			offDiagPairs++
		}
	}

	rReg := make([][]float64, mCount)
	skillVec := make([]float64, mCount)
	for i := 0; i < mCount; i++ {
		rReg[i] = make([]float64, mCount)
		skillVec[i] = skills[agentIDs[i]]
		for j := 0; j < mCount; j++ {
			if i == j {
				rReg[i][j] = 1.0
			} else {
				rReg[i][j] = (1.0 - shrinkageDelta) * corrMat[i][j]
			}
		}
	}
	rawU := solveSPDLinearSystemGo(rReg, skillVec)
	uSum := 0.0
	for i := range rawU {
		if rawU[i] < 0 {
			rawU[i] = 0
		}
		uSum += rawU[i]
	}
	weights := make(map[string]float64, mCount)
	for i, aid := range agentIDs {
		if uSum <= 1e-12 {
			weights[aid] = 1.0 / float64(mCount)
		} else {
			weights[aid] = rawU[i] / uSum
		}
	}
	meanCorr := 0.0
	if offDiagPairs > 0 {
		meanCorr = offDiagSum / float64(offDiagPairs)
	}
	nEff := float64(mCount) / (1.0 + float64(mCount-1)*meanCorr)
	return CorrelatedEnsembleCalibrationResult{
		AgentIDs:                agentIDs,
		Weights:                 weights,
		MurphySkillScores:       skills,
		EffectiveSampleSize:     nEff,
		MeanPairwiseCorrelation: meanCorr,
		ShrinkageDelta:          shrinkageDelta,
	}, nil
}

// CovarianceDiscountedLogitPool pools agent probability maps using Covariance-Discounted Supra-Bayesian weights.
func CovarianceDiscountedLogitPool(
	agentProbs map[string]map[string]float64,
	cal *CorrelatedEnsembleCalibrationResult,
	temperature float64,
) (map[string]float64, error) {
	if len(agentProbs) == 0 {
		return nil, errors.New("mdm: agentProbs must be non-empty")
	}
	if temperature <= 0 || math.IsNaN(temperature) {
		return nil, fmt.Errorf("mdm: temperature must be > 0, got %v", temperature)
	}
	agentIDs := make([]string, 0, len(agentProbs))
	for aid := range agentProbs {
		agentIDs = append(agentIDs, aid)
	}
	sort.Strings(agentIDs)

	firstDist := agentProbs[agentIDs[0]]
	if len(firstDist) == 0 {
		return nil, errors.New("mdm: option distribution must be non-empty")
	}
	optKeys := make([]string, 0, len(firstDist))
	for k := range firstDist {
		optKeys = append(optKeys, k)
	}
	sort.Strings(optKeys)

	wMap := make(map[string]float64, len(agentIDs))
	if cal == nil {
		for _, aid := range agentIDs {
			wMap[aid] = 1.0 / float64(len(agentIDs))
		}
	} else {
		wTot := 0.0
		for _, aid := range agentIDs {
			wTot += cal.Weights[aid]
		}
		for _, aid := range agentIDs {
			if wTot <= 1e-12 {
				wMap[aid] = 1.0 / float64(len(agentIDs))
			} else {
				wMap[aid] = cal.Weights[aid] / wTot
			}
		}
	}

	invTemp := 1.0 / temperature
	pooledLogits := make(map[string]float64, len(optKeys))
	for _, aid := range agentIDs {
		dist := agentProbs[aid]
		wM := wMap[aid]
		tot := 0.0
		for _, k := range optKeys {
			if dist[k] > 0 {
				tot += dist[k]
			}
		}
		if tot <= 0 {
			return nil, fmt.Errorf("mdm: agent %q distribution sum must be > 0", aid)
		}
		for _, k := range optKeys {
			pMK := math.Max(1e-9, dist[k]/tot)
			pooledLogits[k] += invTemp * wM * math.Log(pMK)
		}
	}

	maxL := math.Inf(-1)
	for _, k := range optKeys {
		if pooledLogits[k] > maxL {
			maxL = pooledLogits[k]
		}
	}
	out := make(map[string]float64, len(optKeys))
	z := 0.0
	for _, k := range optKeys {
		e := math.Exp(pooledLogits[k] - maxL)
		out[k] = e
		z += e
	}
	for _, k := range optKeys {
		out[k] /= z
	}
	return out, nil
}
