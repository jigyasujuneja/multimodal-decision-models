package mdm

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"
)

func TestFusePriorWithGateModes(t *testing.T) {
	keys := []string{"opt_a", "opt_b", "opt_c"}
	uncertain := map[string]float64{"opt_a": 0.46, "opt_b": 0.44, "opt_c": 0.10}
	prior := map[string]float64{"opt_a": -1.0, "opt_b": 2.5, "opt_c": -1.0}

	// Confident mode skips hesitant inputs while preserving rawNormH.
	_, winConf, _, rawH1, fusedConf := FusePriorWithGate(uncertain, prior, keys, 1.0, 0.35, 0.16, "confident")
	if fusedConf || winConf != "opt_a" || rawH1 < 0.16 {
		t.Fatalf("expected confident mode to skip hesitant fusion: fused=%v win=%s rawH=%.4f", fusedConf, winConf, rawH1)
	}

	// Hesitant mode fuses on uncertain inputs to break the near-tie while keeping rawNormH unchanged.
	_, winHes, _, rawH2, fusedHes := FusePriorWithGate(uncertain, prior, keys, 1.0, 0.35, 0.16, "hesitant")
	if !fusedHes || winHes != "opt_b" || math.Abs(rawH1-rawH2) > 1e-12 {
		t.Fatalf("expected hesitant mode to flip tie to opt_b and preserve rawH: fused=%v win=%s", fusedHes, winHes)
	}

	// Always mode fuses regardless of hesitation while still preserving rawNormH.
	_, winAlw, _, rawH3, fusedAlw := FusePriorWithGate(uncertain, prior, keys, 1.0, 0.35, 0.16, "always")
	if !fusedAlw || winAlw != "opt_b" || math.Abs(rawH1-rawH3) > 1e-12 {
		t.Fatalf("expected always mode to fuse and preserve rawH: fused=%v win=%s", fusedAlw, winAlw)
	}

	// Off mode never fuses.
	_, winOff, _, _, fusedOff := FusePriorWithGate(uncertain, prior, keys, 1.0, 0.35, 0.16, "off")
	if fusedOff || winOff != "opt_a" {
		t.Fatalf("expected off mode to leave winner unchanged: fused=%v win=%s", fusedOff, winOff)
	}
}

func TestShortlistContrastiveTrieSomAndTriangles(t *testing.T) {
	crit := make(map[string]string, 40)
	priors := make(map[string]float64, 40)
	for i := 0; i < 39; i++ {
		k := fmt.Sprintf("sku_%02d", i)
		crit[k] = fmt.Sprintf("Sparkling Water Variant %d", i)
		priors[k] = float64(i)
	}
	crit["none_of_the_above"] = "None match"
	if got := TopKFromPriorLogits(crit, nil, 6, "none_of_the_above"); got != nil {
		t.Fatalf("expected nil shortlist when priorLogits is empty, got %v", got)
	}
	short := TopKFromPriorLogits(crit, priors, 5, "none_of_the_above")
	if len(short) != 6 || short[0] != "sku_38" || short[5] != "none_of_the_above" {
		t.Fatalf("unexpected shortlist: %v", short)
	}

	cues := BuildContrastiveCriteria(
		map[string]string{
			"card_arrival":      "Track physical card delivery",
			"card_linking":      "Link existing card to app",
			"none_of_the_above": "None match",
		},
		[]string{"card_arrival", "card_linking", "none_of_the_above"},
		"none_of_the_above",
		2,
	)
	if !strings.Contains(cues["card_arrival"], "[distinguishing: arrival, track]") {
		t.Fatalf("expected sibling-unique distinguishing cue on card_arrival, got %q", cues["card_arrival"])
	}
	if strings.Contains(cues["none_of_the_above"], "[distinguishing:") {
		t.Fatalf("catch-all option should not get contrastive cue: %q", cues["none_of_the_above"])
	}

	// Trie-constrained joint decode eliminates invalid (c1=dom_a, c2=role_2) pair.
	leaves := []TaxonomyLeaf{
		{LeafKey: "dom_a_role_1", CoarseKey: "dom_a", FineKey: "role_1"},
		{LeafKey: "dom_b_role_2", CoarseKey: "dom_b", FineKey: "role_2"},
		{LeafKey: "oos", CoarseKey: "oos", FineKey: "oos"},
	}
	coarse := map[string]float64{"dom_a": 0.55, "dom_b": 0.40, "oos": 0.05}
	fine := map[string]float64{"role_1": 0.25, "role_2": 0.70, "oos": 0.05}
	leafProbs, trieWin := DecodeTaxonomyTrie(coarse, fine, leaves, "oos")
	if trieWin != "dom_b_role_2" || leafProbs["dom_b_role_2"] <= leafProbs["dom_a_role_1"] {
		t.Fatalf("expected valid path dom_b_role_2 (0.40*0.70=0.28) > dom_a_role_1 (0.55*0.25=0.1375), got win=%s probs=%v", trieWin, leafProbs)
	}

	marks := []CandidateMark{
		{ID: "M1", Label: "Header", Box: [4]float64{0, 0, 20, 100}},
		{ID: "M2", Label: "Icon", Box: [4]float64{4, 90, 10, 98}},
	}
	m, dist, ok := SelectContainingOrNearestMark(7, 94, marks)
	if !ok || m.ID != "M2" || dist != 0 {
		t.Fatalf("expected nested mark M2, got %+v dist=%v", m, dist)
	}

	cs := CheckTransitiveTriangles([]PairwiseEdge{
		{NodeA: "a", NodeB: "b", EquivProb: 0.92},
		{NodeA: "b", NodeB: "c", EquivProb: 0.88},
		{NodeA: "a", NodeB: "c", EquivProb: 0.10},
	}, 0.80, 0.25)
	if len(cs) != 1 || cs[0].WeakEdge != [2]string{"a", "c"} {
		t.Fatalf("expected 1 contradiction on (a,c), got %+v", cs)
	}
	if ParsePackMultiplier("Sparkling Water 6x330ml") != 6 {
		t.Fatalf("expected pack multiplier 6")
	}
}

func TestNormalizeCosinePriorPackedBracketsGrid4x4AndProbabilisticMonteCarlo(t *testing.T) {
	// 1. Packed bracket pass accounting (K=59 -> 2, K=151 -> 2, K=202 -> 3)
	if got := PackedBracketPassCount(26, 20, 8); got != 1 {
		t.Fatalf("expected 1 pass for K=26, got %d", got)
	}
	if got := PackedBracketPassCount(59, 20, 8); got != 2 {
		t.Fatalf("expected 2 passes for K=59, got %d", got)
	}
	if got := PackedBracketPassCount(151, 20, 8); got != 2 {
		t.Fatalf("expected 2 passes for K=151, got %d", got)
	}
	if got := PackedBracketPassCount(202, 20, 8); got != 3 {
		t.Fatalf("expected 3 passes for K=202, got %d", got)
	}

	// 2. Temperature-scaled log-softmax cosine priors sum to 1 in probability space.
	cos := map[string]float64{"a": 0.51, "b": 0.42, "c": 0.28}
	lp := NormalizeCosinePriorLogits(cos, DefaultCosinePriorTemperature)
	probSum := 0.0
	for _, v := range lp {
		probSum += math.Exp(v)
	}
	if math.Abs(probSum-1.0) > 1e-9 || lp["a"] <= lp["b"] || lp["b"] <= lp["c"] {
		t.Fatalf("unexpected log-softmax cosine priors: sum=%.9f lp=%v", probSum, lp)
	}

	// 3. 4x4 UI spatial grid and probability-weighted expected mark center.
	box := [4]float64{30.0, 60.0, 42.0, 74.0}
	if cell := AssignBoxToGrid4x4(box); cell != "r1_c2" {
		t.Fatalf("expected r1_c2, got %s", cell)
	}
	bounds, ok := Grid4x4CellBounds("r1_c2")
	if !ok || bounds != [4]float64{25, 50, 50, 75} {
		t.Fatalf("unexpected r1_c2 bounds: ok=%v bounds=%v", ok, bounds)
	}
	cy, cx, ok := ExpectedMarkCenterFromProbabilities(
		map[string]float64{"M1": 0.80, "M2": 0.10, "none_visible": 0.10},
		[]CandidateMark{
			{ID: "M1", Label: "Save", Box: [4]float64{10, 20, 20, 40}},
			{ID: "M2", Label: "Cancel", Box: [4]float64{10, 60, 20, 80}},
		},
		"none_visible",
	)
	if !ok || math.Abs(cy-15.0) > 1e-6 || math.Abs(cx-(0.8*30.0+0.1*70.0)/0.9) > 1e-6 {
		t.Fatalf("unexpected expected mark center: ok=%v cy=%.4f cx=%.4f", ok, cy, cx)
	}

	// 4. Seeded 400-trial Monte Carlo probabilistic calibration under induced prevalence shift
	// and continuous triangle contradiction separation.
	rng := rand.New(rand.NewSource(20261023))
	keys := []string{"pos", "neg"}
	shiftPrior := map[string]float64{"pos": math.Log(0.90), "neg": math.Log(0.10)}
	brierBase, brierConf, brierHes := 0.0, 0.0, 0.0
	confidentFlips := 0
	wrongPrior := map[string]float64{"pos": -4.0, "neg": 4.0}

	for trial := 0; trial < 400; trial++ {
		isPos := rng.Float64() < 0.90
		var pPos float64
		if trial%3 == 0 {
			// Confident prediction (rawNormH < 0.16)
			if isPos {
				pPos = 0.985 - 0.01*rng.Float64()
			} else {
				pPos = 0.005 + 0.01*rng.Float64()
			}
			_, winWrong, _, rawHW, _ := FusePriorWithGate(
				map[string]float64{"pos": pPos, "neg": 1.0 - pPos},
				wrongPrior, keys, 1.0, 0.35, 0.16, "hesitant",
			)
			baseWin := "neg"
			if pPos > 0.5 {
				baseWin = "pos"
			}
			if rawHW < 0.16 && winWrong != baseWin {
				confidentFlips++
			}
		} else {
			// Ambiguous prediction under label shift
			if isPos {
				pPos = 0.45 + 0.16*rng.Float64()
			} else {
				pPos = 0.39 + 0.16*rng.Float64()
			}
		}
		raw := map[string]float64{"pos": pPos, "neg": 1.0 - pPos}
		y := 0.0
		if isPos {
			y = 1.0
		}
		baseP, _, _, _, _ := FusePriorWithGate(raw, shiftPrior, keys, 1.0, 0.35, 0.16, "off")
		confP, _, _, _, _ := FusePriorWithGate(raw, shiftPrior, keys, 1.0, 0.35, 0.16, "confident")
		hesP, _, _, _, _ := FusePriorWithGate(raw, shiftPrior, keys, 1.0, 0.35, 0.16, "hesitant")

		brierBase += (baseP["pos"] - y) * (baseP["pos"] - y)
		brierConf += (confP["pos"] - y) * (confP["pos"] - y)
		brierHes += (hesP["pos"] - y) * (hesP["pos"] - y)
	}
	if confidentFlips != 0 {
		t.Fatalf("expected 0 flips on confident items under wrong prior in hesitant mode, got %d", confidentFlips)
	}
	if brierHes >= brierBase*0.88 || brierHes >= brierConf*0.88 {
		t.Fatalf("expected hesitant gate to materially reduce Brier under 90%% shift: base=%.4f conf=%.4f hes=%.4f", brierBase/400, brierConf/400, brierHes/400)
	}

	// Soft triangle contradiction caught by ScoreProbabilisticTriangles even below hard 0.80/0.25 cutoffs.
	softTris := ScoreProbabilisticTriangles([]PairwiseEdge{
		{NodeA: "x1", NodeB: "x2", EquivProb: 0.76},
		{NodeA: "x2", NodeB: "x3", EquivProb: 0.78},
		{NodeA: "x1", NodeB: "x3", EquivProb: 0.28},
	}, 0.20)
	if len(softTris) != 1 || softTris[0].WeakEdge != [2]string{"x1", "x3"} || softTris[0].ContradictionScore < 0.40 {
		t.Fatalf("expected soft triangle contradiction on (x1,x3), got %+v", softTris)
	}
}

func TestFusePriorWithGateNeverAndUnknownModes(t *testing.T) {
	raw := map[string]float64{"a": 0.52, "b": 0.48}
	prior := map[string]float64{"a": 0.0, "b": 4.0}
	keys := []string{"a", "b"}

	offProbs, offWin, _, _, offFused := FusePriorWithGate(raw, prior, keys, 1.0, 0.8, 0.16, "off")
	neverProbs, neverWin, _, _, neverFused := FusePriorWithGate(raw, prior, keys, 1.0, 0.8, 0.16, "never")
	unkProbs, unkWin, _, _, unkFused := FusePriorWithGate(raw, prior, keys, 1.0, 0.8, 0.16, "unknown_mode")

	if offFused || neverFused || unkFused {
		t.Fatalf("expected fused=false for off/never/unknown, got off=%v never=%v unk=%v", offFused, neverFused, unkFused)
	}
	if offWin != "a" || neverWin != "a" || unkWin != "a" {
		t.Fatalf("expected winner 'a', got off=%q never=%q unk=%q", offWin, neverWin, unkWin)
	}
	for _, k := range keys {
		if math.Abs(offProbs[k]-neverProbs[k]) > 1e-12 || math.Abs(offProbs[k]-unkProbs[k]) > 1e-12 {
			t.Fatalf("mismatch across off/never/unknown on key %q", k)
		}
	}
}

func TestExtended255CodeBijectionAndReasoningToReadoutDistillation(t *testing.T) {
	if len(Universal256DecisionCodes) != 256 {
		t.Fatalf("expected 256 universal codes, got %d", len(Universal256DecisionCodes))
	}
	opts := make([]string, 255)
	for i := 0; i < 255; i++ {
		opts[i] = "opt_" + Universal256DecisionCodes[i]
	}
	codeToKey, err := BuildExtended255CodeBijection(opts)
	if err != nil {
		t.Fatalf("unexpected error building 255-code bijection: %v", err)
	}
	if len(codeToKey) != 255 {
		t.Fatalf("expected 255 bijection entries, got %d", len(codeToKey))
	}
	for _, splitCode := range []string{"BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "FJ", "FQ", "FV", "FZ", "GJ", "GQ", "GX", "GZ"} {
		if _, exists := codeToKey[splitCode]; exists {
			t.Fatalf("multi-token BPE split code %q must not appear in Universal256DecisionCodes", splitCode)
		}
	}

	teacher := map[string]float64{"0": 0.05, "1": 0.80, "2": 0.15}
	base := map[string]float64{"0": 0.25, "1": 0.55, "2": 0.20}
	student := map[string]float64{"0": 0.12, "1": 0.70, "2": 0.18}
	res, err := ReasoningToReadoutDistillationLoss(teacher, student, base, "score", 1.0, 0.25, 0.20, 1.50, 1.10)
	if err != nil {
		t.Fatalf("unexpected error in ReasoningToReadoutDistillationLoss: %v", err)
	}
	if res.TotalLoss <= 0.0 || res.OrdinalScoreLoss <= 0.0 {
		t.Fatalf("unexpected distillation result: %+v", res)
	}
	sumP := 0.0
	for _, p := range res.DistilledPosterior {
		sumP += p
	}
	if math.Abs(sumP-1.0) > 1e-9 {
		t.Fatalf("distilled posterior does not sum to 1: %f", sumP)
	}
}

