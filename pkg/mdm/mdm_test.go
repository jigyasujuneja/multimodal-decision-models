package mdm

import (
	"fmt"
	"math"
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
}

func TestShortlistSomAndTriangles(t *testing.T) {
	crit := make(map[string]string, 40)
	priors := make(map[string]float64, 40)
	for i := 0; i < 39; i++ {
		k := fmt.Sprintf("sku_%02d", i)
		crit[k] = fmt.Sprintf("Item %d", i)
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
