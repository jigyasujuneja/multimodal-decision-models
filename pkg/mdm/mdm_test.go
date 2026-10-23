package mdm

import (
	"fmt"
	"math"
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
