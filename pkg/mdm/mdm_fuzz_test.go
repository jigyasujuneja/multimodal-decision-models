package mdm

import (
	"fmt"
	"math"
	"testing"
	"time"
)

func TestSafeExpectedMarkCenterAndUnitVolumeConsistency(t *testing.T) {
	marks := []CandidateMark{
		{ID: "M_left", Label: "Cancel", Box: [4]float64{45, 5, 55, 15}},
		{ID: "M_right", Label: "Confirm", Box: [4]float64{45, 85, 55, 95}},
	}
	probs := map[string]float64{"M_left": 0.48, "M_right": 0.52, "none_visible": 0.0}

	rawY, rawX, ok := ExpectedMarkCenterFromProbabilities(probs, marks, "none_visible")
	if !ok {
		t.Fatalf("expected raw center to succeed")
	}
	if (rawX >= 5 && rawX <= 15) || (rawX >= 85 && rawX <= 95) {
		t.Fatalf("expected bimodal raw center (%.1f, %.1f) to fall in empty space between marks", rawY, rawX)
	}

	safeY, safeX, chosen, safeOk := SafeExpectedMarkCenter(probs, marks, "none_visible", 18.0)
	if !safeOk {
		t.Fatalf("expected SafeExpectedMarkCenter to succeed")
	}
	if chosen.ID != "M_right" {
		t.Fatalf("expected bimodal snap to M_right, got %q", chosen.ID)
	}
	if safeY < 45 || safeY > 55 || safeX < 85 || safeX > 95 {
		t.Fatalf("expected snapped coordinates inside M_right, got (%.1f, %.1f)", safeY, safeX)
	}

	// Unit volume + multipack consistency
	if ValidatePackQuantityConsistency("San Pellegrino 6x330ml Cans", "San Pellegrino 6x500ml Cans", "exact_duplicate", "identical_pack_and_volume") {
		t.Fatalf("expected 6x330ml vs 6x500ml to fail ValidatePackQuantityConsistency")
	}
	if !ValidatePackQuantityConsistency("San Pellegrino 6x330ml Cans", "San Pellegrino 6 x 0.33L Cans", "exact_duplicate", "identical_pack_and_volume") {
		t.Fatalf("expected 6x330ml vs 6x0.33L to pass ValidatePackQuantityConsistency")
	}
	if ValidatePackQuantityConsistency("Coffee Beans 1kg", "Coffee Beans 500g", "exact_duplicate", "identical_pack_and_volume") {
		t.Fatalf("expected 1kg vs 500g to fail ValidatePackQuantityConsistency")
	}

	// Sparse graph scaling: V = 2,000 nodes completes in < 500 ms even under race/fuzz instrumentation
	edges := make([]PairwiseEdge, 0, 2010)
	for i := 0; i < 2000; i++ {
		n0 := fmt.Sprintf("n_%04d", i)
		n1 := fmt.Sprintf("n_%04d", (i+1)%2000)
		edges = append(edges, PairwiseEdge{NodeA: n0, NodeB: n1, EquivProb: 0.92})
		if i%500 == 0 {
			n2 := fmt.Sprintf("n_%04d", (i+2)%2000)
			edges = append(edges, PairwiseEdge{NodeA: n0, NodeB: n2, EquivProb: 0.06})
		}
	}
	t0 := time.Now()
	hard := CheckTransitiveTriangles(edges, 0.80, 0.25)
	soft := ScoreProbabilisticTriangles(edges, 0.50)
	elapsed := time.Since(t0)
	if len(hard) != 4 || len(soft) != 4 {
		t.Fatalf("expected 4 contradictions, got hard=%d soft=%d", len(hard), len(soft))
	}
	if elapsed > 500*time.Millisecond {
		t.Fatalf("expected sparse V=2000 triangle scan < 500ms, took %v", elapsed)
	}
}

func FuzzNormalizeCosinePriorAndFusion(f *testing.F) {
	f.Add(0.52, 0.44, -0.15, 0.07, 0.35)
	f.Add(0.99, 0.98, 0.97, 0.05, 0.50)
	f.Add(-0.80, -0.20, 0.10, 0.10, 0.0)
	f.Fuzz(func(t *testing.T, s1, s2, s3, temp, weight float64) {
		if math.IsNaN(s1) || math.IsNaN(s2) || math.IsNaN(s3) || math.IsInf(s1, 0) || math.IsInf(s2, 0) || math.IsInf(s3, 0) {
			return
		}
		if math.IsNaN(temp) || math.IsInf(temp, 0) || temp <= 1e-4 || temp > 100.0 {
			return
		}
		if math.IsNaN(weight) || math.IsInf(weight, 0) || weight < 0.0 || weight > 10.0 {
			return
		}
		scores := map[string]float64{"a": s1, "b": s2, "c": s3}
		lp := NormalizeCosinePriorLogits(scores, temp)
		if len(lp) != 3 {
			t.Fatalf("expected 3 log-prior entries, got %d", len(lp))
		}
		probSum := 0.0
		for _, v := range lp {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("non-finite log-prior: %v", v)
			}
			probSum += math.Exp(v)
		}
		if math.Abs(probSum-1.0) > 1e-6 {
			t.Fatalf("expected exp(log_prior) sum == 1.0, got %.9f", probSum)
		}

		raw := map[string]float64{"a": 0.40, "b": 0.35, "c": 0.25}
		fusedProbs, _, normH, rawNormH, _ := FusePriorWithGate(raw, lp, []string{"a", "b", "c"}, 1.0, weight, 0.16, "hesitant")
		if math.IsNaN(normH) || math.IsNaN(rawNormH) || rawNormH < 0 || rawNormH > 1.000001 {
			t.Fatalf("invalid normalized entropies: normH=%v rawNormH=%v", normH, rawNormH)
		}
		fSum := 0.0
		for _, p := range fusedProbs {
			if p < 0 || p > 1.000001 || math.IsNaN(p) {
				t.Fatalf("invalid fused probability: %v", p)
			}
			fSum += p
		}
		if math.Abs(fSum-1.0) > 1e-6 {
			t.Fatalf("expected fused sum == 1.0, got %.9f", fSum)
		}
	})
}
