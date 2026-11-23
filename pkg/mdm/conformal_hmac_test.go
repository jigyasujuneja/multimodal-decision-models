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
	"reflect"
	"sync"
	"testing"
)

func TestHmacUint64ExactParityVector(t *testing.T) {
	got := HmacUint64("dgem-conformal-v1", "req-42:intent", "u")
	const want uint64 = 14094744155160802019
	if got != want {
		t.Fatalf("HmacUint64 parity mismatch: got %d, want %d", got, want)
	}
	emptyKeyDigest := HmacUint64("", "req-42:intent", "u")
	if emptyKeyDigest != want {
		t.Fatalf("empty seedKey did not default to DefaultConformalSeedKey: got %d, want %d", emptyKeyDigest, want)
	}
	uniformVal := HmacUniformFloat64("dgem-conformal-v1", "req-42:intent", "u")
	if math.Abs(uniformVal-0.7640776116826353) > 1e-12 {
		t.Fatalf("HmacUniformFloat64 mismatch: got %.16f, want 0.7640776116826353", uniformVal)
	}
}

func TestStatelessHMACConformalOrderIndependenceAndConcurrency(t *testing.T) {
	calibProbs := []map[string]float64{
		{"allow": 0.45, "ask": 0.45, "deny": 0.10},
		{"allow": 0.20, "ask": 0.40, "deny": 0.40},
		{"allow": 0.70, "ask": 0.15, "deny": 0.15},
		{"allow": 0.33, "ask": 0.33, "deny": 0.34},
	}
	calibLabels := []string{"ask", "deny", "allow", "ask"}
	itemKeys := []string{"row-0", "row-1", "row-2", "row-3"}

	cfg := DefaultConformalConfig()
	cfg.Randomized = true
	cfg.SeedKey = "parity-key-2026"

	qForward, err := CalibrateConformalQuantileWithKeys(calibProbs, calibLabels, itemKeys, 0.25, cfg)
	if err != nil {
		t.Fatalf("CalibrateConformalQuantileWithKeys forward failed: %v", err)
	}

	revProbs := []map[string]float64{calibProbs[3], calibProbs[2], calibProbs[1], calibProbs[0]}
	revLabels := []string{calibLabels[3], calibLabels[2], calibLabels[1], calibLabels[0]}
	revKeys := []string{itemKeys[3], itemKeys[2], itemKeys[1], itemKeys[0]}

	qReversed, err := CalibrateConformalQuantileWithKeys(revProbs, revLabels, revKeys, 0.25, cfg)
	if err != nil {
		t.Fatalf("CalibrateConformalQuantileWithKeys reversed failed: %v", err)
	}
	if math.Abs(qForward-qReversed) > 1e-15 {
		t.Fatalf("stateless HMAC q_hat depended on calibration order: %v vs %v", qForward, qReversed)
	}

	tieProbs := map[string]float64{"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
	evalCfg := cfg
	evalCfg.ItemKey = "req-991"
	baselineSet, err := ConformalPredictionSet(tieProbs, 0.55, evalCfg)
	if err != nil {
		t.Fatalf("ConformalPredictionSet failed: %v", err)
	}

	var waitGroup sync.WaitGroup
	for worker := 0; worker < 16; worker++ {
		waitGroup.Add(1)
		go func(workerID int) {
			defer waitGroup.Done()
			otherCfg := cfg
			otherCfg.ItemKey = "noise-item"
			_, _ = ConformalPredictionSet(tieProbs, 0.55, otherCfg)
			gotSet, setErr := ConformalPredictionSet(tieProbs, 0.55, evalCfg)
			if setErr != nil {
				t.Errorf("worker %d error: %v", workerID, setErr)
				return
			}
			if !reflect.DeepEqual(gotSet.PredictionSet, baselineSet.PredictionSet) {
				t.Errorf("worker %d prediction set mismatch: got %v, want %v", workerID, gotSet.PredictionSet, baselineSet.PredictionSet)
			}
		}(worker)
	}
	waitGroup.Wait()
}
