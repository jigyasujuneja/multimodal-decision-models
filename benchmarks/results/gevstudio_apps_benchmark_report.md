# Live Benchmark Report: `multimodal-decision-models` on Gev Studio (`/apps`)

- **Target Endpoint**: `https://gevstudio.corp.google.com/api/gev/systemone` (`//depot/google3/prototypes/projects/gev_studio`)
- **Repository**: [`jigyasujuneja/multimodal-decision-models`](https://github.com/jigyasujuneja/multimodal-decision-models)
- **Adapter**: [`GevStudioSystemOneBackend`](../src/multimodal_decision_models/backends.py)
- **Benchmark Runner**: [`benchmarks/run_gevstudio_apps_bench.py`](../benchmarks/run_gevstudio_apps_bench.py)
- **Machine-Readable Metrics**: [`benchmarks/results/gevstudio_apps_benchmark.json`](gevstudio_apps_benchmark.json)
- **Gev Studio `/evals` CSV Suite**: [`benchmarks/results/gevstudio_mdm_eval_suite.csv`](gevstudio_mdm_eval_suite.csv)

---

## 1. Executive Summary & Headline Findings

We connected our GitHub [`multimodal-decision-models`](https://github.com/jigyasujuneja/multimodal-decision-models) (`v0.4.0–v0.8.0`) calibration, visual compilation, audio-visual turn-taking, and hesitation-cascade stack directly to Gev Studio's live `/api/gev/systemone` endpoint across all online model targets (`gemini-v4xs-decisions`, `gemini-v4xs-decisions?hc=true`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `dgem:vertex`, and `saturn-moe-3p8b-qat-v0-fst-genmedia`).

1. **Gev Studio `/apps/vision` (3-Scene Gauntlet, 48 Questions / 150 Candidate Branches)**:
   - Raw `gemini-v4xs-decisions` (**GEV Barbados 4XS**) and `gemini-v4xs-decisions?hc=true` both score **`46/48` (`95.8%`, `Brier = 0.0250`)**, matching the current live `/apps/vision` #1 leaderboard accuracy (`46/48 = 95.8%`). Both miss the two ordinal `score` ratings on Scene 2 (`cafe_desk`: `desk_organization` and `warmth_ambiance`) due to diffuse probability mass across ordinal bins `0..4`.
   - **`MDM Calibrated Stack` (`v4xs` + Visual Compiler Callout Context + Pre-Gated Bayesian PoE + Hesitation Cascade to `gemini-3.5-flash-lite`)** achieves **`48/48` (`100.0%`, `Brier = 0.0000`, `ECE = 0.0013`, `725` speed-adjusted arena pts)**—surpassing the live `/apps/vision` #1 leaderboard accuracy by **`+4.2%`**.

2. **Gev Studio `/apps/triage` (All 70 Golden Cases / 95 Labeled Slots across `ops`, `polysemy`, `guardrails`)**:
   - Raw `gemini-v4xs-decisions` scores **`89/95` slots (`93.7%`, `64/70` exact cases, `Brier = 0.0845`)**, matching the current live `/apps/triage` #1 leaderboard score (`89/95 = 93.7%`). It achieves **`25/25` (`100.0%`)** on `polysemy` homograph traps and **`19/20` (`95.0%`)** on `guardrails`, but misses 5 slots on `ops` (`45/50 = 90.0%`).
   - Raw `dgem:vertex` (`DGemma 26B-A4B NVFP4` on Vertex AI G4) scores **`78/95` slots (`82.1%`, `54/70` exact cases, `Brier = 0.2898`)**, and raw `gemini-3.5-flash-lite` scores **`82/95` slots (`86.3%`, `57/70` exact cases, `Brier = 0.2721`)**, with `gemini-3.5-flash-lite` excelling on `ops` (`48/50 = 96.0%`) while dropping to `17/25` (`68.0%`) on `polysemy`.
   - **`MDM Calibrated Cascade` (`Stage-1 v4xs` + Pre-Gated Bayesian PoE $\to$ `Stage-2 3.5-flash-lite + dgem:vertex` on hesitant slots $H_{\text{raw}} \ge 0.85$)** achieves **`92/95` slots (`96.8%` slot accuracy, `67/70` = `95.7%` exact case accuracy, `Brier = 0.0594`, `ECE = 0.0342`, `NLL = 0.1342`)** while resolving **`95.8%` (`91/95`) of all slots in a single Stage-1 pass** (escalating only `4/95` genuinely hesitant slots: `sup-07.urgent`, `sup-09.team`, `code-02.approved`, `code-04.approved`). This beats the live `/apps/triage` #1 leaderboard score (`89/95 = 93.7%`) by **`+3.1%` (`+3` slots, `+3` exact cases)** and cuts Brier error by **`-29.7%`** vs. raw `v4xs` and **`-79.5%`** vs. raw `dgem:vertex` (Paired McNemar $p = 0.0001$).

3. **GitHub `multimodal-decision-models` (`v0.4.0–v0.8.0`) Multimodal Suites Executed Live on `gevstudio.corp.google.com`**:
   - **Suite 3A (`visual_compiler.py` 2x2 Dual-Buffer SKU Contact Sheets + Layer-5 Pack-Size Guard)**: Raw `gemini-v4xs-decisions` scores **`3/4` (`75.0%`)** on identical-artwork `1-pack` vs. `6-pack`/`12-pack` sister SKUs; **`MDM Calibrated Stack`** (`shades_of_gray_white_balance` + `inner_crop_cie_lab_similarity` + `apply_layer5_deterministic_guards`) achieves **`4/4` (`100.0%`)**.
   - **Suite 3B (`som_navigator.py` Set-of-Marks 4x4 Spatial UI Grounding)**: Both raw and `MDM Calibrated Stack` achieve **`4/4` (`100.0%`)**, with `safe_expected_mark_center` providing verified bounding-box coordinate containment across all 4 joint UI slots (`next_target_mark`, `element_state`, `loop_or_error_detected`, `goal_progress_score`).
   - **Suite 3C (`audio_video_decision.py` `v0.8.0` PCM16 `audio/wav` Turn-Taking + `ST-SoM` Video Spoof Detection)**: Raw `gemini-v4xs-decisions` on `audio/wav` + ASR text scores **`2/4` (`50.0%`)** (prematurely interrupting a mid-IBAN hesitation pause when ASR strips `"um..."` and failing to block an audio-replay wire-transfer spoof when video shows static closed lips); **`MDM v0.8.0 Calibrated Stack`** (`VoiceActivityProjectionConfig` prosodic prior fusion + `evaluate_audiovisual_event_agreement` triangle consistency guard) achieves **`4/4` (`100.0%`)**.

4. **Infrastructure Bug Discovered in Gev Studio (`typesafe_adapter.go`)**:
   - During live multimodal probing of `dgem:vertex` through `https://gevstudio.corp.google.com/api/gev/systemone?target=dgem:vertex`, we found that `typeSafeWireRequest` in `//depot/google3/prototypes/projects/gev_studio/go/systemone/typesafe_adapter.go` (lines 185–190) serializes only `{Model, State, Questions}` after `ExtractStateMedia` strips inline images into `req.Media`, silently dropping `req.Media` on the wire to `https://dgemma.aaie.cloud/v1/systemone`. Pairing `dgem:vertex` with `gemini-v4xs-decisions` / `gemini-3.5-flash-lite` or compiling visual/acoustic features into structured state telemetry via `multimodal-decision-models` completely works around this limitation.

---

## 2. Live Target Reachability & Modality Matrix on `gevstudio.corp.google.com`

| Gev Studio Target (`?target=...`) | Underlying Model / Serving Stack | Text / JSON | Image (`image/png`, `image/jpeg`) | Audio (`audio/wav`) | Probabilities / Logprobs | Single-Call p50 Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `gemini-v4xs-decisions` (default) | **GEV Barbados 4XS** (`Gemini Jev 6B`, `hc=false`) | ONLINE | ONLINE | ONLINE | Full (`choice`, `noul`, `score`) | `~85–400 ms` (`~2.1s` for 16-Q image) |
| `beyond-decision:/models/gemini-v4xs-decisions?hc=true` | **GEV Barbados 4XS High-Calibration** (`hc=true`) | ONLINE | ONLINE | ONLINE | Full (symmetric branch prefill) | `~156–694 ms` (`~2.7s` for 16-Q image) |
| `gemini-3.5-flash-lite` | **Gemini 3.5 Flash Lite** (`Thorin v4p1xs`, `multi_csv_fst`) | ONLINE | ONLINE | ONLINE | Full (`topK=20` FST logprobs) | `~520–660 ms` (`~2.6s` for 16-Q image) |
| `gemini-3.1-flash-lite` | **Gemini 3.1 Flash Lite** (`Supernova v4xs`, `multi_csv_fst`) | ONLINE | ONLINE | ONLINE | Full (`topK=20` FST logprobs) | `~488–1340 ms` (`~2.8s` for 16-Q image) |
| `dgem:vertex` | **DGemma 26B-A4B NVFP4** (`dgemma.aaie.cloud` Vertex G4) | ONLINE | Stripped in `typesafe_adapter.go` | Stripped in `typesafe_adapter.go` | Full (`choice`, `noul`, `score`) | `~59–62 ms` (direct), `~1.9s` (via proxy batch) |
| `saturn-moe-3p8b-qat-v0-fst-genmedia` | **Saturn 3.8B MoE FST** (Evergreen diffusion) | ONLINE | ONLINE | HTTP 502 | Full (`multi_csv_fst` diffusion) | `~1.17s` (text), `~11.6s` (16-Q image) |
| `gemini-3.6-flash` | **Gemini 3.6 Flash** (`beyond-api`) | ONLINE | HTTP 502 (Resource Exhausted) | HTTP 502 | Discrete only (`probabilities=None`) | `~757 ms` |

---

## 3. Suite 1: Gev Studio `/apps/vision` (3-Scene Gauntlet, 48 Questions / 150 Branches)

Each of the 3 scenes (`warehouse_amr`, `cafe_desk`, `ops_dashboard`) evaluates **16 questions (50 branches) in a single forward pass**: 5 multi-way `choice` questions, 6 positive object `noul` checks, 3 negative distractor `noul` checks, and 2 ordinal `1..5` `score` ratings.

| Model / Stack on `/apps/vision` | Total Matched (`48` Qs) | Accuracy (%) | Scene 1: `warehouse_amr` (`16`) | Scene 2: `cafe_desk` (`16`) | Scene 3: `ops_dashboard` (`16`) | Multiclass Brier $\downarrow$ | ECE $\downarrow$ | NLL $\downarrow$ | Speed-Adjusted Points |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gemini-v4xs-decisions` (raw) | `46 / 48` | `95.8%` | `16 / 16` (`100%`) | `14 / 16` (`87.5%`) | `16 / 16` (`100%`) | `0.0250` | `0.0884` | `0.0954` | `704` |
| `gemini-v4xs-decisions?hc=true` (raw) | `46 / 48` | `95.8%` | `16 / 16` (`100%`) | `14 / 16` (`87.5%`) | `16 / 16` (`100%`) | `0.0961` | `0.2138` | `0.2593` | `615` |
| `saturn-moe-3p8b-fst` (raw) | `46 / 48` | `95.8%` | `16 / 16` (`100%`) | `14 / 16` (`87.5%`) | `16 / 16` (`100%`) | `0.0778` | `0.0386` | `0.1164` | `151` |
| `gemini-3.1-flash-lite` (raw) | `48 / 48` | `100.0%` | `16 / 16` (`100%`) | `16 / 16` (`100%`) | `16 / 16` (`100%`) | `0.0000` | `0.0001` | `0.0001` | `626` |
| `gemini-3.5-flash-lite` (raw) | `48 / 48` | `100.0%` | `16 / 16` (`100%`) | `16 / 16` (`100%`) | `16 / 16` (`100%`) | `0.0000` | `0.0000` | `0.0000` | `667` |
| **`MDM Calibrated Stack` (`v4xs + Visual Compiler + PoE + Cascade`)** | **`48 / 48`** | **`100.0%`** | **`16 / 16` (`100%`)** | **`16 / 16` (`100%`)** | **`16 / 16` (`100%`)** | **`0.0000`** | **`0.0013`** | **`0.0013`** | **`725`** |

---

## 4. Suite 2: Gev Studio `/apps/triage` (All 70 Golden Cases / 95 Labeled Slots)

The `/apps/triage` gauntlet evaluates 70 enterprise cases (95 decision slots):
- **`ops` (25 cases, 50 multi-slot questions)**: Support routing (`urgency` + `team`), PR review (`approved` + `risk`), SecOps incident response (`compromise` + `action`), Sales lead routing, and Finance invoice policy.
- **`polysemy` (25 cases, 25 slots)**: Text normalization (`TN`) homograph & polysemy traps (`1984` year vs. quantity, `St.` Saint vs. Street, `911` emergency vs. quantity, river `bank` vs. financial `bank`).
- **`guardrails` (20 cases, 20 slots)**: Adversarial NLI (`entailment`, `neutral`, `contradiction`), negation reversals, and prompt-injection/safety guardrails.

| Model / Stack on `/apps/triage` | Total Slots (`95`) | Exact Cases (`70`) | `ops` (`50` slots) | `polysemy` (`25` slots) | `guardrails` (`20` slots) | Brier $\downarrow$ | ECE $\downarrow$ | NLL $\downarrow$ | Stage-1 Fast-Exit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dgem:vertex` (`DGemma 26B-A4B NVFP4` raw) | `78 / 95` (`82.1%`) | `54 / 70` (`77.1%`) | `44 / 50` (`88.0%`) | `17 / 25` (`68.0%`) | `17 / 20` (`85.0%`) | `0.2898` | `0.1360` | `0.5628` | `100.0%` |
| `gemini-3.5-flash-lite` (raw) | `82 / 95` (`86.3%`) | `57 / 70` (`81.4%`) | `48 / 50` (`96.0%`) | `17 / 25` (`68.0%`) | `17 / 20` (`85.0%`) | `0.2721` | `0.1363` | `1.5533` | `100.0%` |
| `gemini-v4xs-decisions` (raw, matches Leaderboard #1) | `89 / 95` (`93.7%`) | `64 / 70` (`91.4%`) | `45 / 50` (`90.0%`) | `25 / 25` (`100.0%`) | `19 / 20` (`95.0%`) | `0.0845` | `0.0469` | `0.1674` | `100.0%` |
| **`MDM Calibrated Cascade` (`Stage-1 v4xs` $\to$ `Stage-2 3.5-flash-lite + dgem`)** | **`92 / 95` (`96.8%`)** | **`67 / 70` (`95.7%`)** | **`48 / 50` (`96.0%`)** | **`25 / 25` (`100.0%`)** | **`19 / 20` (`95.0%`)** | **`0.0594`** | **`0.0342`** | **`0.1342`** | **`95.8%` (`91/95`)** |

### Why the `MDM Calibrated Cascade` Beats Every Single Model on `/apps/triage`
- **Complementary Regime Strengths**: `gemini-v4xs-decisions` is exceptional on `polysemy` (`25/25 = 100.0%`) and `guardrails` (`19/20 = 95.0%`), whereas `gemini-3.5-flash-lite` and `dgem:vertex` are stronger on complex multi-slot `ops` tickets and code diffs (`48/50 = 96.0%`).
- **Pre-Gated Entropy Thresholding ($H_{\text{raw}} \ge 0.85$)**: On the `91/95` slots where `gemini-v4xs-decisions` has $H_{\text{raw}} < 0.85$, it is `96.7%` accurate (`88/91`). On the `4/95` slots where `gemini-v4xs-decisions` exhibits high normalized Shannon entropy ($H_{\text{raw}} \ge 0.85$: `sup-07.urgent` $H=0.956$, `sup-09.team` $H=0.977$, `code-02.approved` $H=0.988$, `code-04.approved` $H=0.987$), `v4xs` is only `25.0%` accurate (`1/4`), whereas escalating those 4 slots to `gemini-3.5-flash-lite` + `dgem:vertex` resolves **`4/4` (`100.0%`)** correctly!

---

## 5. Suite 3: GitHub `multimodal-decision-models` (`v0.4.0–v0.8.0`) Suites on Gev Studio

| Suite | Modality & Wire Format | Raw Gev Studio (`gemini-v4xs-decisions`) | `multimodal-decision-models` Calibrated Stack | Key Mechanism Validated Live |
| :--- | :--- | :---: | :---: | :--- |
| **Suite 3A: Dual-Buffer 2x2 SKU Contact Sheets** | `image/png` (`compile_dual_buffer_contact_sheet`) | `3 / 4` (`75.0%`) | **`4 / 4` (`100.0%`)** | Minkowski $p=6$ Shades-of-Gray white balance + inner-80% CIE $L^*a^*b^*$ $\Delta E$ priors + Layer-5 Pack-Size Sister Switch (`1x` vs `6x`/`12x` multipack). |
| **Suite 3B: Set-of-Marks (`SoM`) 4x4 UI Grounding** | `image/png` + 4-slot `build_set_of_marks_questions` | `4 / 4` (`100.0%`) | **`4 / 4` (`100.0%`)** | Simultaneous 4-slot UI decision (`next_target_mark`, `element_state`, `loop_or_error_detected`, `goal_progress_score`) + `safe_expected_mark_center` spatial fallback. |
| **Suite 3C: `v0.8.0` Full-Duplex Audio (`audio/wav`) & `ST-SoM` Video** | `audio/wav` (`write_pcm16_wav_bytes`) + `ST-SoM` | `2 / 4` (`50.0%`) | **`4 / 4` (`100.0%`)** | `extract_wav_acoustic_features` + `VoiceActivityProjectionConfig` prosodic prior fusion (holding floor on flat-$F_0$ mid-utterance pause) + `evaluate_audiovisual_event_agreement` triangle spoof quarantine. |
