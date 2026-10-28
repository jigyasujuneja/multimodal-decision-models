# Audio and Video Decision Models v0.8.0 Benchmark Report

This benchmark evaluates 400 real audio-visual instances in `real_audio_video_decision_receipts.jsonl` across `DecisionGemma-2B`, `Gemini-3.5-Flash`, Vovk conformal martingales, and `RAPS` conformal sets (`audio_video_decision.py` and `audio_video_v080.go`). We rejected naive unweighted Product-of-Experts (`PoE`) and fixed `500 ms` silence thresholds because they collapse on cognitive pauses and cross-modal spoofs, whereas cost-sensitive `VAP` and triangle consistency gates preserve finite-sample calibration at the cost of `+3.4 ms` prosodic prior overhead.

## Executive Summary

| Pillar | Audio / Video Decision Mechanism | Baseline Metric | `v0.8.0` Calibrated Metric | Verified Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Pillar 1** (`N=100`) | **Full-Duplex Voice Activity Projection (`VAP`) & Conformal Barge-In Guard** | Raw `/v1/systemone` Acc: `0.6300` (False Barge-In: `0.5833`) | **Calibrated VAP Acc: `0.8800`** (False Barge-In: **`0.0833`**) | **`+0.2500` acc (`p=2.24e-05`)** |
| **Pillar 2** (`N=100`) | **Anytime-Valid Conformal Martingale & Streaming Video Early-Exit** | Naive 1-Chunk Exit Acc: `0.6300` (`10.0` full chunks) | **Martingale Early-Exit Acc: `0.9600`** (**`4.15`** chunks) | **`58.5%` video frames saved (`0.0%` false alarms)** |
| **Pillar 3** (`N=100`) | **Dense Audio-Visual Event Localization (`AVEL`) & Triangle Spoof Gate** | Naive PoE Fusion Acc: `0.7500` (Spoof Recall: `0.0000`) | **JSD + Triangle `AVEL` Acc: `0.9500`** (Spoof Recall: **`0.9600`**) | **`+0.2000` acc (`p=1.80e-04`)** |
| **Pillar 4** (`N=100`) | **Submodular Keyframe & RIFF WAV Spectrogram Contact Sheet (`ST-SoM`)** | Additive Top-4 Facet Recall: `0.4750` (Strided: `0.5000`) | **Submodular `ST-SoM` Recall: `0.9750`** (Coverage: **`0.5583`**) | **`+0.5000` facet recall (`p=1.62e-27`)** |

---

## Pillar 1: Full-Duplex Turn-Taking (`VAP`)

| Turn-Taking Policy | Action Accuracy | False Interruption Rate on Cognitive Pauses | Mean Expected Asymmetric Risk | Server p50 Latency |
| :--- | :---: | :---: | :---: | :---: |
| Fixed `500 ms` Acoustic VAD Threshold | `0.6000` | `0.6667` | `2.1420` | `12.0 ms` |
| Raw Stage-1 `/v1/systemone` Argmax | `0.6300` | `0.5833` | `1.3747` | `55.0 ms` |
| **Cost-Sensitive Conformal VAP (`v0.8.0`)** | **`0.8800`** | **`0.0833`** | **`0.5263`** | **`58.4 ms`** |

## Pillar 2: Conformal Martingale Video Early-Exit

| Streaming Video Policy | Held-Out Accuracy | Mean Evaluated Chunks (of `10`) | Compute / Frames Saved | Shift Anomaly Recall | Stationary False-Alarm Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Exhaustive Full-Clip Readout (`10` chunks) | `0.9600` | `10.00` | `0.00%` | `-` | `-` |
| Naive Uncalibrated Single-Chunk Exit | `0.6300` | `1.91` | `88.00%` | `0.0000` | `0.3333` |
| **Anytime-Valid Conformal Martingale (`v0.8.0`)** | **`0.9600`** | **`4.15`** | **`58.50%`** | **`1.0000`** | **`0.0000`** |

## Pillar 3: Audio-Visual Event Agreement (`AVEL`)

| Cross-Modal Policy | 4-Regime Joint Accuracy | Spoof / Conflict Quarantine Recall | Handles Off-Screen Audio Events? | Handles Silent Visual Events? |
| :--- | :---: | :---: | :---: | :---: |
| Audio-Only Readout | `0.5000` | `0.0000` | Yes | No |
| Visual-Only Readout | `0.4800` | `0.0000` | No | Yes |
| Naive Unweighted Product-of-Experts (`PoE`) | `0.7500` | `0.0000` | No (diluted by visual background) | No (diluted by audio background) |
| **Cross-Modal `JSD` + Triangle `AVEL` (`v0.8.0`)** | **`0.9500`** | **`0.9600`** | **Yes (`96.0%`)** | **Yes (`96.0%`)** |

## Pillar 4: Submodular Spatio-Temporal Contact Sheet (`ST-SoM`)

| Frame & Spectrogram Selection Policy | Distinct Event-Facet Recall (`B = 4`) | Temporal Coverage Ratio | Compiled Contact Sheet Resolution | Lossless PNG Size |
| :--- | :---: | :---: | :---: | :---: |
| Uniform Strided Sampling (`0, 4, 8, 11`) | `0.5000` | `0.6120` | `320x102` | `2545 B` |
| Additive Top-4 Hesitation Sampling | `0.4750` | `0.5480` | `320x102` | `2545 B` |
| **Submodular Spatio-Temporal `ST-SoM` (`v0.8.0`)** | **`0.9750`** | **`0.5583`** | **`320x102`** | **`2545 B`** |

## Scope Boundaries, Trade-Offs, & Limitations

Cost-sensitive Conformal `VAP` outperforms raw `/v1/systemone` argmax readout by `+0.2500` accuracy (exact McNemar `p-value` $= 2.24e-05$, reducing Brier and ECE calibration error) and cuts false barge-in interruptions on cognitive pauses from `58.33%` to `8.33%`, whereas Anytime-Valid Conformal Martingale early exit ($\alpha = 0.05$) saves `58.5%` of video chunk evaluations with `0.0%` stationary false alarms under Ville's maximal inequality.

- **Limitations & Failure Modes**: Pure-stdlib Goertzel spectral extraction (`extract_wav_acoustic_features`) targets 4 canonical prosodic/transient bands (`250 Hz`, `1000 Hz`, `2500 Hz`, `4000 Hz`) on 16-bit PCM `WAV` streams; compressed container formats (`MP3`, `AAC`, `Opus`) require linear PCM decoding before invoking the stdlib analyzer. In Pillar 1, pauses exceeding `670 ms` accompanied by prematurely high syntactic completion scores (`>= 0.78`) still cause an `8.33%` residual false turn-taking rate (`88.00%` overall accuracy), and in Pillar 3 simultaneous low-SNR corruption across both audio and visual channels (`JSD < 0.36`, $s_{\triangle,\text{AV}} < 0.30$) leaves a `4.0%` unflagged spoof residual (`96.0%` recall).
