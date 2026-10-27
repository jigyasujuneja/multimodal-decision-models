# Calibrated Multimodal Decision Models at Catalog Scale: Model-Agnostic Readouts, Dual-Buffer Visual Contact Sheets, and Split Conformal Cascades

**Authors:** Jigyasu Juneja  
**Version:** 0.4.0 (October 2026)  
**Repository:** [https://github.com/jigyasujuneja/multimodal-decision-models](https://github.com/jigyasujuneja/multimodal-decision-models)

## Abstract

Visual and relational classification across catalogs of $K = 2$ to $10,000+$ classes fails when systems rely on uncalibrated single-model completions, naive cosine retrieval priors, or unguarded multi-turn context windows. We present `multimodal-decision-models`, a zero-dependency Python and Go framework that decouples probability calibration, finite-sample set-valued abstention, dual-buffer visual compilation, and trajectory context pruning from the underlying model inference engine. First, we formalize a unified readout contract spanning four inference paradigms: parallel masked diffusion (`/v1/systemone`), autoregressive single-token `[A-Z]` logprobs (`/v1/chat/completions`), causal Evidence-First schema verifiers, and logprob-free sampled models smoothed under a symmetric Dirichlet prior ($\alpha_0 = 0.5$) with exact epistemic variance. Second, we prove that evaluating Regularized Adaptive Prediction Sets (RAPS) and Shannon hesitation on raw Stage-1 probabilities prior to external Product-of-Experts fusion preserves finite-sample marginal coverage $\mathbb{P}(Y_{n+1} \in \mathcal{C}_\alpha(X_{n+1})) \ge 1 - \alpha$ while preventing miscalibrated retrieval priors from masking model uncertainty.

Third, we introduce a dual-buffer visual compiler that separates illuminant-normalized Minkowski $p = 6$ Shades-of-Gray retrieval buffers from local contrast-stretched contact sheets augmented with inter-candidate pixel variance saliency bands and $2\times$ net-weight zoom strips. Fourth, we formulate agent trajectory context compression as a single-pass 8-slot Multiple-Choice Knapsack Problem (MCKP) and enforce `<untrusted_data>` XML quarantine against indirect prompt injection. Across 10 evaluated models and 20 experimental arms, out-of-fold temperature scaling and pre-fusion RAPS cascades achieve $0.1419$ Brier score and $0.0177$ selective AURC on multi-task calibration ($N = 50$), $0.9694$ accuracy and $0.0020$ AURC at $175.8\text{ ms}$ on XSTest safety boundaries ($N = 360$, exact McNemar $p = 7.84 \times 10^{-10}$ for cascade escalation), $58.85\%$ trajectory token reduction at $100.00\%$ required-block recall ($p \le 7.45 \times 10^{-9}$), and $0.00\%$ attack success rate on a 220-query adversarial RAG benchmark.

## 1. Problem Formulation

Deploying multimodal classifiers at catalog scale exposes four failure modes that motivate our architecture:

1. **Incompatible Output Surfaces and Overconfidence**: Parallel masked diffusion deciders return full simplex distributions over 8 question slots simultaneously, autoregressive endpoints expose restricted top-$k$ token logprobs, structured APIs enforce JSON schemas, and black-box reasoning endpoints return discrete text completions without token logprobs. Treating raw model outputs as calibrated probabilities yields severe reliability error ($\operatorname{REL} > 0.17$) and unbounded negative log-likelihood on black-box draws.
2. **Scale Mismatch in Retrieval Prior Fusion**: Raw dense-embedding cosine similarities over a top-$M$ shortlist cluster in a narrow band ($\Delta \cos \approx 0.10$ to $0.16$). Multiplying raw cosine scores by a fusion weight $w = 0.35$ shifts log-odds by only $\approx 0.05\text{ nats}$, which is too small to flip a hesitant Stage-1 prediction ($\ln(0.44 / 0.40) = 0.095\text{ nats}$). Conversely, applying temperature-scaled log-softmax priors unconditionally overwrites confident visual judgments when retrieval priors reflect popularity bias or shifted base rates.
3. **Photometric Distortion in Single-Buffer Visual Pipelines**: Applying $3 \times 3$ local contrast stretching to product crops improves fine-print legibility for Stage-2 visual verifiers, but distorts brand packaging chromaticity in CIE $L^*a^*b^*$ space and degrades Stage-2A dense-embedding retrieval.
4. **Context Exhaustion and Indirect Prompt Injection**: Multi-step coding and catalog agents accumulate verbose tool outputs that exceed context budgets, while retrieved product descriptions or document chunks frequently contain adversarial instruction overrides targeting the scoring prompt.

## 2. Four-Paradigm Readout and Dirichlet Consensus

Let $\mathcal{Q} = \{q_1, \dots, q_S\}$ denote a batch of $S \le 8$ decision slots, where each slot $q_s$ defines an ordered option set $\mathcal{O}_s = (o_{s,1}, \dots, o_{s,K_s})$ with $2 \le K_s \le 26$. Every backend adapter maps $\mathcal{Q}$ onto a validated probability simplex $\Delta^{K_s - 1}$:

- **Parallel Masked Diffusion (`DgemSystemOneBackend`)**: Packs up to $8$ slots $\times 26$ options into a single forward pass over `/v1/systemone`, using `"lines"` formatting when $S \le 8$ and $K_s \le 4$ or `"compact"` formatting for wider option sets.
- **Autoregressive Single-Token Bijection (`OpenAILogprobBackend`)**: Maps $o_{s,k}$ onto uppercase ASCII tokens $\texttt{A}, \dots, \texttt{Z}$, restricts decoding to `max_tokens=1`, and normalizes exponentiated `top_logprobs` over the exact $K_s$ bijected letters.
- **Causal Evidence-First Schema Verifier (`VertexGeminiBackend`)**: Enforces `propertyOrdering` in structured JSON generation so `evidence_read` (verbatim optical character recognition and visual feature readout) is decoded before any slot choice enum. Under lower-triangular causal attention, slot logits condition directly on the generated evidence tokens.
- **Symmetric Dirichlet Consensus (`BlackBoxSampledBackend`)**: For endpoints that do not expose token logprobs, the adapter draws $M$ completions (or parses verbalized vote counts $c_{s,1}, \dots, c_{s,K_s}$ with $\sum_{k=1}^{K_s} c_{s,k} = M$) under a restricted enum grammar. Placing a symmetric Dirichlet prior $\operatorname{Dir}(\alpha_0, \dots, \alpha_0)$ with $\alpha_0 = 0.5$ (Jeffreys prior) yields the posterior predictive distribution and exact per-option epistemic variance:

$$p_{s,k} = \frac{c_{s,k} + \alpha_0}{A_0}, \qquad \operatorname{Var}(\theta_{s,k} \mid c_s) = \frac{p_{s,k}(1 - p_{s,k})}{A_0 + 1}, \qquad A_0 = M + K_s \alpha_0$$

Because $p_{s,k} \in [\alpha_0 / A_0, (M + \alpha_0) / A_0] \subset (0, 1)$ for all finite $M$, cross-entropy loss remains bounded, normalized Shannon hesitation remains strictly positive on small sample sizes, and total epistemic variance $\sum_{k=1}^{K_s} \operatorname{Var}(\theta_{s,k} \mid c_s)$ decreases monotonically with $M$.

## 3. Calibration, Prior Fusion, and Conformal Guarantees

### 3.1 Pre-Gated Bayesian Product-of-Experts and Log-Softmax Priors

Given raw retrieval cosines $z_1, \dots, z_K \in [-1, 1]$ for a candidate shortlist, `normalize_cosine_prior_logits` maps cosines onto a centered log-probability scale at temperature $\tau > 0$ (default $\tau = 0.07$):

$$\pi_k = \frac{z_k - \max_j z_j}{\tau} - \ln \sum_{i=1}^K \exp\left(\frac{z_i - \max_j z_j}{\tau}\right)$$

At $\tau = 0.07$, a cosine margin of $0.14$ expands to $\Delta \pi = 2.0\text{ nats}$. To prevent strong priors from corrupting confident model readouts, `fuse_with_prior` computes cardinality-normalized Shannon hesitation on the raw Stage-1 probabilities $p_{\text{raw}}$ before applying any external prior:

$$\tilde{H}_{\text{raw}}(p_{\text{raw}}) = -\frac{1}{\ln K}\sum_{k=1}^K p_{\text{raw},k}\ln p_{\text{raw},k} \in [0, 1]$$

The fused distribution is then computed as:

$$\ln p_{\text{fused}}(k) \propto \frac{1}{T}\ln p_{\text{raw},k} + w_{\text{eff}}(\tilde{H}_{\text{raw}}) \cdot \pi_k$$

where $w_{\text{eff}}(\tilde{H}_{\text{raw}})$ equals $w \cdot \mathbb{I}[\tilde{H}_{\text{raw}} \ge \tau_H]$ under `"hesitant"` gating, $w \cdot \tilde{H}_{\text{raw}}$ under `"proportional"` gating, $w \cdot \mathbb{I}[\tilde{H}_{\text{raw}} < \tau_H]$ under `"confident"` gating, and $w$ under `"always"` gating.

### 3.2 Pre-Fusion Split Conformal Prediction and RAPS

Let $\{(X_i, Y_i)\}_{i=1}^n$ be exchangeable calibration examples and let $p_{(1)}(x) \ge p_{(2)}(x) \ge \dots \ge p_{(K)}(x)$ denote the sorted raw Stage-1 probabilities for input $x$, with $r(x, y)$ denoting the $1$-based rank of label $y$ (ties broken deterministically by canonical option order). `conformal.py` implements three nonconformity scores:

$$s_{\text{lac}}(x, y) = 1 - p_y(x)$$

$$s_{\text{aps}}(x, y) = \sum_{j < r(x, y)} p_{(j)}(x) + u \cdot p_y(x), \qquad u \in [0, 1]$$

$$s_{\text{raps}}(x, y) = s_{\text{aps}}(x, y) + \lambda_{\text{reg}} \max\bigl(0, r(x, y) - k_{\text{reg}}\bigr)$$

Sorting the calibration nonconformity scores $s_{(1)} \le s_{(2)} \le \dots \le s_{(n)}$ and selecting order index $m = \lceil (n + 1)(1 - \alpha) \rceil$ (with $\hat{q} = +\infty$ if $m > n$) yields the prediction set $\mathcal{C}_\alpha(x) = \{y \in \mathcal{O} : s(x, y) \le \hat{q}\}$. By exchangeability of $(s(X_1, Y_1), \dots, s(X_{n+1}, Y_{n+1}))$, the rank of $s(X_{n+1}, Y_{n+1})$ among all $n + 1$ scores is uniformly distributed on $\{1, \dots, n + 1\}$ and guarantees finite-sample marginal coverage:

$$\mathbb{P}\bigl(Y_{n+1} \in \mathcal{C}_\alpha(X_{n+1})\bigr) \ge 1 - \alpha$$

In `CascadeDecisionOrchestrator`, $\mathcal{C}_\alpha(x)$ is evaluated on $p_{\text{raw}}$ before prior fusion. The orchestrator escalates $x$ to the Stage-2 verifier whenever $|\mathcal{C}_\alpha(x)| \ne 1$, an abstention option (`none_of_the_above` / `abstain`) enters $\mathcal{C}_\alpha(x)$, or $\tilde{H}_{\text{raw}}(p_{\text{raw}}) \ge \tau_H$.

### 3.3 Post-Hoc Temperature and Regularized Vector Scaling

To calibrate raw model logits $z_i \in \mathbb{R}^K$ on held-out folds without overfitting rare classes:

- **Scalar Temperature Scaling (`fit_temperature_scaling`)**: Minimizes $L_2$-regularized cross-entropy over log-temperature $\eta = \ln T \in [\ln 0.05, \ln 20.0]$ using an 81-point grid search warm start followed by damped Newton-Raphson iterations:

$$\mathcal{L}(\eta) = -\frac{1}{N}\sum_{i=1}^N \ln \operatorname{softmax}_y\bigl(e^{-\eta} z_i\bigr) + \frac{\lambda}{2}\eta^2$$

- **Diagonal Vector Scaling (`fit_vector_scaling`)**: Fits per-class scale $a_k$ and bias $b_k$ via L-BFGS with diagonal Gauss-Newton preconditioning under shrinkage toward scalar temperature $T_0$:

$$\mathcal{L}(a, b) = -\frac{1}{N}\sum_{i=1}^N \ln \operatorname{softmax}_y(a \odot z_i + b) + \frac{\lambda}{2}\sum_{k=1}^K \left(\left(a_k - \frac{1}{T_0}\right)^2 + b_k^2\right)$$

### 3.4 Exact Within-Bin Murphy Brier Decomposition

For multiclass probability vectors $p_i \in \Delta^{K-1}$ and one-hot targets $y_i \in \{0, 1\}^K$, binning predictions into $B$ partitions generates within-bin variance whenever predictions inside a bin are continuous rather than identical. `brier_murphy_decomposition` computes the exact four-term identity:

$$\operatorname{Brier} = \frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K (p_{i,k} - y_{i,k})^2 = \operatorname{REL} - \operatorname{RES} + \operatorname{UNC} + \operatorname{WBV}$$

where $\operatorname{REL} = \frac{1}{N}\sum_{b=1}^B N_b \|\bar{p}_b - \bar{y}_b\|_2^2$, $\operatorname{RES} = \frac{1}{N}\sum_{b=1}^B N_b \|\bar{y}_b - \bar{y}\|_2^2$, $\operatorname{UNC} = \sum_{k=1}^K \bar{y}_k(1 - \bar{y}_k)$, and $\operatorname{WBV} = \frac{1}{N}\sum_{b=1}^B \sum_{i \in \mathcal{I}_b} \|p_i - \bar{p}_b\|_2^2$. Accounting for $\operatorname{WBV}$ keeps the decomposition identity exact to machine precision ($< 10^{-12}$) when comparing discrete rubric grades against continuous expected-grade tie-breaking.

## 4. Dual-Buffer Visual Photometry and Trajectory Pruning

### 4.1 Dual-Buffer Visual Compiler (`visual_compiler.py`)

`visual_compiler.py` implements zero-dependency PNG (`IHDR`, `IDAT` scanline filters `0..4`, `IEND`, IEEE 802.3 CRC32) and PPM (`P6`/`P3`) encoders/decoders to maintain two isolated photometric buffers:

1. **Stage-2A Retrieval Buffer**: Applies Minkowski $p = 6$ Shades-of-Gray illuminant normalization (`shades_of_gray_white_balance`) without local contrast stretching, preserving packaging chromaticity for inner-80% CIE $L^*a^*b^*$ $\Delta E$ comparison (`inner_crop_cie_lab_similarity`), Lidstone-smoothed container aspect priors (`lidstone_container_aspect_log_prior`), and the residual InfoNCE adapter $e' = \operatorname{norm}(q + \alpha W_2 \operatorname{GELU}(W_1 q))$ (`apply_residual_mlp_adapter`). When top-1 retrieval probability satisfies $P_1 - P_2 \ge 0.14$, $Z \ge 1.65$, and the top candidates do not share a sister pack-size conflict, `stage2a_fast_exit_gate` resolves the query in zero VLM calls.
2. **Stage-2B Verification Contact Sheet**: Clones the query and candidate images, applies $3 \times 3$ local contrast stretching, computes the horizontal band maximizing inter-candidate pixel variance $V(y) = \operatorname{Mean}_{x,c}\bigl(\operatorname{Var}_k R_k(x, y, c)\bigr)$ (`extract_inter_candidate_saliency_band`), and renders a composite contact sheet containing $2\times$ zoomed saliency and bottom-30% net-weight strips plus optional $1.15\times$ horizontal cylindrical unwarping.

### 4.2 Single-Pass 8-Slot MCKP Context Pruning (`context_pruner.py`)

Given $n$ trajectory context blocks $b_1, \dots, b_n$ with full token costs $t_{\text{full},i}$ and stub costs $t_{\text{stub},i} < t_{\text{full},i}$, `build_context_pruning_questions` packs up to 8 blocks per `/v1/systemone` pass with options `{"full", "stub", "drop"}`. Using expected utility weights:

$$u_i(\text{full}) = p_{i,\text{full}} + 0.5\,p_{i,\text{stub}}, \qquad u_i(\text{stub}) = 0.6\,p_{i,\text{full}} + p_{i,\text{stub}}, \qquad u_i(\text{drop}) = 0$$

`prune_agent_context_blocks` solves the Multiple-Choice Knapsack Problem over $c_i \in \{\text{full}, \text{stub}, \text{drop}\}$ (restricting $c_i \in \{\text{full}, \text{stub}\}$ whenever block $i$ is `pinned`) via dominance-pruned Pareto dynamic programming under total token budget $B$.

## 5. Empirical Evaluation

### 5.1 Multi-Model Calibration and Conformal Benchmark (`N = 50` and `N = 360`)

We evaluate 10 models and 20 experimental arms across two frozen panels in `benchmarks/run_multimodel_benchmark.py`:
- **Panel 1 (`multimodel_calibration_50`, $N = 50$, $11$ clusters, $K \in [2, 26]$)**: Evaluates masked diffusion (`dgem v0.3.7`, `DecisionGemma`), compact 2B deciders (`Strands Decider 2B v19`), verbalized-confidence frontier models (`Gemini 3.8 Flash`), and 3-draw symmetric Dirichlet ($\alpha_0 = 0.5$) consensus models (`Gemini 3.8 Flash`, `Gemini 3.6 Flash`, `Gemini 3.5 Flash Lite`), alongside 5-fold out-of-fold temperature scaling and pre-fusion RAPS ($\alpha = 0.10$) cascades.
- **Panel 2 (`safety_xstest_360`, $N = 360$, $12$ clusters, $K = 2$)**: Evaluates safety refusal calibration on XSTest across embedding probes (`Classy v0.3.3`, `Classier v0.2.0`), `Strands Decider 2B v19`, `dgem v0.3.1`, and `DecisionGemma`.

| Panel & Arm | Acc | Brier (95% Cluster CI) | Murphy REL / RES | NLL | ECE | AURC | Conformal Cov@0.05 / @0.10 | Mean Set@0.10 | Latency (ms) | Pareto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P1:** `Gemini 3.8 Flash (Verbalized + TempScaled OOF)` | 0.920 | 0.1419 ([0.0246, 0.2633]) | 0.0576 / 0.8365 | 0.3404 | 0.0231 | 0.0177 | 0.960 / 0.920 | 1.00 | 4247.8 | yes |
| **P1:** `Gemini 3.8 Flash (Verbalized Conf)` | 0.920 | 0.1452 ([0.0102, 0.2774]) | 0.0440 / 0.8196 | 0.3531 | 0.0484 | 0.0191 | 0.960 / 0.920 | 1.02 | 4247.8 | no |
| **P1:** `DecisionGemma` | 0.920 | 0.1796 ([0.0833, 0.2887]) | 0.1342 / 0.8753 | 0.3194 | 0.1364 | 0.0204 | 0.960 / 0.920 | 1.04 | 278.3 | yes |
| **P1:** `dgem v0.3.7 (TempScaled OOF)` | 0.860 | 0.2047 ([0.0638, 0.3326]) | 0.1327 / 0.8488 | 0.4534 | 0.0951 | 0.1175 | 0.960 / 0.920 | 1.14 | 389.6 | no |
| **P1:** `dgem v0.3.7` | 0.860 | 0.2071 ([0.0542, 0.3463]) | 0.0988 / 0.8125 | 0.4778 | 0.0973 | 0.0996 | 0.960 / 0.920 | 1.14 | 389.6 | no |
| **P1:** `Cascade: dgem v0.3.7 -> Gemini 3.8 Flash (M=3)` | 0.880 | 0.2340 ([0.0714, 0.3823]) | 0.1195 / 0.8063 | 0.5597 | 0.0972 | 0.0963 | 0.960 / 0.920 | 2.30 | 8577.8 | no |
| **P1:** `Cascade: Strands 2B v19 -> DecisionGemma` | 0.860 | 0.2585 ([0.1511, 0.3592]) | 0.1753 / 0.8376 | 0.4624 | 0.1866 | 0.0536 | 0.960 / 0.920 | 1.30 | 390.8 | no |
| **P1:** `Gemini 3.6 Flash (Dirichlet M=3)` | 0.920 | 0.2624 ([0.1222, 0.4240]) | 0.1637 / 0.8220 | 0.6289 | 0.2625 | 0.0410 | 0.960 / 0.920 | 1.00 | 62709.9 | no |
| **P1:** `Gemini 3.8 Flash (Dirichlet M=3)` | 0.900 | 0.2791 ([0.1373, 0.4371]) | 0.1779 / 0.8196 | 0.6678 | 0.2617 | 0.0485 | 0.960 / 0.920 | 3.94 | 56613.1 | no |
| **P1:** `Gemini 3.5 Flash Lite (Dirichlet M=3)` | 0.840 | 0.3404 ([0.2004, 0.4839]) | 0.1965 / 0.7769 | 0.7474 | 0.2058 | 0.0798 | 0.960 / 0.920 | 5.70 | 62790.4 | no |
| **P1:** `Strands Decider 2B v19` | 0.720 | 0.4249 ([0.3046, 0.5101]) | 0.2643 / 0.7603 | 0.8787 | 0.1524 | 0.2398 | 0.960 / 0.880 | 2.36 | 163.0 | yes |
| **P2:** `DecisionGemma` | 0.9694 | 0.0465 ([0.0106, 0.1005]) | 0.0137 / 0.4610 | 0.0845 | 0.0209 | 0.0020 | 0.9694 / 0.9694 | 1.00 | 175.8 | yes |
| **P2:** `dgem v0.3.1` | 0.9639 | 0.0565 ([0.0225, 0.1021]) | 0.0231 / 0.4604 | 0.1001 | 0.0204 | 0.0025 | 0.9639 / 0.9639 | 1.00 | 215.1 | no |
| **P2:** `Cascade: Classy v0.3.3 -> DecisionGemma` | 0.8833 | 0.1634 ([0.1187, 0.2273]) | 0.0084 / 0.3388 | 0.2453 | 0.0284 | 0.0245 | 0.9444 / 0.8972 | 1.04 | 866.8 | no |
| **P2:** `Cascade: Classier v0.2.0 -> DecisionGemma` | 0.8917 | 0.1720 ([0.1214, 0.2316]) | 0.0180 / 0.3398 | 0.2860 | 0.0270 | 0.0326 | 0.9444 / 0.9083 | 1.05 | 537.1 | no |
| **P2:** `Classier v0.2.0` | 0.7889 | 0.2986 ([0.2571, 0.3439]) | 0.0317 / 0.2269 | 0.4581 | 0.0273 | 0.0965 | 0.9417 / 0.9111 | 1.31 | 482.4 | no |
| **P2:** `Strands Decider 2B v19 (VectorScaled OOF)` | 0.8083 | 0.3031 ([0.2142, 0.3883]) | 0.0257 / 0.2164 | 0.4773 | 0.0834 | 0.1181 | 0.9500 / 0.9000 | 1.30 | 153.9 | yes |
| **P2:** `Strands Decider 2B v19` | 0.7583 | 0.3623 ([0.2680, 0.4420]) | 0.0847 / 0.2162 | 0.5445 | 0.0907 | 0.1489 | 0.9500 / 0.9000 | 1.44 | 153.9 | no |
| **P2:** `Classy v0.3.3 (VectorScaled OOF)` | 0.6444 | 0.4581 ([0.3861, 0.5180]) | 0.0175 / 0.0532 | 0.6520 | 0.0585 | 0.3308 | 0.9528 / 0.9000 | 1.69 | 746.2 | no |
| **P2:** `Classy v0.3.3` | 0.6444 | 0.4634 ([0.4107, 0.5071]) | 0.0164 / 0.0468 | 0.6563 | 0.0785 | 0.3297 | 0.9528 / 0.9000 | 1.69 | 746.2 | no |

Key findings from Panel 1 and Panel 2:
1. **Speed-Calibration Pareto Frontier**: On Panel 1, `DecisionGemma` reaches $0.920$ accuracy, $0.1796$ Brier score, and $0.0204$ AURC at $278.3\text{ ms}$ ($15.3\times$ faster than `Gemini 3.8 Flash` verbalized confidence at $4247.8\text{ ms}$). Out-of-fold temperature scaling on `Gemini 3.8 Flash` lowers ECE from $0.0484$ to $0.0231$ and Brier score from $0.1452$ to $0.1419$.
2. **Conformal Cascade Efficiency**: On Panel 1, `dgem v0.3.7 -> Gemini 3.8 Flash (M=3)` escalates only $14.0\%$ of queries where the Stage-1 RAPS set is non-singleton, lifting accuracy from $0.860$ to $0.880$ and cutting AURC to $0.0963$. On Panel 2, `Classier v0.2.0 -> DecisionGemma` escalates $31.1\%$ of queries to lift accuracy from $0.7889$ to $0.8917$ ($39$ wins vs $2$ losses, exact two-sided McNemar $p = 7.84 \times 10^{-10}$), while `Classy v0.3.3 -> DecisionGemma` lifts accuracy from $0.6444$ to $0.8833$ ($91$ wins vs $5$ losses, exact McNemar $p = 1.63 \times 10^{-21}$).
3. **Vector Scaling on Compact Probes**: Out-of-fold regularized vector scaling on `Strands Decider 2B v19` (`safety_xstest_360`) reduces Murphy reliability error (`REL`) from $0.0847$ to $0.0257$, lowers Brier score from $0.3623$ to $0.3031$, and lifts accuracy from $0.7583$ to $0.8083$ at $153.9\text{ ms}$.

### 5.2 Wide-Option Shortlists, Taxonomy Tries, and Induced Base-Rate Shift

Across wide-option intent and catalog routing benchmarks ($K = 60$ to $202$ classes):
- **Sibling Confusions in Flat Bracket Tournaments**: Error-taxonomy analysis on `MASSIVE` ($K = 60$) shows that $30.8\%$ of flat tournament bracket errors are same-domain sibling confusions caused by arbitrary Round-1 bracket partitioning. Two-level `[A-Z]` taxonomy tries (`beam_2pass` and `hesitation_hybrid` in `hierarchy.py`) group domain siblings into a single conditional pass and pack up to 8 brackets per pass (`2` passes on `MASSIVE` and `CLINC150`, `3` passes on 202-tool routing).
- **Log-Softmax Priors vs Raw Cosines (Offline 300-Trial Sweep vs Live `EXP-48` on Cloud Run `dgemma` `762e545`, `dgem` `d6ff165`)**:
  - *Live `EXP-48` (`ghchinoy/decision-lab#94`, `benchmarks/runs/20261023-exp48-shift-and-logsoftmax/RESULT.md`)*:
    - **Part 1 (`H43a` — Induced Label Shift)**: Fails `H43a` by the letter, though the binary benefit at `hesitant` ($w = 1.0$) is large (**Brier $-15\%$ to $-47\%$ at $5\%$ and $90\%$ prevalence across all $3/3$ binary suites**: `rag_dev` $-17.9\% / -14.6\%$, `jev` yes/no $-46.8\% / -16.0\%$, `gate_mixed_noul` $-45.0\% / -42.4\%$, accuracy gains up to $+5.5$ pts; `default` $w = 0.35$ passes on $1/3$). On multi-class `banking77` ($K = 30, n = 30$, Dirichlet(0.2)), $\Delta\text{Brier}$ is $-0.056 \pm 0.044$ ($w = 0.35$) and $-0.042 \pm 0.056$ ($w = 1.0$), within $2\text{ SE}$ (motivating our `max_effective_classes = 2` guard in `estimate_saerens_em_prior_logits`), and unpaired `jev` yes/no ($n = 74$) raw-entropy AUROC drops $-0.050$ while the other 3 suites improve by $+0.030$ to $+0.037$. Crucially, fused `normalized_entropy` drops $-0.255$ on `banking77` at $w = 1.0$, confirming callers must gate on `raw_normalized_entropy`.
    - **Part 2 (`H43b` — `NormalizeCosinePriorLogits` on Wide Choices)**: At the pre-registered $\tau = 0.07$ (`m20_lsm_hes`), `banking77` improves by $+3.3 \pm 2.4$ pts (below the $+5.0$ bar) and pooled improves by $+1.2 \pm 0.9$ pts (below $2\text{ SE}$), whereas raw cosines (`m20_raw_hes`) do nothing ($-0.1 \pm 0.5$ pts). The exploratory $\tau = 0.05$ arm (`m20_lsm005_hes`) achieves **$+6.7 \pm 3.2$ pts on `banking77`** ($80.0\% \to 86.7\%$) and **$+2.8 \pm 1.1$ pts pooled** ($> 2.5\text{ SE}$) with zero negative suites.
  - *Offline 300-Trial Local Simulation (`multimodal-decision-models`)*: Across 300-trial offline cosine-prior sweeps (`MASSIVE` $K = 59$, `banking77` $K = 77$, `CLINC150` $K = 151$), log-softmax cosine priors improve `banking77` by $+5.33$ pts ($0.800 \to 0.853$, $\text{SE } 1.19$) and pooled accuracy by $+1.83$ pts ($\text{SE } 0.37$) in $1$ pass per item.
- **Cold-Start Reranking Under $95\times$ to $125\times$ Popularity Bias (`EXP-47`)**: Using 6 candidates per query ($6$ score slots + $2$ QA slots = $8$ slots in `"lines"` format, $1$ forward pass) across 100 dev and 100 frozen page-disjoint test queries anchored on 200 distinct real documentation sections with `Page:` and `#` headers stripped, a 3-way query-token tie between `cold_match` and the two `popular_distractor` candidates, zero label-derived `candidate_tags`, and 1:1 `cold_negative` metadata parity (all metadata and lexical shortcut classifiers audited at $\le 0.56$), 100% label-free hesitant 2-pass intent priors significantly reduce popularity hijack rates ($p < 10^{-5}$).

### 5.3 Agent Trajectory Context Pruning (`8`-Slot MCKP)

We evaluate `prune_agent_context_blocks` (`mckp_hesitation_stub`) across 8 models on coding agent trajectories ($8$ blocks per trajectory, $55\%$ token budget cap, pinned user goal and active test failure blocks) against three baselines: recency `FIFO`, `TF-IDF` cosine selection, and greedy utility drop (`greedy_drop`).

| Pruning Policy | Mean Token Reduction (%) | Required-Block Recall (%) | Retained Task Utility | Exact DP Optimality Match | Paired Sign / McNemar $p$ vs `mckp_hesitation_stub` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`mckp_hesitation_stub` (Ours)** | **58.85%** | **100.00%** | **0.9412** | **202 / 202 (100%)** | Reference |
| `greedy_drop` (No Stub Action) | 56.10% | 81.33% | 0.7840 | 114 / 202 (56.4%) | $p = 7.45 \times 10^{-9}$ |
| `tfidf_cosine` | 57.40% | 62.00% | 0.6510 | 48 / 202 (23.8%) | $p = 1.12 \times 10^{-14}$ |
| `fifo_recency` | 58.20% | 45.33% | 0.5190 | 19 / 202 (9.4%) | $p = 2.84 \times 10^{-19}$ |

Allowing the intermediate `"stub"` action (retaining function signatures and error headers when Stage-1 hesitation $\tilde{H}_{\text{raw}} \ge 0.16$) enables `mckp_hesitation_stub` to preserve $100.00\%$ of required evidence blocks within a $58.85\%$ token reduction budget, matching brute-force $3^8 = 6,561$ enumeration on $202/202$ test instances.

### 5.4 Single-Pass RAG Reranking and Indirect Prompt-Injection Quarantine

We evaluate single-pass 6-chunk RAG reranking ($6$ relevance slots + $1$ injection-detection slot + $1$ answerability slot = $8$ slots) on a frozen $N = 220$ real-corpus benchmark (`rerank_suite.jsonl` + `docs_eval/dev_pool.jsonl`) across 6 payload families (`ignore_previous_override`, `fake_system_delimiter`, `authority_spoof_relevance`, `nested_xml_escape_attempt`, `unicode_homoglyph_directive`, and `benign_Format_lookalike`). Every `(slot, family)` cell ($6 \times 6 = 36$ cells) is balanced to exact $P(\text{injected} \mid \text{slot}, \text{family}) = 0.5000$, and all 16 lexical/structural shortcut classifiers fall strictly inside $[0.40, 0.60]$ (`0.4227` to `0.5500`).

| Model & Defense Arm | Clean `nDCG@3` | Adversarial `nDCG@3` | Attack Success Rate (`ASR@1`) | Injection Detection `Macro-F1` | False Positive Rate on Benign Lookalikes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Gemini 3.8 Flash` (Unfenced Raw Prompt) | 0.9310 | 0.6840 | 28.18% | 0.7120 | 14.55% |
| **`Gemini 3.8 Flash` (`<untrusted_data>` Quarantine)** | **0.9310** | **0.9295** | **0.00%** | **0.9727** | **0.91%** |
| `Gemini 3.5 Flash Lite` (Unfenced Raw Prompt) | 0.8940 | 0.6120 | 34.55% | 0.6680 | 18.18% |
| **`Gemini 3.5 Flash Lite` (`<untrusted_data>` Quarantine)** | **0.8940** | **0.8900** | **0.00%** | **0.9455** | **1.82%** |

Wrapping untrusted retrieval chunks via `sanitize_untrusted_context` (neutralizing spoofed closing tags, `SYSTEM:` headers, and role delimiters before enclosing text inside `<untrusted_data>` blocks) eliminates top-1 hijack (`ASR@1 = 0.00%`) across all 6 attack families while preserving clean `nDCG@3`.

### 5.5 Expected-Grade Autorater Tie-Breaking and Cascade Gate Calibration

When using 5-level Likert rubrics (`A..E` mapped to grades $g_k \in \{1.0, 0.75, 0.5, 0.25, 0.0\}$), argmax decoding collapses all items with the same modal token to identical ties. Computing continuous expected grade $\mathbb{E}[g \mid p] = \sum_{k=1}^5 p_k g_k$ resolves 5 discrete levels into 100 unique continuous scores across a 100-item pairwise tie-breaking benchmark, lifting pairwise ranking concordance from $0.540$ to $0.960$. Cross-validated threshold calibration on toxicity autorater cascades (`j4_vs_j1_toxicity_prompt`, $N = 76$) achieves statistical parity with the frontier verifier (exact McNemar $p = 0.1250 > 0.10$) while escalating only $3.95\%$ ($3/76$) of items.

## 6. Non-Exchangeable & Multi-Slot Conformal, `PriDe`, Submodular MCKP, Polytope Projection, and `MDM-Bench`

Single-token and multi-slot decision models face six structural limitations in standard split-conformal and heuristic cascade pipelines. In `v0.5.0`, we formalize and resolve all six limitations and evaluate them on **`MDM-Bench`** (`benchmarks/run_mdm_bench.py`, $N = 400$ real items across 8 practical multimodal, agentic, coding, relational, judge, and safety tracks):

1. **Non-Exchangeable Covariate Shift & Rare-Class Under-Coverage (`calibrate_mondrian_conformal_quantiles`, `weighted_conformal_prediction_set`)**:
   Marginal split conformal prediction guarantees $\mathbb{P}(Y \in \mathcal{C}_\alpha(X)) \ge 1 - \alpha$ only on average across the calibration distribution, allowing high coverage on frequent head classes (`98.75%`) to mask severe under-coverage on rare tail classes (`58.75%` at $\alpha = 0.10$ on `MDM-Bench`). Partitioning calibration quantiles by class or rarity group (`MondrianCalibrationResult`) restores rare-tail coverage to `88.75%` (overall `96.00%`), while Tibshirani et al. (2019) likelihood-ratio weighted conformal prediction (`weighted_conformal_prediction_set`) restores rare-tail coverage under covariate shift to `88.75%` (overall `97.75%`).
2. **Simultaneous Multi-Slot Family-Wise Error Rate Control (`calibrate_multislot_joint_conformal`, `multislot_joint_conformal_sets`)**:
   When a `/v1/systemone` or structured verifier pass scores $S = 4$ slots simultaneously, independent per-slot conformal calibration at $\alpha = 0.10$ compounds miscoverage across slots, dropping simultaneous 4-slot coverage to `68.00%`. Calibrating the joint max-score nonconformity statistic $M_i = \max_{s \in \{1..S\}} s(X_{i,s}, Y_{i,s})$ at order index $\lceil(n+1)(1-\alpha_{\text{family}})\rceil$ restores simultaneous 4-slot coverage to `90.25%` (`90.50%` under Bonferroni $\alpha / S$).
3. **Cyclic Option-Permutation Debiasing (`debiased_cyclic_permutation_probs`, `option_permutation_stability`)**:
   Single-token `[A-Z]` readouts exhibit positional bias toward early option letters, causing a `10.25%` top-1 decision flip rate when option order is shifted across real 4-permutation evaluation passes. Factoring out positional priors across cyclic option shifts (`PriDe`, Zheng et al., 2024) cuts option-order flip rate to `6.50%` (mean Jensen-Shannon divergence from `0.0525` bits to `0.0225` bits), lifts accuracy from `74.25%` to `75.25%`, and reduces multiclass Brier from `0.4286` to `0.4136`.
4. **Learn-then-Test (`LTT`) Joint Funnel Gate Calibration (`calibrate_ltt_funnel_gates`)**:
   Replacing heuristic cascade thresholds with Angelopoulos et al. (2021) Learn-then-Test Hoeffding-Bentkus upper confidence bounds over $\lambda = (\tau_H, w, T)$ guarantees finite-sample selective risk $\text{UCB}(\hat{\lambda}) = 0.0922 \le r_{\max} = 0.14$ at $\delta = 0.05$ (`0.0100` held-out selective risk) while fast-exiting `47.5%` of traffic at `128.4 ms` p50 latency. Across `MDM-Bench` ($N = 400$), the `LTT + RAPS Cascade (DecisionGemma-2B -> Gemini-3.8-Flash)` reaches `88.75%` accuracy, **`0.1598` Brier score**, **`0.3205` NLL**, **`0.0390` ECE**, and **`0.0376` AURC**, outperforming every standalone model on calibration and selective risk while cutting p50 latency by $2.7\times$ vs standalone `Gemini-3.8-Flash`.
5. **Submodular Facility-Location Redundancy-Penalized Context Pruning (`prune_agent_context_blocks(..., redundancy_penalty_weight=0.70)`)**:
   Discounting candidate block utilities by pairwise word-shingle Jaccard overlap against already-retained trajectory blocks prevents duplicate stack traces from crowding out orthogonal git-blame and config evidence, raising multi-facet root-cause recall across 50 real `EXP-49` trajectories from `66.67%` (additive MCKP) to `78.67%` at `61.17%` token reduction.
6. **Exact Transitive Equivalence Polytope Projection (`project_triangles_to_transitive_consistency`)**:
   Projecting noisy pairwise equivalence probabilities onto the transitive partition polytope $E_{uv} + E_{vw} - E_{uw} \le 1$ via exact branch-and-bound log-odds correlation clustering eliminates all `50 -> 0` violated 3-cycles across 50 real binary receipt entity-resolution graphs and lifts pairwise accuracy from `71.43%` to `91.43%`.

## 7. Frontier GenAI Extensions (`v0.7.0`): Online `BC-ACI`, Primal-Dual `ParetoBandit`, Calibrated `PRM` PUCT, and Supra-Bayesian Multi-Agent Pooling

To extend decision calibration beyond static exchangeability and single-step classification into non-stationary streams, dynamic multi-LLM routing, multi-step reasoning trees, and correlated multi-agent ensembles, `v0.7.0` (`benchmarks/run_frontier_genai_v070_bench.py`, $N = 400$ deterministic instances across 4 pillars) introduces four additional capabilities with full Python and Go parity (`pkg/mdm/frontier_v070.go`):

1. **Pillar 1 — Online Bias-Corrected Adaptive Conformal Inference (`OnlineBCACITracker`, `evaluate_online_bc_aci_stream`)**:
   Combines Gibbs & Candès (2021) online quantile tracking $q_{t+1} = q_t + \gamma_{\text{aci}}(\text{err}_t - \alpha)$ with online EMA label-shift bias correction (`estimate_saerens_em_prior_logits`). Under an abrupt step-40 regime shock ($N = 100$), `OnlineBCACITracker` tracks the target $90.0\%$ post-shift coverage (`0.9000` vs over-conservative static split `1.0000`) while improving Brier score and Murphy skill.
2. **Pillar 2 — Primal-Dual Budget-Paced Contextual & Dueling Bandit Router (`PrimalDualParetoBanditRouter`, `evaluate_pareto_bandit_stream`)**:
   Formulates multi-model routing (`fast` / `mid` / `pro` or `laya` / `deem` / `jev`) as an online Lagrangian primal-dual contextual bandit over query features $(\tilde{H}_{\text{raw}}, \Delta p_{1,2}, c_{\text{ctx}})$ with dual multipliers $(\lambda_{\text{cost}}, \lambda_{\text{lat}})$ updated via online subgradient ascent. Across $N = 100$ queries, `PrimalDualParetoBanditRouter` cuts mean per-query cost by **$-86.2\%$** vs static entropy cascades (`$0.00080` vs `$0.00582`) with zero budget violations (`0.0000`).
3. **Pillar 3 — Calibrated Process Reward Verifiers (`PRM`) & Conformal Branch-and-Bound PUCT Tree Search (`calibrate_prm_horizon_discount`, `run_calibrated_prm_puct_search`)**:
   Treats step-level reasoning verification as a calibrated multi-level rubric distribution with depth-dependent horizon discounting $\gamma_{\text{depth}}^d$ and conformal lower-bound pruning. Across $N = 100$ reasoning trees, calibrated conformal PUCT prunes **$-43.8\%$** of verifier node evaluations (`13.00 -> 7.31` nodes/tree) with **`0.0%` accuracy drop** (`100.0%` terminal solve rate).
4. **Pillar 4 — Covariance-Discounted Supra-Bayesian Multi-Agent Pooling (`calibrate_correlated_ensemble_weights`, `covariance_discounted_logit_pool`)**:
   Standard independent Product-of-Experts (`PoE`) double-counts redundant or cloned agents. By estimating the empirical logit-residual covariance matrix $\hat{\Sigma}$ on a calibration split and solving for inverse-covariance weights $w \propto (\hat{\Sigma} + \lambda I)^{-1}\mathbf{1}$ with Effective Ensemble Sample Size $N_{\text{eff}} = (\sum_m \Sigma^{-1}_{mm}) / (\mathbf{1}^\top \Sigma^{-1} \mathbf{1})$, `covariance_discounted_logit_pool` neutralizes a $4\times$ correlated clone attack ($N_{\text{eff}} = 1.34$ down from $M = 6$), lifting top-1 accuracy from **`65.00%` to `77.00%`** ($+12.0$ pts, exact McNemar $p = 0.0357$) and cutting NLL by **$-55.7\%$** (`1.3044 -> 0.5773`).

## 8. Six-Tier Verification and Reproducibility

Every module in `multimodal-decision-models` is verified across six testing tiers:
- **Tier 0 (Static AST and Hygiene Gates)**: 100% parameter and return type annotations verified via Python `ast` parsing (`tests/test_tier0_to_tier5_production_gates.py`), zero external runtime dependencies, and `gofmt`/`go vet` checks.
- **Tier 1 (Deterministic Mathematical Invariants)**: Exact simplex validation, log-softmax shift invariance, finite-sample conformal quantile order statistics, Murphy Brier identity ($\operatorname{REL} - \operatorname{RES} + \operatorname{UNC} + \operatorname{WBV}$), and Python-vs-Go numerical parity within $10^{-9}$.
- **Tier 2 (Wire-Format and Protocol Compliance)**: Live HTTP socket round-trips for `/v1/systemone`, `/v1/chat/completions`, Vertex `:generateContent` and `:predict`, and `stdio` JSON-RPC 2.0 MCP framing (`Content-Length` and newline-delimited).
- **Tier 3 (Probabilistic Calibration and Conformal Coverage)**: Out-of-fold temperature/vector scaling NLL reduction, empirical split conformal coverage $\ge 1 - \alpha$ across `lac`, `aps`, and `raps`, and Dirichlet epistemic variance monotonicity.
- **Tier 4 (Cross-Component Pipeline Integration)**: End-to-end Stage-2A embedding fast-exit, pre-fusion RAPS gating, dual-buffer PNG contact-sheet compilation, Layer-5 pack-size and visual-energy vetoes, and SHA-256 JSONL decision caching.
- **Tier 5 (Adversarial Stress and Quarantine Gates)**: 6-family prompt-injection quarantine, Unicode control-character stripping, truncated PNG/CRC32 corruption rejection, SLA timeout quarantine (`RETRY_PENDING`), and $3^8$ brute-force MCKP optimality checks.

Reproduce all tests and empirical benchmark tables with the following commands:

```bash
# Run Python unit and 6-tier production gate suite
PYTHONPATH=src python3 -m unittest discover -s tests -v

# Run Go reference package tests with race detector and fuzz smoke
go vet ./pkg/...
go test -race -v ./pkg/...
go test -fuzz=FuzzNormalizeCosinePriorAndFusion -fuzztime=2s ./pkg/mdm

# Reproduce the 10-model calibration and conformal benchmark report
PYTHONPATH=src python3 benchmarks/run_multimodel_benchmark.py

# Reproduce the 400-item 8-track MDM-Bench report and 8 practical workflow recipes
PYTHONPATH=src python3 benchmarks/run_mdm_bench.py
PYTHONPATH=src python3 examples/how_i_ai_workflows.py
```

## References

1. Vovk, V., Gammerman, A., and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
2. Sadinle, M., Lei, J., and Wasserman, L. (2019). Least Ambiguous Set-Valued Classifiers with Bounded Error Levels. *Journal of the American Statistical Association*, 114(525), 223-234.
3. Romano, Y., Sesia, M., and Candes, E. J. (2020). Classification with Valid and Adaptive Coverage. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 3581-3591.
4. Angelopoulos, A. N., Bates, S., Malik, J., and Jordan, M. I. (2021). Uncertainty Sets for Image Classifiers using Conformal Prediction. *International Conference on Learning Representations (ICLR)*.
5. Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *International Conference on Machine Learning (ICML)*, 1321-1330.
6. Murphy, A. H. (1973). A New Vector Partition of the Probability Score. *Journal of Applied Meteorology*, 12(4), 595-600.
7. Finlayson, G. D., and Trezzi, E. (2004). Shades of Gray and Colour Constancy. *IS&T/SID Twelfth Color Imaging Conference*, 37-41.
8. Kellerer, H., Pferschy, U., and Pisinger, D. (2004). *Knapsack Problems* (Chapter 11: The Multiple-Choice Knapsack Problem). Springer.
9. Tibshirani, R. J., Foygel Barber, R., Candes, E. J., and Ramdas, A. (2019). Conformal Prediction Under Covariate Shift. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
10. Angelopoulos, A. N., Bates, S., Candes, E. J., Jordan, M. I., and Lei, L. (2021). Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. *arXiv:2110.01052*.
11. Zheng, C., Zhou, H., Meng, F., Zhou, J., and Huang, M. (2024). Large Language Models Are Not Robust Multiple Choice Selectors. *International Conference on Learning Representations (ICLR)*.
12. Bansal, N., Blum, A., and Chawla, S. (2004). Correlation Clustering. *Machine Learning*, 56, 89-113.
