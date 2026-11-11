# Changelog

All notable changes to `multimodal-decision-models` are documented in this file.

## [1.0.0] - 2026-11-10

### Added
- **Stateless `HMAC-SHA256` Cross-Language Conformal Determinism (`conformal.py`, `pkg/mdm/conformal.go`, `sdk/ts/src/calibration.ts`)**: Order-independent, lock-free `hmac_uint64` / `HmacUint64` / `hmacUint64` and `hmac_uniform_float64` tie-breaking across Python, Go, and TypeScript (`14094744155160802019` canonical parity vector).
- **Zero-Trust HTTP Transport & Single-Route Vertex `:rawPredict` Auto-Dispatch (`backends.py`, `sdk/ts/src/client.ts`)**: Permanent credential header stripping (`Authorization`, `X-DGEM-Key`, `X-DGEM-API-Key`, `X-Goog-Api-Key`) across cross-origin HTTP redirects (including `307`/`308` bounce-back hops), `20.0 s` TCP+TLS connect timeout cap, CR/LF header injection rejection, coroutine rejection in synchronous factories, and `POST {"health": true}` health probing for `:rawPredict` URLs.
- **Google ADK & LangGraph Adapters (`adapters.py`)**: `ActionStakesPolicy`, `evaluate_adk_tool_guard`, `create_adk_before_tool_guard`, `create_langgraph_decision_node`, and `neutralize_untrusted_delimiters` enforcing asymmetric $V_{\text{irrev}}$ vs. $V_{\text{rev}}$ confidence gates and 2-slot joint conformal escalation.
- **Zero-Runtime-Dependency TypeScript SDK (`sdk/ts`, `@mdm/client` `1.0.0`)**: `MDMClient`, split and multi-slot joint conformal calibration, temperature scaling, Bayesian `PoE` fusion, and Genkit JS guards (`createGenkitToolGuard`, `createGenkitDecisionStep`).
- **Empirical Wide-Option Routing Ladder & 8 Canonical Policy Templates (`hierarchy.py`, `templates/`)**: `recommend_wide_routing_strategy` plus 8 `.json.tmpl` + `.examples.json` policy fixtures.
- **Capability-Aware MCP Schemas & Declarative GCP Terraform (`mcp_server.py`, `deploy/terraform/`)**: `build_mcp_tool_schemas` with fail-closed `tools/call` capability gating, digest-pinned Cloud Run + optional Vertex AI Terraform root module, 6 Cloud Logging funnel metrics + Monitoring dashboard, 40-char SHA-pinned GitHub Actions + `CodeQL` workflows, and progressive 8-page developer documentation hub (`docs/index.md`).

## [0.5.0] - 2026-10-25

### Added
- **Non-Exchangeable, Mondrian, and Multi-Slot Joint Conformal Prediction (`conformal.py`, `pkg/mdm/conformal.go`)**: Class- and group-conditional Mondrian conformal quantiles (`calibrate_mondrian_conformal_quantiles`, `mondrian_conformal_prediction_set`), Tibshirani likelihood-ratio weighted conformal prediction under covariate shift (`weighted_conformal_prediction_set`), and simultaneous $S$-slot family-wise joint conformal calibration (`calibrate_multislot_joint_conformal`, `multislot_joint_conformal_sets`).
- **Cyclic Permutation Debiasing (`PriDe`) and Learn-then-Test (`LTT`) Funnel Calibration (`calibration.py`, `eval_suite.py`)**: Cyclic option-shift positional bias decomposition (`debiased_cyclic_permutation_probs`), option-order permutation flip-rate and Jensen-Shannon stability metrics (`option_permutation_stability`), and finite-sample Hoeffding-Bentkus risk-controlling joint gate calibration (`calibrate_ltt_funnel_gates`).
- **Submodular Redundancy-Penalized Context Pruning and Exact Transitive Polytope Projection (`context_pruner.py`, `consistency.py`)**: Facility-location word-shingle Jaccard redundancy discount (`redundancy_penalty_weight` in `prune_agent_context_blocks`) and exact branch-and-bound log-odds correlation clustering (`project_triangles_to_transitive_consistency`) eliminating 3-cycle equivalence contradictions.
- **`MDM-Bench` 8-Track Practical Benchmark (`N = 400`) and "How I AI" Workflows Cookbook (`benchmarks/run_mdm_bench.py`, `examples/how_i_ai_workflows.py`, `docs/practical_use_cases.md`)**: Reproducible 400-item benchmark across 8 practical multimodal, agentic, coding, relational, judge, and safety tracks plus 8 runnable end-to-end recipes.

## [0.4.0] - 2026-10-25

### Added
- **Split Conformal and RAPS Calibration (`conformal.py`, `pkg/mdm/conformal.go`)**: Finite-sample nonconformity quantile calibration (`calibrate_conformal_quantile`) at order index $\lceil(n+1)(1-\alpha)\rceil$ and prediction set generation (`conformal_prediction_set`) across Least Ambiguous Set-Valued Classifiers (`lac`), Adaptive Prediction Sets (`aps`), and Regularized Adaptive Prediction Sets (`raps`) with pre-fusion cascade escalation when $|\mathcal{C}_\alpha(x)| \ne 1$ or an abstain option enters the set.
- **Fourth Inference Paradigm (`BlackBoxSampledBackend` in `backends.py`)**: Logprob-free consensus adapter aggregating $M$ discrete completion draws or verbalized vote maps under a symmetric Dirichlet prior ($\alpha_0 = 0.5$) with exact analytic epistemic variance $\operatorname{Var}(\theta_k \mid c) = p_k(1 - p_k)/(A_0 + 1)$ and restricted enum grammar constraints.
- **Post-Hoc Temperature and Vector Scaling (`calibration.py`)**: $L_2$-regularized scalar temperature scaling (`fit_temperature_scaling`) via 81-point grid search plus damped Newton-Raphson refinement, and class-wise diagonal vector scaling (`fit_vector_scaling`) shrunk toward scalar temperature $T_0$.
- **Dual-Buffer Visual Contact-Sheet Compiler (`visual_compiler.py`)**: Pure-stdlib PNG (filters `0..4`, RGB/RGBA/Gray, IEEE CRC32) and PPM (`P6`/`P3`) codecs; Stage-2A unsharpened Minkowski $p=6$ Shades-of-Gray white balance, inner-80% CIE $L^*a^*b^*$ $\Delta E$ similarity, and residual MLP adapter `apply_residual_mlp_adapter`; Stage-2B contact sheets (`compile_dual_buffer_contact_sheet`) with automatic inter-candidate pixel variance saliency bands (`extract_inter_candidate_saliency_band`), `2x` bottom-30% pack-weight zoom strips, and `1.15x` cylindrical unwarping.
- **Multimodal Embedding Index (`embedding_index.py`)**: `VertexMultimodalEmbeddingAdapter` wire payload builder and parser for `gemini-embedding-2` plus JSONL-backed `MultimodalEmbeddingIndex` combining cosine or `CSLS` similarity, CIE $L^*a^*b^*$ color similarity, Lidstone container aspect priors, and Stage-2A fast-exit gating.
- **8-Slot Trajectory Context Pruner (`context_pruner.py`, `pkg/mdm/conformal.go`)**: Single-pass 8-slot context pruning policy builder (`build_context_pruning_questions`) and exact dominance-pruned Multiple-Choice Knapsack optimizer (`prune_agent_context_blocks`) over `{"full", "stub", "drop"}` with pinned-block protection.
- **MCP `stdio` Server (`mcp_server.py`)**: Zero-dependency JSON-RPC 2.0 server (`dgem-catalog`, CLI entry point `dgem-catalog-mcp`) supporting newline-delimited and `Content-Length` framing for `shortlist_candidates`, `evaluate_conformal_decision`, `prune_context_blocks`, and `compile_contact_sheet`.
- **10-Model Empirical Calibration and Conformal Benchmark (`benchmarks/`)**: Reproducible benchmark harness (`benchmarks/run_multimodel_benchmark.py`) and frozen evaluation panels (`N = 50` multi-task calibration and `N = 360` XSTest safety refusal boundary) with 5-fold out-of-fold scaling, RAPS cascade escalation analysis, 95% cluster bootstrap confidence intervals, and exact two-sided McNemar tests.
- **Research Paper and Documentation (`docs/research_paper.md`, `docs/research_summary.md`, `CITATION.cff`, `LICENSE`, `CONTRIBUTING.md`)**: Complete open-source documentation and research manuscript.

## [0.3.0] - 2026-10-24

### Added
- **Total Contracts and Closed State Machine (`contracts.py`)**: Closed terminal decision states (`STAGE1_RESOLVED`, `STAGE2_VERIFIED`, `STAGE2A_FAST_EXIT`, `GUARD_OVERRIDDEN`, `ABSTAIN_OOS`, `RETRY_PENDING`, `REJECTED_CONTRACT_VIOLATION`), strict probability simplex validation (`validate_probability_simplex`), `<untrusted_data>` prompt-injection fencing (`sanitize_untrusted_context`), and SHA-256 JSONL caching (`ContentAddressedDecisionCache`).
- **Model-Agnostic Backend Adapters (`backends.py`)**: `DgemSystemOneBackend` (`/v1/systemone`), `OpenAILogprobBackend` (`/v1/chat/completions`), `VertexGeminiBackend` (Evidence-First `propertyOrdering` schema), and `CascadeDecisionOrchestrator` with PT180S SLA timeout quarantine.
- **Six-Tier Evaluation Suite (`eval_suite.py`)**: Multiclass Brier score, continuous within-bin Murphy (`REL`, `RES`, `UNC`, `WBV`) decomposition, NLL, `ECE`, quantile-binned `ACE`, Macro-F1, `nDCG@k`, `AUROC`, selective `AURC`, exact paired McNemar test, Rogan-Gladen prevalence adjustment, Cohen's $\kappa$, and cluster bootstrap CIs.
- **Go Reference Package (`pkg/mdm/`)**: Zero-dependency Go 1.22 package with race-detector and fuzz test suites.

## [0.2.1] - 2026-10-23

### Added
- **Temperature-Scaled Log-Softmax Cosine Priors (`shortlist.py`)**: Converts raw retrieval cosines onto a calibrated log-probability scale ($\tau = 0.07$) before Product-of-Experts fusion, with opt-in `pairwise_diff` sibling cue extraction.
- **Two-Level Semantic Taxonomy Tries (`hierarchy.py`)**: `TaxonomyTrie` and `HierarchicalDecisionRouter` supporting `mtp_trie`, `beam_2pass`, and `hesitation_hybrid` routing up to $K = 625$ classes with `[A-Z]` slot bounds.
- **Continuous Triangle Consistency and Set-of-Marks Helpers (`consistency.py`, `som_navigator.py`)**: Continuous 3-cycle contradiction scoring (`score_probabilistic_triangles`), canonical pack-quantity guards (`validate_pack_quantity_consistency`), unimodal-safe mark centroids (`safe_expected_mark_center`), and nested containment (`select_containing_or_nearest_mark`).
