# Contributing to Multimodal Decision Models

Contributions that improve probability calibration, set-valued uncertainty quantification, multimodal visual preprocessing, or model-agnostic backend adapters are welcome.

## Design Principles

1. **Zero External Runtime Dependencies**: Both the Python library (`src/multimodal_decision_models/`) and the Go reference package (`pkg/mdm/`) run strictly on the standard library (`python >= 3.10`, `go >= 1.22`). Do not add third-party runtime packages to `pyproject.toml` or `go.mod`.
2. **Total Contracts and Zero Silent Fallbacks**: Every public function validates its inputs and raises `ContractViolationError` (or returns a typed error in Go) on NaN/Inf logits, unnormalized probability vectors, empty candidate sets, or schema mismatches. Never return uniform dummy probabilities or fabricated fallback completions when an upstream backend fails.
3. **100% AST Type Annotations**: Every function and method in `src/multimodal_decision_models/*.py` must declare explicit parameter and return type annotations. The automated gate in `tests/test_tier0_to_tier5_production_gates.py` parses the Python AST on every test run and fails on any unannotated signature.
4. **Python and Go Mathematical Parity**: Core calibration, conformal prediction, context-pruning, shortlist, taxonomy-trie, Set-of-Marks, and consistency routines must agree within $10^{-9}$ across Python (`src/multimodal_decision_models/`) and Go (`pkg/mdm/`).

## Verification Commands

Run the full verification suite before opening a pull request:

```bash
# 1. Python unit and 6-tier production gate suite
PYTHONPATH=src python3 -m unittest discover -s tests -v

# 2. Go formatting, static analysis, race-detector tests, and fuzz smoke
gofmt -l ./pkg
go vet ./pkg/...
go test -race -v ./pkg/...
go test -fuzz=FuzzNormalizeCosinePriorAndFusion -fuzztime=2s ./pkg/mdm

# 3. Reproducible 10-model calibration and conformal benchmark
PYTHONPATH=src python3 benchmarks/run_multimodel_benchmark.py
```

## Documentation Style

Write documentation and commit messages in direct, factual technical prose:
- State claims and empirical measurements with exact metrics, sample sizes, and confidence intervals.
- Use active voice and short descriptive section labels.
- Avoid promotional adjectives, vague attributions, em dashes, en dashes, and horizontal divider lines.

## License

By contributing to `multimodal-decision-models`, you agree that your contributions will be licensed under the [Apache License, Version 2.0](LICENSE).
