## Summary

Describe the problem this PR solves and the behavioral or mathematical contract it adds or updates.

## Verification Checklist

- [ ] Python unit tests pass (`PYTHONPATH=src python -m unittest discover -s tests -v`)
- [ ] Go unit tests, race detector, and `gofmt` pass (`go vet ./pkg/... && go test -race ./pkg/...`)
- [ ] Markdown links and `#fragment` heading anchors pass (`python scripts/docs_link_check.py`)
- [ ] Cross-language invariants (conformal quantiles, cosine priors, PoE fusion) maintain parity across Python, Go, and TypeScript when applicable
- [ ] Zero internal hostnames, workspace paths, or real credentials included
