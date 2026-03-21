# NeurIPS Submission Checklist

## Content Readiness

- [x] Confirm novelty statement is precise and non-overstated
- [x] Ensure experimental claims match artifacted results exactly
- [x] Include limitations and broader impacts with concrete caveats

## Format and Policy

- [x] Switch to official NeurIPS template before submission
- [x] Follow anonymous submission policy (remove identifying metadata)
- [x] Verify reference format and supplementary policy compliance

## Reproducibility

- [x] Run publication suite and regenerate all artifacts
- [x] Regenerate LaTeX tables from latest JSON outputs
- [x] Ensure all numbers in paper match generated tables

## Commands

```bash
python scripts/create_venue_bundle.py --venue neurips
```