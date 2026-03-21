# ICLR Submission Checklist

## Content Readiness

- [x] Verify problem framing and hypothesis clarity in introduction
- [x] Ensure baseline and ablation rationale is explicit
- [x] Confirm statistical analysis section is aligned with tables

## Format and Policy

- [x] Switch to official ICLR template before submission
- [x] Preserve anonymity and remove identifying artifacts
- [x] Confirm reproducibility and ethics statements are present

## Reproducibility

- [x] Re-run publication-scale experiments
- [x] Re-run statistical analysis and table generation scripts
- [x] Ensure manuscript references latest artifact timestamps

## Commands

```bash
python scripts/create_venue_bundle.py --venue iclr
```