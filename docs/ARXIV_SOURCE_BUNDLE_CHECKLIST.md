# arXiv Source Bundle Checklist

## Required Package Contents

- [x] Main manuscript source (`paper/main.tex` or target variant)
- [x] Section files under `paper/sections/`
- [x] Bibliography file (`paper/references.bib`)
- [x] Generated table files under `paper/tables/`

## Artifact Traceability (Recommended)

- [x] Benchmark artifact JSON
- [x] Research results JSON
- [x] Statistical analysis JSON
- [x] Human-readable reports under `docs/`

## Cleanliness Requirements

- [ ] No absolute machine-specific paths in TeX files
- [ ] No missing `\input{}` or bibliography references
- [ ] Build succeeds with standard TeX toolchain
- [ ] Bundle excludes temporary TeX files (`.aux`, `.log`, `.out`, ...)

## Commands

Run the full camera-ready artifact flow:

```bash
python scripts/build_camera_ready.py
```

Create arXiv source tarball:

```bash
python scripts/create_arxiv_bundle.py
```

Default bundle output:

- `dist/arxiv_source.tar.gz`