# arXiv Source Bundle Checklist

## Required Package Contents

- [x] Main manuscript source (`paper/main.tex`)
- [x] **`00README.yaml`** at repository root (compiler + top-level TeX for `paper/main.tex`) — see [arXiv `00README`](https://info.arxiv.org/help/00README.html)
- [x] Section files under `paper/sections/`
- [x] Bibliography file (`paper/references.bib`)
- [x] **Generated `paper/main.bbl`** (recommended: run `bibtex main` from `paper/` before bundling so arXiv uses your exact bibliography output)
- [x] Generated table files under `paper/tables/`
- [x] Figures under `paper/figures/` (PDF/PNG as referenced)

## Artifact Traceability (Recommended)

- [x] Benchmark artifact JSON
- [x] Research results JSON
- [x] Statistical analysis JSON
- [x] Human-readable reports under `docs/`

## Cleanliness Requirements

- [ ] No absolute machine-specific paths in TeX files
- [ ] No missing `\input{}` or bibliography references
- [ ] Build succeeds with standard TeX toolchain (preview arXiv’s compiled PDF after upload)
- [ ] Bundle excludes temporary TeX files (`.aux`, `.log`, `.out`, `.blg`, …) and **`paper/main.pdf`** (arXiv rebuilds from TeX), but **includes** `paper/main.bbl`
- [ ] All file names match [arXiv allowed characters](https://info.arxiv.org/help/submit/index.html): `a-z A-Z 0-9 _ + - . , =` (the bundle script **aborts** if any included path violates this)

## Commands

Run the full camera-ready artifact flow:

```bash
python scripts/build_camera_ready.py
```

From `paper/`, regenerate bibliography if needed:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Create arXiv source tarball:

```bash
python scripts/create_arxiv_bundle.py
```

Default bundle output:

- `dist/arxiv_source.tar.gz`

Upload that archive on the [arXiv submission flow](https://info.arxiv.org/help/submit/index.html), confirm **pdflatex** and **top-level** `paper/main.tex`, then **preview** the server-built PDF before final submit.
