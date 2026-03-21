# QAI-Chain Paper Package

This directory contains a conference-style LaTeX manuscript scaffold.

## Entry Files

- `main.tex`: standard manuscript
- `main_neurips_style.tex`: non-official NeurIPS-like draft layout
- `main_iclr_style.tex`: non-official ICLR-like draft layout
- `main_ieee_style.tex`: non-official IEEE-like draft layout

## Regenerate Tables From Artifacts

From project root:

```bash
python experiments/run_research_suite.py
python scripts/generate_benchmark_report.py
python scripts/generate_paper_tables.py
```

This updates:

- `paper/tables/main_comparison.tex`
- `paper/tables/ablation.tex`
- `paper/tables/benchmark.tex`

## Build PDF

From `paper/`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If your environment supports latexmk:

```bash
latexmk -pdf main.tex
```

## One-Command Camera-Ready Pipeline

From project root:

```bash
python scripts/build_camera_ready.py
```

To compile the NeurIPS-like variant:

```bash
python scripts/build_camera_ready.py --paper-main main_neurips_style.tex
```

## arXiv Source Bundle

From project root:

```bash
python scripts/create_arxiv_bundle.py
```

Default output:

- `dist/arxiv_source.tar.gz`

## Venue-Specific Bundles

From project root:

```bash
python scripts/create_venue_bundle.py --venue neurips
python scripts/create_venue_bundle.py --venue iclr
python scripts/create_venue_bundle.py --venue ieee
python scripts/create_venue_bundle.py --venue all
```

This generates date-stamped bundles in `dist/`.