# Contributing to QAI-Chain

Thanks for your interest in improving QAI-Chain.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local Quality Gates

Before opening a pull request, run:

```bash
PYTHONPATH=. .venv/bin/python scripts/healthcheck.py
PYTHONPATH=. .venv/bin/python -m pytest -q
.venv/bin/pip check
```

If your change touches experiment artifacts or paper tables, also run:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_reproducibility_harness.py
PYTHONPATH=. .venv/bin/python scripts/generate_paper_tables.py
```

## Branch and Commit Guidance

- Keep pull requests focused and small.
- Use clear commit messages with intent and scope.
- Do not mix unrelated refactors with functional changes.

## Pull Request Checklist

- [ ] Code builds and tests pass locally.
- [ ] New behavior includes tests (or justification if not testable).
- [ ] Documentation was updated for user-facing changes.
- [ ] Generated artifacts are reproducible via scripts.

## Reporting Issues

Please include:

- Expected behavior
- Actual behavior
- Steps to reproduce
- Environment details (OS, Python version, and relevant dependencies)
