# OmniSafe Integration Notes

## Current Environment Status

The active workspace uses Python 3.12. In this environment, OmniSafe installation fails because current OmniSafe package metadata pins `pandas==2.0.3`, which is not available as a binary wheel for Python 3.12 and fails during build requirements.

Observed failure:
- `pip install omnisafe`
- `pip install git+https://github.com/PKU-Alignment/omnisafe.git`
- both fail while resolving/building `pandas==2.0.3` with `ModuleNotFoundError: No module named 'pkg_resources'` in the build subprocess.

## Practical Path Used In This Repo

To keep experiments moving in this environment, we added constrained-control compatible baseline implementations in:
- `experiments/run_constrained_baseline_suite.py`

Methods included:
- `cpo` (CPO-style projection)
- `p3o` (projection-style clipped trust region)
- `ppo_lagrangian` (adaptive violation-penalty schedule)
- `robust_qgate_shield`, `heuristic`, `random`

These outputs are shape-compatible with paper artifact generation.

## Exact OmniSafe Path (Py3.11)

For strict OmniSafe baselines, run in a Python 3.11 environment:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -U pip setuptools wheel
pip install omnisafe
```

Then add an OmniSafe-backed runner that writes `experiments/results/constrained_baseline_suite.json` with the same schema.

## Why This Is Acceptable For Now

- The repository remains executable in the current environment.
- Baseline interfaces are already wired into the manuscript artifact pipeline.
- Replacing the baseline engine with OmniSafe under Py3.11 is a drop-in backend swap, not a paper-structure rewrite.
