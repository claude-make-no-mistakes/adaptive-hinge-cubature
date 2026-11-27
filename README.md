# Adaptive Hinge Cubature

Adaptive multidimensional numerical integration over $[0,1]^d$ ($2 \leq d \leq 8$) using line-sample planar kink recovery, exact integer corner-power hinge integration, and an embedded Genz–Malik residual rule.

- [Paper (PDF)](output/pdf/latex/cubature.pdf)
- [LaTeX source](manuscripts/latex/cubature.tex)
- [Markdown draft](manuscripts/cubature.md)

## Overview

When an integrand contains a planar derivative discontinuity $f(x) = q(x) + a(w^\top x - b)_+$, standard adaptive cubature rules repeatedly subdivide boxes along the hyperplane. This repository implements:

1. **Line-sample kink recovery (`blind-rays` and `blind-rays-shared`)**: Extracts slope jumps and knot locations from second forward differences along rays through the box center, resolves relative orientation across rays, and validates the recovered hinge on held-out points.
2. **Exact integer analytic integration (`Hinge.integral`)**: Evaluates the truncated-power inclusion–exclusion formula in exact integer arithmetic on binary64 dyadic numerators before a single final IEEE-754 rounding.
3. **Adaptive residual cubature (`integrate`)**: Integrates the residual callback with a degree-7/5 Genz–Malik rule and falls back to active-set fitting or directional bisection when the ray model does not validate.

Across 64 independently seeded quadratic-plus-hinge integrands in dimensions 2, 4, 6, and 8 (128 conditions), the ray pilot verified all 128/128 conditions and achieved a median $2.22\times$ speedup (95% bootstrap interval $2.14\text{--}2.39\times$) on the 124 conditions jointly verified with the cached active-set control. Sharing the common center sample across rays saved 5.80% of evaluations on the quadratic-hinge family (20,480 evaluations across all 640 conditions) with identical estimates.

## Repository Layout

- `workhorses/cubature.py` — Core cubature, hinge recovery, and exact integer kernel implementation.
- `tests/test_cubature.py` — Unit and regression test suite covering recovery, exact integration, signs, reversed boxes, and budgets.
- `benchmarks/` — Reproducible benchmark runners (`shared_ray_counts.py`, `paper_cubature.py`, `followup_cubature.py`, `cubature.py`) and optional Cuhre build helper (`build.py`).
- `manuscripts/` — LaTeX source (`latex/cubature.tex`), Markdown draft (`cubature.md`), archived run journal (`runs/cubature.jsonl`), and summary receipts (`summary.json`).
- `doubling/`, `followup/`, `paper/` — Archived historical measurement journals and study notes.

## Reproduce

Requires Python 3.12+ with `numpy`, `scipy`, and `mpmath`.

```sh
python3 -m pip install -r requirements.txt
python3 -B -m unittest discover -s tests -v
python3 -B -m benchmarks.shared_ray_counts --output .research/replay/cubature.jsonl
```

For single-threaded timing comparisons, set BLAS/OpenMP/MKL thread limits to `1` and use a fresh output path. Building the optional external Cuhre baseline (`python3 -B -m benchmarks.build cuba`) requires a C compiler. Archived timing ratios and coverage totals are recorded in [doubling/summary.json](doubling/summary.json) and [manuscripts/summary.json](manuscripts/summary.json).
