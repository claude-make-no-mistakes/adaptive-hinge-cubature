"""Fresh blocked comparison of scalar cubature at verified tolerances."""

import argparse
from dataclasses import asdict
import hashlib
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.stats import qmc, t

from benchmarks.common import BIN, schedule
from benchmarks.cubature import cuhre_baseline, make_case, stock_scipy
from benchmarks.fetch import sha256
from benchmarks.paper_common import Journal, PAPER, SEED, finite
from workhorses.cubature import Result, _pilot_design, integrate


FAMILIES = {
    "quadratic-kink": 12, "mixed-kink": 12, "two-kinks": 8, "curved": 8,
    "smooth-gaussian": 8, "oscillatory": 8, "product-peak": 8, "narrow": 8,
}
DIMENSIONS = (2, 4, 6, 8)
TOLERANCES = (3e-5, 3e-6)
BUDGET = 65_536
METHODS = ("directional", "blind", "blind-cached", "scipy", "cuhre", "sobol")
SCRAMBLES = 8
LEVELS = (8, 9, 10, 11, 12, 13)
STUDENT = float(t.ppf(1 - 0.05 / (2 * len(LEVELS)), SCRAMBLES - 1))


def randomized_sobol(case, atol, budget, seed):
    generators = [qmc.Sobol(case.dimension, scramble=True, rng=np.random.default_rng(s))
                  for s in np.random.SeedSequence(seed).spawn(SCRAMBLES)]
    sums = np.zeros(SCRAMBLES)
    consumed = 0
    batches = 0
    for exponent in LEVELS:
        size = 1 << exponent
        if size * SCRAMBLES > budget:
            break
        for i, generator in enumerate(generators):
            sums[i] += case.f(generator.random(size - consumed)).sum()
            batches += 1
        consumed = size
        means = sums / size
        estimate = float(means.mean())
        width = float(STUDENT * means.std(ddof=1) / np.sqrt(SCRAMBLES))
        if width <= atol:
            return Result(estimate, width, consumed * SCRAMBLES, batches, 0, "converged")
    return Result(estimate, width, consumed * SCRAMBLES, batches, 0, "budget")


def paper_cases(probe=False):
    for i, (family, count) in enumerate(FAMILIES.items()):
        for dimension, draw in product(DIMENSIONS, range(count)):
            if probe and (i > 2 or draw > 0):
                continue
            seed = SEED + i * 10_000 + dimension * 100 + draw + (50_000_000 if probe else 0)
            yield make_case(dimension, seed, family), seed, draw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PAPER / "runs" / "cubature.jsonl")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    configurations = [list((case.name, seed, draw, case.truth)) for case, seed, draw in paper_cases(args.probe)]
    cuhre = cuhre_baseline()
    protocol = {
        "study": "cubature-v2", "seed": SEED, "probe": args.probe, "methods": list(METHODS),
        "draws_sha256": hashlib.sha256(repr(configurations).encode()).hexdigest(),
        "cuhre_sha256": sha256(BIN / "cuhre.dll"), "numpy": np.__version__, "scipy": scipy.__version__,
        "absolute_tolerances": list(TOLERANCES), "repeats": 3, "budget": BUDGET,
        "scrambles": SCRAMBLES, "sobol_levels": list(LEVELS), "sobol_critical_value": STUDENT,
    }
    journal = Journal(args.output, protocol)

    def measure(case, method, atol, seed):
        if method == "scipy":
            return stock_scipy(case, atol, BUDGET)
        if method == "cuhre":
            return cuhre(case, atol, BUDGET)
        if method == "sobol":
            return randomized_sobol(case, atol, BUDGET, seed + 10_000)
        d = case.dimension
        return integrate(case.f, np.zeros(d), np.ones(d), atol=atol, max_evaluations=BUDGET,
                         mode=method, hinge=case.hinge if method == "oracle" else None)

    # Every cached run uses the warmed, value-independent pilot geometry.
    if "pilot-preparation" not in journal.completed:
        timings = []
        for d in DIMENSIONS:
            for repeat in range(7):
                _pilot_design.cache_clear()
                start = perf_counter()
                design = _pilot_design(d, 256, 64, True)
                elapsed = perf_counter() - start
                timings.append({"dimension": d, "repeat": repeat, "seconds": elapsed,
                                "array_bytes": sum(v.nbytes for v in vars(design).values())})
        journal.add("pilot-preparation", attributes={"kind": "setup"}, attempts=timings)
    for d in DIMENSIONS:
        warm = make_case(d, SEED + 412, "quadratic-kink")
        for method in METHODS:
            measure(warm, method, 4e-3, SEED + 413)
        measure(warm, "oracle", 4e-3, SEED + 413)
    for case, seed, draw in paper_cases(args.probe):
        if case.name in journal.completed:
            continue
        methods = METHODS + (("oracle",) if case.hinge is not None else ())
        rows = []
        for atol in TOLERANCES:
            for repeat, method in schedule(methods, seed=seed + int(atol * 10**7), repeats=3):
                start = perf_counter()
                result = measure(case, method, atol, seed)
                seconds = perf_counter() - start
                err = abs(result.estimate - case.truth)
                data = asdict(result)
                data["estimate"], data["error"] = finite(result.estimate), finite(result.error)
                rows.append({**data, "method": method, "repeat": repeat, "atol": atol,
                             "seconds": seconds, "actual_error": finite(err),
                             "attained": err <= atol, "verified": err <= atol and result.status == "converged",
                             "false_convergence": err > atol and result.status == "converged"})
        journal.add(case.name, attributes={"kind": "integrand", "family": case.family, "dimension": case.dimension,
                                           "seed": seed, "draw": draw, "reference_integral": case.truth},
                    attempts=rows)
        if draw == 0 or draw % 4 == 3:
            print(f"cubature {case.family} {case.dimension}d draw {draw + 1} finished", flush=True)


if __name__ == "__main__":
    main()
