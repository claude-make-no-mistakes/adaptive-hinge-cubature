"""Benchmark exact dyadic hinge integration and the active-set pilot fit."""

import argparse
from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256 as digest
from itertools import product
import math
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np
import scipy

from benchmarks.common import BIN, ROOT, schedule
from benchmarks.cubature import cuhre_baseline, make_case
from benchmarks.fetch import sha256
from benchmarks.paper_common import Journal, finite
from benchmarks.paper_cubature import BUDGET, DIMENSIONS, TOLERANCES
from workhorses.cubature import Hinge, integrate


FOLLOWUP = ROOT / "followup"
SEED = 2_624_811
FAMILIES = {"quadratic-kink": 16, "mixed-kink": 16, "two-kinks": 8, "curved": 8,
            "smooth-gaussian": 8, "oscillatory": 8, "product-peak": 8, "narrow": 8}
METHODS = ("fraction-cached", "blind-cached", "blind-linear", "directional", "cuhre")
MICRO_REPEATS = 6
MICRO_CALLS = 12
REPEATS = 3


def fraction_integral(hinge):
    """The archived implementation, evaluating all box corners with Fraction."""
    slopes = [Fraction(float(item)) for item in hinge.normal]
    offset = Fraction(float(hinge.offset))
    amplitude = Fraction(float(hinge.amplitude))
    low = sum(min(a, 0) for a in slopes) - offset
    widths = [abs(a) for a in slopes if a]
    if low >= 0:
        return float(amplitude * (low + sum(widths, Fraction(0)) / 2))
    if low + sum(widths) <= 0:
        return 0.0
    dimension = len(widths)
    total = Fraction(0)
    for corner in product((0, 1), repeat=dimension):
        argument = low + sum(a for a, use in zip(widths, corner) if use)
        if argument > 0:
            total += (-1) ** (dimension - sum(corner)) * argument ** (dimension + 1)
    return float(amplitude * total / (math.factorial(dimension + 1) * math.prod(widths)))


def cases(probe):
    for index, (family, count) in enumerate(FAMILIES.items()):
        for dimension, draw in product(DIMENSIONS, range(min(count, 1) if probe else count)):
            seed = SEED + 10_000 * index + 100 * dimension + draw + (40_000_000 if probe else 0)
            yield make_case(dimension, seed, family), seed


def _micro(hinge, seed):
    methods = {"fraction": lambda: fraction_integral(hinge), "dyadic": hinge.integral}
    expected = fraction_integral(hinge)
    rows = []
    for repeat, method in schedule(methods, seed=seed, repeats=MICRO_REPEATS):
        func = methods[method]
        start = perf_counter_ns()
        for _ in range(MICRO_CALLS):
            value = func()
        elapsed = perf_counter_ns() - start
        rows.append({"method": method, "repeat": repeat, "seconds": elapsed / (1e9 * MICRO_CALLS),
                     "value": value, "verified": expected.hex() == value.hex()})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FOLLOWUP / "runs" / "cubature.jsonl")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    prepared = list(cases(args.probe))
    protocol = {"study": "cubature-dyadic-linear-v1", "probe": args.probe, "seed": SEED,
                "draws_sha256": digest(repr([(c.name, s, c.truth) for c, s in prepared]).encode()).hexdigest(),
                "dimensions": list(DIMENSIONS), "families": FAMILIES, "methods": list(METHODS),
                "micro_repeats": MICRO_REPEATS, "micro_calls": MICRO_CALLS, "repeats": REPEATS,
                "tolerances": list(TOLERANCES), "budget": BUDGET, "scipy": scipy.__version__,
                "numpy": np.__version__, "cuhre_sha256": sha256(BIN / "cuhre.dll")}
    journal = Journal(args.output, protocol)
    cuhre = cuhre_baseline()

    def attempt(case, method, tolerance):
        if method == "cuhre":
            start = perf_counter_ns()
            result = cuhre(case, tolerance, BUDGET)
            return result, (perf_counter_ns() - start) / 1e9
        a, b = np.zeros(case.dimension), np.ones(case.dimension)
        options = {"atol": tolerance, "max_evaluations": BUDGET,
                   "mode": "blind-cached" if method == "fraction-cached" else method}
        implementation = fraction_integral if method == "fraction-cached" else Hinge.integral
        with patch.object(Hinge, "integral", implementation):
            start = perf_counter_ns()
            result = integrate(case.f, a, b, **options)
            return result, (perf_counter_ns() - start) / 1e9

    for d in DIMENSIONS:
        sample = make_case(d, SEED - 1, "quadratic-kink")
        for method in METHODS:
            attempt(sample, method, TOLERANCES[0])

    for case, seed in prepared:
        if case.family == "quadratic-kink" and f"kernel/{case.name}" not in journal.completed:
            journal.add(f"kernel/{case.name}", attributes={"kind": "kernel", "dimension": case.dimension, "seed": seed},
                        attempts=_micro(case.hinge, seed))
        if case.name in journal.completed:
            continue
        rows = []
        for tolerance in TOLERANCES:
            for repeat, method in schedule(METHODS, seed=seed + int(tolerance * 10**7), repeats=REPEATS):
                result, seconds = attempt(case, method, tolerance)
                error = abs(result.estimate - case.truth)
                values = asdict(result)
                values["estimate"], values["error"] = finite(result.estimate), finite(result.error)
                rows.append({**values, "method": method, "repeat": repeat, "tolerance": tolerance, "seconds": seconds,
                             "actual_error": finite(error), "attained": error <= tolerance,
                             "verified": error <= tolerance and result.status == "converged",
                             "false_convergence": error > tolerance and result.status == "converged"})
        journal.add(case.name, attributes={"kind": "integrand", "family": case.family, "dimension": case.dimension,
                                           "seed": seed, "reference": case.truth}, attempts=rows)
        if seed % 8 == 0:
            print(f"{case.name} finished", flush=True)


if __name__ == "__main__":
    main()
