"""Compare cubature at independently checked tolerances on manufactured integrands."""

import argparse
import ctypes as ct
from dataclasses import asdict, dataclass
from itertools import product
import math
from pathlib import Path
import statistics
from time import perf_counter

import mpmath
import numpy as np
import scipy
from scipy.integrate import cubature as scipy_cubature
from scipy.special import erf

from benchmarks.common import BIN, SEED, interval, metadata, schedule, write_result
from workhorses.cubature import Hinge, Integrand, Result, integrate


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    dimension: int
    f: Integrand
    truth: float
    hinge: Hinge | None = None


def reference_hinge(hinge):
    """Independent high-precision survival-volume integral on a unit box."""
    with mpmath.workdps(100):
        widths = [abs(mpmath.mpf(float(x))) for x in hinge.normal if x]
        origin = sum(min(mpmath.mpf(float(x)), 0) for x in hinge.normal)
        # For Y=sum(width_i*U_i), reflect Y to integrate its CDF from 0 to top.
        top = origin + sum(widths) - mpmath.mpf(hinge.offset)
        ndim = len(widths)
        numerator = mpmath.fsum(
            (-1) ** sum(bits) * max(0, top - sum(w for w, bit in zip(widths, bits) if bit)) ** (ndim + 1)
            for bits in product((0, 1), repeat=ndim)
        )
        return float(mpmath.mpf(hinge.amplitude) * numerator / (math.factorial(ndim + 1) * math.prod(widths)))


def cases(cohort, *, quick=False):
    if cohort == "smoke":
        specs = ((2, 100, "quadratic-kink"), (3, 101, "smooth-gaussian"))
    elif cohort == "development":
        specs = tuple((d, 2000 + i + 10 * j, family)
                      for j, family in enumerate(("quadratic-kink", "mixed-kink", "curved", "smooth-gaussian"))
                      for i, d in enumerate((2, 3, 4, 6, 8)))
    else:
        specs = tuple((d, 7400 + i + 20 * j, family)
                      for j, family in enumerate(("quadratic-kink", "mixed-kink", "two-kinks", "curved",
                                                  "smooth-gaussian", "oscillatory", "product-peak", "narrow"))
                      for i, d in enumerate((2, 3, 4, 6, 8)))
        specs += tuple((d, 8700 + i + 10 * j, family)
                       for j, family in enumerate(("quadratic-kink", "mixed-kink"))
                       for i, d in enumerate((2, 3, 4, 6, 8)))
    return [make_case(d, SEED + seed, family) for d, seed, family in (specs[:4] if quick else specs)]


def make_case(d, seed, family):
    name = f"{family}-{d}d-{seed}"
    rng = np.random.default_rng(seed)
    coefs = rng.uniform(-0.12, 0.12, d)
    intercept = rng.uniform(0.25, 0.75)
    base_truth = intercept + coefs.sum() / 3

    def polynomial(x):
        return intercept + (x * x) @ coefs

    if family in ("quadratic-kink", "mixed-kink", "two-kinks"):
        def random_hinge():
            normal = rng.uniform(0.2, 1.0, d) * rng.choice([-1, 1], d)
            normal /= np.linalg.norm(normal)
            return Hinge(normal, float(0.5 * normal.sum() + rng.uniform(-0.17, 0.17)), float(rng.uniform(0.6, 1.6)))

        hinge = random_hinge()
        truth = base_truth + reference_hinge(hinge)
        if family == "mixed-kink":
            omega = rng.uniform(3, 9, d)
            phase = rng.uniform(-1, 1)
            amplitude = 0.10
            # Center the box to factor the analytic oscillatory integral.
            wave_truth = amplitude * math.cos(phase) * np.prod(np.sinc(omega / (2 * np.pi)))

            def mixed(x):
                return polynomial(x) + hinge(x) + amplitude * np.cos((x - 0.5) @ omega + phase)

            return Case(name, family, d, mixed, float(truth + wave_truth), hinge)
        if family == "two-kinks":
            other = random_hinge()
            return Case(name, family, d, lambda x: polynomial(x) + hinge(x) + other(x),
                        float(truth + reference_hinge(other)))
        return Case(name, family, d, lambda x: polynomial(x) + hinge(x), float(truth), hinge)
    if family == "curved":
        # Integral in x0 gives (1 - 0.25 - 0.50*x1^2)^2 / 2.
        exact = 0.5 * (0.75 ** 2 - 0.75 / 3 + 0.25 / 5)
        return Case(name, family, d, lambda x: polynomial(x) + np.maximum(x[:, 0] - 0.25 - 0.5 * x[:, 1] ** 2, 0),
                    float(base_truth + exact))
    centers = rng.uniform(0.18, 0.82, d)
    if family == "oscillatory":
        omega = rng.uniform(1, 15, d)
        phase = rng.uniform(0, 2 * np.pi)
        truth = math.cos(phase + 0.5 * omega.sum()) * np.prod(np.sinc(omega / (2 * np.pi)))
        return Case(name, family, d, lambda x: np.cos(phase + x @ omega), float(truth))
    if family == "product-peak":
        alpha = rng.uniform(0.3, 1.2, d)
        truth = np.prod(alpha * (np.arctan((1 - centers) / alpha) + np.arctan(centers / alpha)))
        return Case(name, family, d, lambda x: np.prod(1 / (1 + ((x - centers) / alpha) ** 2), axis=1), float(truth))
    if family == "narrow":
        # Normalize the amplitude so a missed peak can fail the absolute target.
        scales = rng.uniform(45, 90, d)
        amplitude = float(np.prod(scales / math.sqrt(math.pi)))
    else:
        scales = rng.uniform(1.0, 4.0, d)
        amplitude = 1.0
    integral = (math.sqrt(math.pi) / (2 * scales)) * (erf(scales * (1 - centers)) + erf(scales * centers))
    return Case(name, family, d, lambda x: amplitude * np.exp(-np.sum((scales * (x - centers)) ** 2, axis=1)),
                float(amplitude * np.prod(integral)))


class _Budget(Exception):
    pass


def stock_scipy(case, atol, budget):
    evaluated = calls = 0

    def sample(points):
        nonlocal evaluated, calls
        if evaluated + len(points) > budget:
            raise _Budget
        evaluated += len(points)
        calls += 1
        return case.f(points)

    try:
        result = scipy_cubature(sample, np.zeros(case.dimension), np.ones(case.dimension),
                                rule="genz-malik", atol=atol, rtol=0.0, workers=1)
    except _Budget:
        return Result(math.nan, math.inf, evaluated, calls, 0, "budget")
    return Result(float(result.estimate), float(result.error), evaluated, calls, result.subdivisions,
                  "converged" if result.status == "converged" else result.status)


def cuhre_baseline():
    integer, real = ct.c_int, ct.c_double
    ip, rp = ct.POINTER(integer), ct.POINTER(real)
    callback_type = ct.CFUNCTYPE(integer, ip, rp, ip, rp, ct.c_void_p, ip, ip)
    library = ct.CDLL(str(BIN / "cuhre.dll"))
    cuhre = library.Cuhre
    cuhre.restype = None
    cuhre.argtypes = [integer, integer, callback_type, ct.c_void_p, integer, real, real,
                      integer, integer, integer, integer, ct.c_char_p, ct.c_void_p, ip, ip, ip, rp, rp, rp]

    def integrate_cuhre(case, atol, budget):
        seen = calls = 0
        failure = []

        @callback_type
        def sample(dimension, points, components, values, data, nvec, core):
            nonlocal seen, calls
            size, d = nvec[0], dimension[0]
            if seen + size > budget:
                failure.append("budget")
                return -999  # Cuba's supported integrand abort return.
            try:
                calls += 1
                seen += size
                x = np.ctypeslib.as_array(points, shape=(size * d,)).reshape(size, d)
                np.ctypeslib.as_array(values, shape=(size,))[:] = case.f(x)
            except Exception as exc:
                failure.append(exc)
                return -999
            return 0

        regions, samples, fail = integer(), integer(), integer()
        estimate, error, probability = real(), real(), real()
        cuhre(case.dimension, 1, sample, None, 2048, 0.0, atol, 0, 0, budget, 0,
              None, None, ct.byref(regions), ct.byref(samples), ct.byref(fail),
              ct.byref(estimate), ct.byref(error), ct.byref(probability))
        if failure and failure[0] != "budget":
            raise failure[0]
        if failure:
            return Result(math.nan, math.inf, seen, calls, max(0, regions.value - 1), "budget")
        return Result(estimate.value, error.value, seen, calls, max(0, regions.value - 1),
                      "converged" if fail.value == 0 else f"cuhre_{fail.value}")

    return integrate_cuhre


def study(args):
    configurations = {
        "directional": {"mode": "directional"},
        "reuse-all-axes": {"mode": "all_axes"},
        "blind": {"mode": "blind"},
        "blind-96": {"mode": "blind", "fit_points": 96, "check_points": 32, "fit_starts": 2},
        "blind-affine": {"mode": "blind", "fit_points": 128, "check_points": 32, "fit_starts": 3, "quadratic": False},
        "oracle": {"mode": "oracle"},
    }
    external = {"scipy": stock_scipy}
    if (BIN / "cuhre.dll").is_file():
        external["cuhre"] = cuhre_baseline()
    elif args.cohort != "smoke":
        raise FileNotFoundError("build Cuhre before recording the benchmark")
    methods = tuple(external) + ("directional", "reuse-all-axes", "oracle") + tuple(args.detector)
    # Warm each kernel on the same small, analytically-integrable pilot.
    warm = make_case(2, 91, "quadratic-kink")

    def evaluate(method, case, atol):
        if method in external:
            return external[method](case, atol, args.budget)
        options = configurations[method]
        return integrate(case.f, np.zeros(case.dimension), np.ones(case.dimension), atol=atol,
                         max_evaluations=args.budget, hinge=case.hinge if method == "oracle" else None, **options)

    for method in methods:
        evaluate(method, warm, 1e-4)

    rows = []
    for i, case in enumerate(cases(args.cohort, quick=args.quick)):
        applicable = [m for m in methods if (m != "oracle" or case.hinge is not None)]
        for atol in args.atol:
            for repeat, method in schedule(applicable, seed=SEED + i, repeats=args.repeats):
                start = perf_counter()
                result = evaluate(method, case, atol)
                seconds = perf_counter() - start
                measured = asdict(result)
                actual_error = abs(result.estimate - case.truth)
                for key in ("estimate", "error"):
                    if not math.isfinite(measured[key]):
                        measured[key] = None
                rows.append({
                    "case": case.name, "family": case.family, "dimension": case.dimension,
                    "atol": atol, "truth": case.truth, "repeat": repeat, "method": method,
                    "seconds": seconds, **measured,
                    "actual_error": actual_error if math.isfinite(actual_error) else None,
                    "attained": actual_error <= atol,
                    "verified": result.status == "converged" and actual_error <= atol,
                    "false_convergence": result.status == "converged" and actual_error > atol,
                })
        print(f"{case.name} finished", flush=True)
    result = {
        "metadata": {**metadata(), "numpy": np.__version__, "scipy": scipy.__version__, "seed": SEED},
        "settings": {"cohort": args.cohort, "detectors": args.detector, "atol": args.atol, "budget": args.budget,
                     "repeats": args.repeats, "quick": args.quick, "detector_options": configurations},
        "summary": summarize(rows, methods),
        "attempts": rows,
    }
    write_result(args.output, result)
    print(result["summary"], flush=True)


def summarize(rows, methods):
    info = {}
    for method in methods:
        unique = [r for r in rows if r["method"] == method and r["repeat"] == 0]
        info[method] = {
            "cases": len(unique), "verified": sum(r["verified"] for r in unique),
            "false_convergences": sum(r["false_convergence"] for r in unique),
            "budget_exhaustions": sum(r["status"] == "budget" for r in unique),
            "fit_accepted": sum(r["fit_accepted"] for r in unique),
            "target_outcomes": {
                family: {"verified": sum(r["verified"] and r["family"] == family for r in unique),
                         "cases": sum(r["family"] == family for r in unique)}
                for family in ("quadratic-kink", "mixed-kink")
            },
        }
        paired_evaluations, paired_time = {}, {}
        for row in unique:
            if row["family"] not in {"quadratic-kink", "mixed-kink"} or not row["verified"]:
                continue
            peers = [r for r in rows if (r["case"], r["atol"]) == (row["case"], row["atol"])]
            controls = [r for r in peers if r["method"] == "directional" and r["verified"]]
            candidates = [r for r in peers if r["method"] == method and r["verified"]]
            if controls:
                paired_evaluations.setdefault(row["case"], []).append(controls[0]["evaluations"] / row["evaluations"])
                paired_time.setdefault(row["case"], []).append(
                    statistics.median(r["seconds"] for r in controls) / statistics.median(r["seconds"] for r in candidates))
        info[method]["target_evaluation_gain_on_verified_pairs"] = interval(map(statistics.median, paired_evaluations.values()))
        info[method]["target_time_gain_on_verified_pairs"] = interval(map(statistics.median, paired_time.values()))
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("smoke", "development", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--atol", type=float, nargs="+", default=[1e-5, 1e-6])
    parser.add_argument("--detector", choices=("blind", "blind-96", "blind-affine"), nargs="+", default=["blind"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.cohort == "holdout" and (args.quick or len(args.detector) != 1):
        parser.error("holdout accepts exactly one frozen detector and uses the full case list")
    study(args)


if __name__ == "__main__":
    main()
