"""Count coalesced pilot samples and verify every frozen integrand and tolerance."""

import argparse
from dataclasses import asdict
from hashlib import sha256 as digest
from pathlib import Path

import numpy as np

from benchmarks.common import ROOT
from benchmarks.fetch import sha256
from benchmarks.followup_cubature import BUDGET, TOLERANCES, cases
from benchmarks.paper_common import Journal, finite, load_journal
from workhorses.cubature import integrate


REFERENCE = ROOT / "doubling/runs/cubature-v2.jsonl"


def run(path):
    draws = list(cases(False))
    stamp = digest(repr([(c.name, seed, c.truth) for c, seed in draws]).encode()).hexdigest()
    header, previous = load_journal(REFERENCE)
    known = {u["unit"]: u for u in previous}
    if header["protocol"]["draws_sha256"] != stamp or known.keys() != {c.name for c, _ in draws}:
        raise ValueError("cubature draw or truth differs from the frozen reference")
    journal = Journal(path, {"study": "shared-ray-centre-v1", "reference_sha256": sha256(REFERENCE),
                             "draws_sha256": stamp, "budget": BUDGET, "tolerances": list(TOLERANCES),
                             "code_sha256": sha256(ROOT / "workhorses/cubature.py"), "wall_timed": False})
    for case, seed in draws:
        if case.name in journal.completed:
            continue
        attrs = {"seed": seed, "dimension": case.dimension, "family": case.family, "truth": case.truth}
        if known[case.name]["attributes"] != attrs:
            raise ValueError(f"{case.name} reference differs")
        rows = []
        for tol in TOLERANCES:
            spent = []
            def observed(x):
                spent.append(len(x))
                return case.f(x)
            result = integrate(observed, np.zeros(case.dimension), np.ones(case.dimension),
                               atol=tol, max_evaluations=BUDGET, mode="blind-rays-shared")
            error = abs(result.estimate - case.truth)
            if sum(spent) != result.evaluations or len(spent) != result.batches or sum(spent) > BUDGET:
                raise ValueError(f"{case.name} callback accounting differs")
            rows.append({"tolerance": tol, **{k: finite(v) if isinstance(v, float) else v
                                             for k, v in asdict(result).items()},
                         "actual_error": finite(error), "verified": result.status == "converged" and error <= tol,
                         "false_convergence": result.status == "converged" and error > tol})
        journal.add(case.name, attributes=attrs, attempts=rows)
    print(f"Verified point counts for {len(journal.completed)} integrands", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "manuscripts/runs/cubature.jsonl")
    run(parser.parse_args().output)
