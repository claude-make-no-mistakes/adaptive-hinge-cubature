import unittest
import math

import numpy as np

from benchmarks.followup_cubature import fraction_integral
from workhorses.cubature import Hinge, integrate


class CubatureTests(unittest.TestCase):
    def test_analytic_hinge_handles_signs_and_tiny_slopes(self):
        # Unit triangle above x+y=1 with height increasing away from the edge.
        self.assertAlmostEqual(Hinge(np.array([1, 1]), 1).integral(), 1 / 6, places=15)
        # Reflect y in x-y and integrate the same triangle with signed height.
        self.assertAlmostEqual(Hinge(np.array([1, -1, 0]), 0, -2).integral(), -1 / 3, places=15)
        # The x integral is (1/2+eps*y)^2/2. Its variation is beneath FP64 rounding.
        self.assertEqual(Hinge(np.array([1, 2.0 ** -300]), 0.5).integral(), 0.125)

    def test_distinct_nodes_preserve_degree_seven_and_reversed_box(self):
        value = integrate(lambda x: x[:, 0] ** 4 * x[:, 1] ** 2 * x[:, 2],
                          [2, 0, 0], [0, 1, 1], atol=1e-12, max_evaluations=33)
        # Product of one-dimensional monomial antiderivatives on the oriented box.
        self.assertAlmostEqual(value.estimate, -32 / (5 * 3 * 2), places=13)
        # The 3D Genz–Malik rule has 1+2*d*(d+1)+2**d distinct nodes.
        self.assertEqual((value.evaluations, value.batches), (33, 1))
        # The degree-five embedded rule sees a genuine degree-six term.
        self.assertEqual(value.status, "budget")

    def test_dyadic_corner_sum_rounds_like_exact_fraction_reference(self):
        rng = np.random.default_rng(77591)
        limits = [Hinge(np.zeros(2), -math.ulp(0.0), -1.0),
                  Hinge(np.array([1.0, math.ulp(0.0)]), 0.5),
                  Hinge(np.array([1e300, -1e-300]), 9e299, 1e-200)]
        for dimension in (1, 2, 4, 8):
            for draw in range(24):
                normal = np.ldexp(rng.uniform(-1, 1, dimension), rng.integers(-1000, 1001, dimension))
                middle = 0.5 * math.fsum(normal)
                offset = math.nextafter(middle, -math.inf if draw % 2 else math.inf)
                limits.append(Hinge(normal, offset, -0.75 if draw % 3 else 0.5))
        for hinge in limits:
            # Each Fraction evaluates exactly the original signed box-corner identity.
            self.assertEqual(hinge.integral().hex(), fraction_integral(hinge).hex(), hinge)

    def test_oracle_integrates_piecewise_polynomial_without_subdivision(self):
        hinge = Hinge(np.array([1.0, 1.0]), 1.0)
        value = integrate(lambda x: x[:, 1] ** 2 + hinge(x), [0, 0], [1, 1],
                          atol=1e-12, mode="oracle", hinge=hinge)
        # Unit-box integral of y^2 plus the analytic triangle integral.
        self.assertAlmostEqual(value.estimate, 0.5, places=14)
        self.assertEqual((value.subdivisions, value.status), (0, "converged"))

    def test_known_blind_hinge_charges_the_pilot(self):
        hinge = Hinge(np.array([1.0, -0.8, 0.5]), 0.25)
        value = integrate(lambda x: x[:, 1] ** 2 + hinge(x), [0] * 3, [1] * 3,
                          atol=1e-9, mode="blind")
        # The exact hinge integral and the monomial antiderivative define the reference.
        self.assertAlmostEqual(value.estimate, 1 / 3 + hinge.integral(), places=9)
        self.assertEqual((value.fit_accepted, value.pilot_evaluations, value.status), (True, 320, "converged"))
        self.assertGreaterEqual(value.evaluations, 353)

    def test_cached_geometry_keeps_value_and_sample_count(self):
        hinge = Hinge(np.array([1.0, -0.8, 0.5]), 0.25)
        integrand = lambda x: x[:, 1] ** 2 + hinge(x)
        first, cached = (
            integrate(integrand, [0] * 3, [1] * 3, atol=1e-9, mode=method)
            for method in ("blind", "blind-cached")
        )
        # Exact result is the hinge integral plus the unit-box y^2 moment.
        self.assertAlmostEqual(cached.estimate, 1 / 3 + hinge.integral(), places=9)
        self.assertEqual((cached.evaluations, cached.fit_accepted, cached.status),
                         (first.evaluations, first.fit_accepted, first.status))

    def test_active_set_uses_same_budget_and_recovers_an_affine_hinge(self):
        hinge = Hinge(np.array([1.0, -0.8, 0.5]), 0.25)
        integrand = lambda x: x[:, 1] ** 2 + hinge(x)
        result = integrate(integrand, [0] * 3, [1] * 3, atol=1e-10, mode="blind-linear")
        # The exact hinge reference and the unit-box y² moment give the answer.
        self.assertAlmostEqual(result.estimate, 1 / 3 + fraction_integral(hinge), places=13)
        self.assertEqual((result.status, result.linear_fit, result.evaluations), ("converged", True, 353))
        nonlinear = lambda x: integrand(x) + 0.1 * np.cos(9 * x[:, 0] + 5 * x[:, 1])
        reference, candidate = (integrate(nonlinear, [0] * 3, [1] * 3, atol=1e-8, mode=mode, max_evaluations=1024)
                                for mode in ("blind-cached", "blind-linear"))
        # These samples contain a non-polynomial component outside the exact-fit trigger.
        self.assertEqual(reference, candidate)

    def test_reported_budget_does_not_drop_an_incomplete_child_pair(self):
        value = integrate(lambda x: np.exp(x[:, 0] + x[:, 1]), [0, 0], [1, 1],
                          atol=1e-12, max_evaluations=50)
        # GM uses 17 nodes in 2D, and another complete split needs 34.
        self.assertEqual((value.status, value.evaluations, value.subdivisions), ("budget", 17, 0))

    def test_rays_correct_the_residual_for_a_negative_hinge_on_a_reversed_box(self):
        lo = np.array([-3., 0., 7.])
        width = np.array([2., 3., 4.])
        hinge = Hinge(np.array([-0.8, 0.4, 0.9]), 0.23, -2.)

        def integrand(x):
            u = (x - lo) / width
            return u[:, 0] * u[:, 2] + hinge(u)

        a, b = lo.copy(), lo + width
        a[1], b[1] = b[1], a[1]
        result = integrate(integrand, a, b, atol=1e-10, mode="blind-rays")
        self.assertEqual((result.status, result.ray_fit), ("converged", True))
        self.assertAlmostEqual(result.estimate, -24 * (1 / 4 + fraction_integral(hinge)), places=10)
        shared = integrate(integrand, a, b, atol=1e-10, mode="blind-rays-shared")
        self.assertTrue(shared.ray_fit)
        # The same exact integral holds, and 23 rays share one centre in 3D.
        self.assertAlmostEqual(shared.estimate, -24 * (1 / 4 + fraction_integral(hinge)), places=10)
        self.assertEqual(result.evaluations - shared.evaluations, 22)
        self.assertEqual(result.pilot_evaluations - shared.pilot_evaluations, 22)

    def test_rays_defer_to_existing_fit_when_the_budget_is_tight(self):
        hinge = Hinge(np.array([1., -0.8, 0.5]), 0.35)
        integrand = lambda x: x[:, 1] ** 2 + hinge(x)
        control, result = (integrate(integrand, [0] * 3, [1] * 3, atol=1e-10,
                                     mode=mode, max_evaluations=353)
                           for mode in ("blind-linear", "blind-rays"))
        self.assertEqual(result, control)
        self.assertLessEqual(result.evaluations, 353)


if __name__ == "__main__":
    unittest.main()
