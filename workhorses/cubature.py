"""Sample-reusing adaptive cubature and a bounded oblique-hinge fit."""

from dataclasses import dataclass
from functools import lru_cache
import heapq
from itertools import count, product
import math
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate._rules import GenzMalikCubature
from scipy.optimize import least_squares
from scipy.stats import qmc


Array = NDArray[np.float64]
Integrand = Callable[[Array], Array]
EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class Hinge:
    """A scaled positive part of an affine form, in unit-box coordinates."""

    normal: Array
    offset: float
    amplitude: float = 1.0

    def __post_init__(self):
        normal = np.asarray(self.normal, dtype=float)
        if normal.ndim != 1 or not np.isfinite(normal).all() or not all(
            map(math.isfinite, (self.offset, self.amplitude))
        ):
            raise ValueError("hinge coefficients must be finite real numbers")
        object.__setattr__(self, "normal", normal)

    def __call__(self, x: Array) -> Array:
        return self.amplitude * np.maximum(0.0, x @ self.normal - self.offset)

    def integral(self) -> float:
        """Integrate over the unit box with exact arithmetic before FP64 rounding."""
        ratios = [float(item).as_integer_ratio() for item in self.normal]
        offset, offset_denominator = float(self.offset).as_integer_ratio()
        amplitude, amplitude_denominator = float(self.amplitude).as_integer_ratio()
        # All FP64 denominators are powers of two. Use one common scale so
        # the signed truncated powers can be summed without reducing fractions.
        scale = max([offset_denominator, *(den for _, den in ratios)])
        slopes = [num * (scale // den) for num, den in ratios]
        low = sum(min(slope, 0) for slope in slopes) - offset * (scale // offset_denominator)
        widths = [abs(slope) for slope in slopes if slope]
        span = sum(widths)
        if low >= 0:
            return amplitude * (2 * low + span) / (2 * scale * amplitude_denominator)
        high = low + span
        if high <= 0:
            return 0.0
        # Reflecting every coordinate maps the negative tail to a positive
        # corner sum. The positive-part identity lets us use the shorter tail.
        reflect = -low < high
        corners = [(-low if reflect else high, 1)]
        for width in widths:
            shifted = [(arg - width, -sign) for arg, sign in corners if arg > width]
            corners.extend(shifted)
        power = len(widths) + 1
        numerator = sum(sign * arg**power for arg, sign in corners)
        divisor = math.factorial(power) * math.prod(widths)
        if reflect:
            numerator = 2 * numerator + (2 * low + span) * divisor
            divisor *= 2
        return amplitude * numerator / (amplitude_denominator * scale * divisor)


@dataclass(frozen=True)
class Result:
    estimate: float
    error: float
    evaluations: int
    batches: int
    subdivisions: int
    status: str
    pilot_evaluations: int = 0
    fit_accepted: bool = False
    linear_fit: bool = False
    ray_fit: bool = False


@dataclass(frozen=True)
class _Rule:
    nodes: Array
    weights: Array
    difference: Array
    inner: NDArray[np.intp]
    outer: NDArray[np.intp]
    unit_nodes: Array


@lru_cache
def _rule(dimension: int) -> _Rule:
    source = GenzMalikCubature(dimension)
    nodes, weights = (np.asarray(x) for x in source.nodes_and_weights)
    low_nodes, low_weights = (np.asarray(x) for x in source.lower_nodes_and_weights)
    lookup = {tuple(row): i for i, row in enumerate(nodes)}
    scale = 2 ** dimension
    difference = weights.copy() / scale
    difference[[lookup[tuple(row)] for row in low_nodes]] -= low_weights / scale

    def axial(radius_squared):
        axes = np.eye(dimension) * math.sqrt(radius_squared)
        return np.array([[lookup[tuple(row)], lookup[tuple(-row)]] for row in axes])

    return _Rule(nodes, weights / scale, difference, axial(9 / 70), axial(9 / 10), (1 + nodes) / 2)


def _quadratics(points: Array) -> Array:
    first, second = np.triu_indices(points.shape[1])
    return np.column_stack((np.ones(len(points)), points, points[:, first] * points[:, second]))


def _hinge_directions(ndim):
    rng = np.random.default_rng(1925)
    vectors = np.vstack((np.eye(ndim), rng.normal(size=(24, ndim))))
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def _background(points, quadratic):
    return _quadratics(points) if quadratic else np.column_stack((np.ones(len(points)), points))


def _project(columns, values):
    return values - columns @ (columns.T @ values)


def _pilot_points(dimension, size):
    return qmc.Sobol(dimension, scramble=True, rng=np.random.default_rng(11831)).random_base2(
        (size - 1).bit_length()
    )[:size]


@dataclass(frozen=True)
class _Pilot:
    points: Array
    fit: Array
    columns: Array
    triangular: Array
    check_background: Array
    parameters: Array
    features: Array
    norms_squared: Array


@lru_cache(maxsize=16)
def _pilot_design(dimension, fitting, checking, quadratic):
    points = _pilot_points(dimension, fitting + checking)
    train, check = points[:fitting], points[fitting:]
    fitting_model = np.column_stack((train, np.ones(fitting)))
    columns, triangular = np.linalg.qr(_background(train, quadratic), mode="reduced")
    directions = _hinge_directions(dimension)
    parameters, features = [], []
    for normal in directions:
        projection = train @ normal
        for quantile in (0.3, 0.5, 0.7):
            offset = np.quantile(projection, quantile)
            parameters.append(np.r_[normal, -offset])
            features.append(_project(columns, np.maximum(projection - offset, 0)))
    features = np.array(features)
    return _Pilot(points, fitting_model, columns, triangular, _background(check, quadratic),
                  np.array(parameters), features, np.einsum("ij,ij->i", features, features))


def _linear_hinge(fit, columns, target, initial, sign, tolerance):
    """Solve one linear least-squares problem per observed hinge active set."""
    parameters, previous = initial, math.inf
    for _ in range(8):
        locations = fit @ parameters
        residual = _project(columns, sign * np.maximum(locations, 0)) - target
        error = np.linalg.norm(residual)
        if error <= tolerance:
            return Hinge(parameters[:-1], -float(parameters[-1]), sign)
        if error >= previous:
            break
        previous = error
        jacobian = _project(columns, sign * fit * (locations > 0)[:, None])
        parameters, _, rank, _ = np.linalg.lstsq(jacobian, target, rcond=None)
        if rank < fit.shape[1]:
            break
    return None


@dataclass(frozen=True)
class _Rays:
    directions: Array
    points: Array
    nodes: Array


@lru_cache(maxsize=7)
def _ray_design(dimension):
    t = np.linspace(-1, 1, 9)
    rng = np.random.default_rng(9572)
    directions = rng.choice((-1., 1.), size=(4 * dimension + 8, dimension))
    directions *= rng.uniform(0.30, 0.5, directions.shape)
    directions = np.vstack((np.eye(dimension) / 2, directions))
    nodes = 0.5 + directions[:, None, :] * t[None, :, None]
    return _Rays(directions, nodes.reshape(-1, dimension), t)


@lru_cache(maxsize=7)
def _shared_ray_points(dimension):
    rays = _ray_design(dimension)
    length = len(rays.nodes)
    centres = np.arange(length // 2, len(rays.points), length)
    keep = np.ones(len(rays.points), dtype=bool)
    keep[centres[1:]] = False
    inverse = np.cumsum(keep, dtype=np.intp) - 1
    inverse[centres] = inverse[centres[0]]
    return rays.points[keep], inverse


@lru_cache(maxsize=16)
def _ray_validation(dimension, fitting, checking, quadratic):
    points = _pilot_points(dimension, fitting + checking)
    q, r = np.linalg.qr(_background(points[:fitting], quadratic), mode="reduced")
    prediction = _background(points[fitting:], quadratic) @ np.linalg.solve(r, q.T)
    combined = np.vstack((points, _rule(dimension).unit_nodes))
    return combined, prediction


def _recover_ray_hinge(rays, values):
    values = values.reshape(len(rays.directions), -1)
    scale = abs(values).max(axis=1)
    step = rays.nodes[1] - rays.nodes[0]
    # A quadratic has a constant second difference. An interior hinge
    # perturbs at most two adjacent differences, in proportion to its
    # fractional position between the corresponding sampling nodes.
    differences = np.diff(values, n=2, axis=1)
    differences -= np.partition(differences, 3, axis=1)[:, 3:4]
    significant = abs(differences) > (64 * EPS * scale)[:, None]
    index = np.argmax(significant, axis=1)
    last = significant.shape[1] - 1
    count = significant.sum(axis=1)
    following = np.minimum(index + 1, last)
    rows = np.arange(len(values))
    first = differences[rows, index]
    second = np.where(count == 2, differences[rows, following], 0)
    # One outermost difference is ambiguous with a cut near the ray end.
    valid = ((count == 1) & (index > 0) & (index < last)) | (
        (count == 2) & (index < last) & significant[rows, following] & (np.sign(first) == np.sign(second))
    )
    jumps = (first + second) / step
    root_height = rays.nodes[index + 1] * jumps + second
    indices = np.flatnonzero(valid & (abs(root_height) > 512 * EPS * scale) & np.isfinite(jumps))
    if len(indices) < rays.directions.shape[1]:
        return None
    selected_jumps = jumps[indices]
    signs = np.sign(selected_jumps)
    if (signs != signs[0]).any():
        return None
    # Along a ray the slope jump is amplitude * abs(normal @ direction).
    # Root signs give its orientation relative to the side containing the
    # centre. Reversing the original hinge changes it by an affine function.
    projections = -np.sign(root_height[indices]) * selected_jumps
    normal, _, rank, singular = np.linalg.lstsq(rays.directions[indices], projections, rcond=None)
    if rank < len(normal) or singular[-1] < 1e-3 * singular[0]:
        return None
    height = float(np.median(abs(root_height[indices])))
    offset = float(normal.sum() / 2 - height)
    if not np.isfinite(normal).all() or not math.isfinite(offset):
        return None
    return Hinge(normal, offset, float(signs[0]))


def _check_ray_hinge(hinge, points, values, prediction):
    fitting = prediction.shape[1]
    location = points @ hinge.normal - hinge.offset
    side = location[fitting:] > 0
    if min(side.sum(), (~side).sum()) < max(3, len(side) // 10):
        return False
    observed = values[fitting:]
    baseline = prediction @ values[:fitting]
    kink = hinge.amplitude * np.maximum(0, location)
    corrected = kink[fitting:] + prediction @ (values[:fitting] - kink[:fitting])
    error = np.linalg.norm(corrected - observed)
    return error <= min(128 * EPS * np.linalg.norm(observed), 0.55 * np.linalg.norm(baseline - observed))


def _fit_hinge(points: Array, values: Array, check: Array, check_values: Array, *,
               starts: int, quadratic: bool, design: _Pilot | None = None,
               linear: bool = False) -> tuple[Hinge | None, bool]:
    if design is None:
        fit = np.column_stack((points, np.ones(len(points))))
        q, r = np.linalg.qr(_background(points, quadratic), mode="reduced")
        check_background = _background(check, quadratic)
    else:
        fit, q, r, check_background = design.fit, design.columns, design.triangular, design.check_background

    def project(vector):
        return _project(q, vector)

    target = project(values)
    training_floor = 128 * EPS * np.linalg.norm(values)
    if np.linalg.norm(target) <= training_floor:
        return None, False

    candidates = []
    if design is None:
        for normal in _hinge_directions(points.shape[1]):
            projection = points @ normal
            for quantile in (0.3, 0.5, 0.7):
                offset = np.quantile(projection, quantile)
                feature = project(np.maximum(projection - offset, 0))
                if (norm_squared := feature @ feature) > EPS:
                    amplitude = float(feature @ target / norm_squared)
                    if amplitude:
                        improvement = abs(amplitude) * math.sqrt(norm_squared)
                        parameter = np.r_[normal, -offset] * abs(amplitude)
                        candidates.append((improvement, math.copysign(1, amplitude), parameter))
    else:
        for basis, dot, norm_squared in zip(design.parameters, design.features @ target, design.norms_squared):
            if norm_squared > EPS and dot:
                amplitude = float(dot / norm_squared)
                candidates.append((abs(amplitude) * math.sqrt(norm_squared),
                                   math.copysign(1, amplitude), basis * abs(amplitude)))

    def validates(hinge, *, validation_floor=math.inf):
        # Avoid extrapolating an unsampled side. Both validation models fit
        # the same background basis using only the training points.
        side = check @ hinge.normal > hinge.offset
        if min(side.sum(), (~side).sum()) < max(3, len(side) // 10):
            return False
        baseline = check_background @ np.linalg.solve(r, q.T @ values)
        corrected = hinge(check) + check_background @ np.linalg.solve(r, q.T @ (values - hinge(points)))
        error = np.linalg.norm(corrected - check_values)
        return error <= validation_floor and error <= 0.55 * np.linalg.norm(baseline - check_values)

    ranked = sorted(candidates, key=lambda item: -item[0])[:starts]
    if linear:
        validation_floor = 128 * EPS * np.linalg.norm(check_values)
        for _, sign, initial in ranked:
            candidate = _linear_hinge(fit, q, target, initial, sign, training_floor)
            if candidate is not None and validates(candidate, validation_floor=validation_floor):
                return candidate, True

    best = None
    for _, sign, initial in ranked:
        def residual(parameters):
            return project(sign * np.maximum(fit @ parameters, 0)) - target

        def jacobian(parameters):
            return project(sign * fit * ((fit @ parameters) > 0)[:, None])

        solution = least_squares(residual, initial, jac=jacobian, max_nfev=35)
        if best is None or solution.cost < best[0]:
            best = (solution.cost, Hinge(solution.x[:-1], -float(solution.x[-1]), sign))
    return (best[1], False) if best is not None and validates(best[1]) else (None, False)


def integrate(
    f: Integrand, a: Array, b: Array, *, atol: float, rtol: float = 0,
    max_evaluations: int = 100_000, mode: str = "directional",
    hinge: Hinge | None = None, fit_points: int = 256, check_points: int = 64,
    fit_starts: int = 4, quadratic: bool = True,
) -> Result:
    """Integrate a deterministic scalar callback over a finite FP64 box.

    Modes are directional bisection, all-axis splits, a fitted hinge, or a
    caller-supplied hinge in normalized box coordinates. ``blind-linear``
    attempts a linear solve on each sampled active set before nonlinear
    fitting. The blind-rays mode first looks for a quadratic with one affine
    kink on rays through the box centre. The blind-rays-shared mode evaluates
    that common centre once per ray pilot. A fitted hinge corrects the integral
    of the actual callback's residual. Error is estimated.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a.size
    if a.shape != (d,) or b.shape != a.shape or not 2 <= d <= 8:
        raise ValueError("cubature requires two vectors with 2 to 8 coordinates")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("cubature bounds must be finite")
    if not (math.isfinite(atol) and math.isfinite(rtol) and atol >= 0 and 0 <= rtol < 1
            and (atol > 0 or rtol > 0)):
        raise ValueError("atol and rtol must specify a positive finite tolerance")
    ray_mode = mode in {"blind-rays", "blind-rays-shared"}
    if not ray_mode and mode not in {"directional", "all_axes", "blind", "blind-cached", "blind-linear", "oracle"}:
        raise ValueError(f"unknown cubature mode {mode!r}")
    if not isinstance(max_evaluations, (int, np.integer)) or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    if mode == "oracle" and (hinge is None or len(hinge.normal) != d):
        raise ValueError("oracle mode requires a hinge in normalized box coordinates")
    if mode != "oracle" and hinge is not None:
        raise ValueError("only oracle mode accepts a supplied hinge")
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    if (lo == hi).any():
        return Result(0.0, 0.0, 0, 0, 0, "converged")
    width = hi - lo
    volume = float(np.prod(width))
    if not math.isfinite(volume) or volume <= 0 or not np.isfinite(width).all():
        raise ValueError("box volume is outside the representable FP64 range")
    sign = -1 if np.count_nonzero(a > b) % 2 else 1
    rule = _rule(d)
    node_count = len(rule.nodes)
    evaluations = batches = pilot = subdivisions = 0
    linear_fit = ray_fit = False
    if max_evaluations < node_count:
        return Result(0.0, math.inf, 0, 0, 0, "budget")

    def sample(points):
        nonlocal evaluations, batches
        batches += 1
        evaluations += len(points)
        values = np.asarray(f(lo + points * width), dtype=float)
        if values.shape != (len(points),) or not np.isfinite(values).all():
            raise ValueError("integrand must return one finite value per point")
        return values

    root_values = None
    if ray_mode or mode in {"blind", "blind-cached", "blind-linear"}:
        terms = ((d + 1) * (d + 2)) // 2
        if fit_points < terms or check_points < 8 or fit_starts < 1:
            raise ValueError("the hinge fit needs more pilot points or starts")
        size = fit_points + check_points
        if ray_mode:
            rays = _ray_design(d)
            ray_points, gather = (_shared_ray_points(d) if mode == "blind-rays-shared" else (rays.points, None))
            fitting = min(fit_points, max(24, (terms if quadratic else d + 1) + 8))
            trial_size = len(ray_points) + fitting + check_points
            if trial_size + size + node_count <= max_evaluations:
                with np.errstate(over="ignore", invalid="ignore"):
                    sampled = sample(ray_points)
                    candidate = _recover_ray_hinge(rays, sampled if gather is None else sampled[gather])
                    if candidate is not None:
                        points, prediction = _ray_validation(d, fitting, check_points, quadratic)
                        combined = sample(points)
                        root_values = combined[-node_count:]
                        if _check_ray_hinge(candidate, points[:-node_count], combined[:-node_count], prediction):
                            hinge, linear_fit, ray_fit = candidate, True, True
                pilot = evaluations - (node_count if root_values is not None else 0)
        spent_root = node_count if root_values is not None else 0
        if hinge is None and evaluations + size + node_count - spent_root <= max_evaluations:
            design = _pilot_design(d, fit_points, check_points, quadratic) if mode != "blind" else None
            points = design.points if design else _pilot_points(d, size)
            values = sample(points)
            pilot = evaluations - spent_root
            hinge, linear_fit = _fit_hinge(points[:fit_points], values[:fit_points], points[fit_points:],
                                           values[fit_points:], starts=fit_starts, quadratic=quadratic,
                                           design=design, linear=ray_mode or mode == "blind-linear")
    correction = hinge.integral() * volume if hinge is not None else 0.0
    correction_roundoff = math.ulp(correction) if hinge is not None else 0.0
    if ray_fit:
        kink = hinge(rule.unit_nodes)
        residual = root_values - kink
        total = float(residual @ rule.weights) * volume
        rounding = abs(root_values) @ abs(rule.weights) + abs(kink) @ abs(rule.weights)
        embedded = max(abs(residual @ rule.difference), 16 * EPS * rounding) * volume
        error = math.fsum((correction_roundoff, embedded))
        if math.isfinite(total) and math.isfinite(error) and error <= atol + rtol * abs(total + correction):
            return Result(sign * math.fsum((correction, total)), error, evaluations, batches, 0, "converged",
                          pilot, True, True, True)
    serial = count()

    def measure(lower, upper, measured=None):
        mid, scale = (lower + upper) / 2, (upper - lower) / 2
        points = mid[:, None, :] + scale[:, None, :] * rule.nodes
        flattened = points.reshape(-1, d)
        vals = (sample(flattened) if measured is None else measured).reshape(len(lower), node_count)
        rounding = abs(vals) @ abs(rule.weights)
        if hinge is not None:
            h = hinge(flattened).reshape(vals.shape)
            rounding += abs(h) @ abs(rule.weights)
            vals = vals - h
        factor = np.prod(upper - lower, axis=1) * volume
        estimates = (vals @ rule.weights) * factor
        local_errors = np.maximum(abs(vals @ rule.difference), 16 * EPS * rounding)
        errors = local_errors * factor
        if not (np.isfinite(estimates).all() and np.isfinite(errors).all()):
            raise FloatingPointError("a cubature estimate or error overflowed FP64")
        fourth = abs(vals[:, rule.outer].sum(axis=-1) + 12 * vals[:, :1]
                     - 7 * vals[:, rule.inner].sum(axis=-1))
        # HCubature's tie rule compares the fourth differences near the
        # embedded-error floor, then takes the longest eligible side.
        tied = fourth.max(axis=1, keepdims=True) - fourth <= (local_errors / 10 ** d)[:, None]
        axes = np.argmax(np.where(tied, width * scale, -1), axis=1)
        return [
            (-float(error), next(serial), float(value), int(axis), start, stop)
            for error, value, axis, start, stop in zip(errors, estimates, axes, lower, upper)
        ]

    regions = measure(np.zeros((1, d)), np.ones((1, d)), root_values)
    total, error = regions[0][2], -regions[0][0]
    status = "budget"
    while True:
        if error + correction_roundoff <= atol + rtol * abs(total + correction):
            total = math.fsum(row[2] for row in regions)
            error = math.fsum(-row[0] for row in regions)
            if error + correction_roundoff <= atol + rtol * abs(total + correction):
                status = "converged"
                break
        neg_error, _, value, axis, lower, upper = regions[0]
        axes = tuple(range(d)) if mode == "all_axes" else (axis,)
        child_count = 2 ** len(axes)
        if evaluations + child_count * node_count > max_evaluations:
            break
        midpoint = (lower + upper) / 2
        if any(not lower[i] < midpoint[i] < upper[i] for i in axes):
            status = "roundoff"
            break
        children_low, children_high = [], []
        for part in product((0, 1), repeat=len(axes)):
            start, stop = lower.copy(), upper.copy()
            for i, upper_half in zip(axes, part):
                if upper_half:
                    start[i] = midpoint[i]
                else:
                    stop[i] = midpoint[i]
            children_low.append(start)
            children_high.append(stop)
        children = measure(np.array(children_low), np.array(children_high))
        heapq.heappop(regions)
        for child in children:
            heapq.heappush(regions, child)
        total = math.fsum((total, -value, *(row[2] for row in children)))
        error = math.fsum((error, neg_error, *(-row[0] for row in children)))
        subdivisions += 1
    estimate = math.fsum((correction, *(row[2] for row in regions)))
    error = math.fsum((correction_roundoff, *(-row[0] for row in regions)))
    return Result(sign * estimate, error, evaluations, batches, subdivisions, status, pilot,
                  hinge is not None, linear_fit, ray_fit)
