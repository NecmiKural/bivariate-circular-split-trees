"""Brute-force correctness check for the circular split finder.

The find_best_split() in python/ sweeps a growing radius around each candidate
centre and only scores the boundaries between consecutive distinct distances.
This script checks that shortcut against a reference implementation that
enumerates every candidate circle explicitly and recomputes information gain
from scratch, on randomly generated problems.

For each random problem it asserts that
  1. the best information gain matches the brute-force optimum,
  2. the returned (centre, radius) is one of the optimal representative
     circles, i.e. radius = (r_min + r_max) / 2 for that partition, and
  3. the returned circle really produces the reported gain when the
     inside/outside test is applied to the data.

Every third problem uses a small integer grid so that many instances sit at
exactly the same distance from a centre, which exercises the tie handling.

Run with numpy and numba installed:
    python experiments/validate_against_brute_force.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from BivariateNumericAttributeSplitter import find_best_split  # noqa: E402

TRIALS = 200
TOL = 1e-9


def entropy(counts):
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())


def information_gain(left, right):
    n = left.sum() + right.sum()
    if left.sum() == 0 or right.sum() == 0:
        return 0.0
    return (entropy(left + right)
            - left.sum() / n * entropy(left)
            - right.sum() / n * entropy(right))


def brute_force(x1, x2, y, num_classes):
    """Best gain and every representative circle achieving it."""
    best_gain, best_circles = 0.0, []
    for c in range(len(y)):
        d = np.sqrt((x1 - x1[c]) ** 2 + (x2 - x2[c]) ** 2)
        distinct = np.unique(d)
        # A radius anywhere in (r_min, r_max] gives the same partition, so one
        # band between consecutive distinct distances is one candidate circle.
        for r_min, r_max in zip(distinct[:-1], distinct[1:]):
            inside = d <= r_min
            gain = information_gain(np.bincount(y[inside], minlength=num_classes),
                                    np.bincount(y[~inside], minlength=num_classes))
            circle = (x1[c], x2[c], (r_min + r_max) / 2.0)
            if gain > best_gain + TOL:
                best_gain, best_circles = gain, [circle]
            elif abs(gain - best_gain) <= TOL:
                best_circles.append(circle)
    return best_gain, best_circles


def random_problem(rng, trial):
    n = int(rng.integers(8, 40))
    num_classes = int(rng.integers(2, 4))
    if trial % 3 == 0:                      # tied distances
        x1 = rng.integers(0, 4, n).astype(float)
        x2 = rng.integers(0, 4, n).astype(float)
    else:
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return x1, x2, rng.integers(0, num_classes, n).astype(np.int64), num_classes


def main():
    rng = np.random.default_rng(0)
    failures = []
    for trial in range(TRIALS):
        x1, x2, y, num_classes = random_problem(rng, trial)
        counts = np.bincount(y, minlength=num_classes)

        gain, cx1, cx2, r = find_best_split(x1, x2, y, counts)
        best_gain, best_circles = brute_force(x1, x2, y, num_classes)

        inside = (x1 - cx1) ** 2 + (x2 - cx2) ** 2 < r * r
        realised = information_gain(np.bincount(y[inside], minlength=num_classes),
                                    np.bincount(y[~inside], minlength=num_classes))

        problems = []
        if abs(gain - best_gain) > TOL:
            problems.append(f"gain {gain:.9f} != brute force {best_gain:.9f}")
        if gain > 0.0:
            if not any(abs(cx1 - a) < TOL and abs(cx2 - b) < TOL and abs(r - rr) < TOL
                       for a, b, rr in best_circles):
                problems.append(f"circle ({cx1}, {cx2}, r={r}) is not an optimal one")
            if abs(realised - gain) > TOL:
                problems.append(f"circle realises {realised:.9f}, not {gain:.9f}")
        if problems:
            failures.append(f"trial {trial}: " + "; ".join(problems))

    for failure in failures:
        print("FAIL", failure)
    print(f"{TRIALS - len(failures)}/{TRIALS} random problems agree with brute force")
    assert not failures, f"{len(failures)} mismatches"
    print("PASS")


if __name__ == "__main__":
    main()
