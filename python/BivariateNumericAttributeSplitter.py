from Splitter import Splitter

from numba import jit
import numpy as np

import matplotlib.pyplot as plt

@jit
def information_gain(left_counts, right_counts):
    total = left_counts + right_counts
    sum_total = np.sum(total)
    if sum_total == 0:
        return 0.0
    sum_left = np.sum(left_counts)
    if sum_left == 0:
        return 0.0
    sum_right = np.sum(right_counts)
    if sum_right == 0:
        return 0.0
    log2 = np.log(2)
    entropy_total = -np.nansum(total * np.log(total))
    entropy_left = -np.nansum(left_counts * np.log(left_counts))
    entropy_right = -np.nansum(right_counts * np.log(right_counts))
    return (entropy_total + sum_total * np.log(sum_total) -
            (entropy_left + sum_left * np.log(sum_left) +
             entropy_right + sum_right * np.log(sum_right))) / (sum_total * log2)

@jit
def find_best_split(x1_values, x2_values, y, initial_counts):
    """
    Finds the best circular split for one pair of attributes.

    For every instance used as a candidate centre, the instances are ordered by
    their (squared) Euclidean distance to that centre in the (x1, x2) plane. As
    the radius grows, instances move from the "outside" (right) branch into the
    "inside" (left) branch one distinct-distance group at a time. At every
    boundary between two consecutive distinct distances a candidate split is
    scored with information gain. Among all circles giving the same partition,
    the representative radius is the midpoint (rmin + rmax) / 2 of the two
    consecutive distances, mirroring the midpoint rule used for univariate splits.

    Returns (best_quality, centre_x1, centre_x2, radius) for this attribute pair.
    """
    n = len(y)
    best_quality = 0.0
    best_x1 = 0.0
    best_x2 = 0.0
    best_r = 0.0

    dist = np.empty(n, dtype=np.float64)

    for c in range(n):
        cx1 = x1_values[c]
        cx2 = x2_values[c]

        # Squared distance from centre c to every instance (monotonic in radius,
        # so it preserves ordering; sqrt is only taken at the two boundaries).
        for i in range(n):
            d1 = x1_values[i] - cx1
            d2 = x2_values[i] - cx2
            dist[i] = d1 * d1 + d2 * d2

        order = np.argsort(dist)

        # Start with everything outside the circle (radius 0).
        counts_inside = np.zeros_like(initial_counts)
        counts_outside = initial_counts.copy()

        prev_dist = -1.0
        for idx in range(n):
            i = order[idx]
            d = dist[i]
            if d > prev_dist:
                if prev_dist >= 0.0:
                    # inside = instances with distance <= prev_dist (already added),
                    # outside = the remaining instances (distance >= d). Both non-empty.
                    quality = information_gain(counts_inside, counts_outside)
                    if quality > best_quality:
                        best_quality = quality
                        best_x1 = cx1
                        best_x2 = cx2
                        # Representative radius: midpoint of the two consecutive
                        # actual (non-squared) distances rmin and rmax.
                        best_r = (np.sqrt(prev_dist) + np.sqrt(d)) / 2.0
                prev_dist = d
            label = y[i]
            counts_inside[label] += 1
            counts_outside[label] -= 1

    return best_quality, best_x1, best_x2, best_r

class BivariateNumericAttributeSplitter(Splitter):
    def __init__(self, args):
        self.attribute1_index = None
        self.attribute2_index = None
        self.x1 = None
        self.x2 = None
        self.r = None
        self.split_quality = 0.0
        self.num_classes = args["num_classes"]
        self.attribute_names = args.get("attributes")

    def find_split(self, X, y):
        initial_counts = np.bincount(y, minlength=self.num_classes)
        num_attributes = X.shape[1]
        # Every unordered pair of attributes defines a 2D plane. A circle is
        # symmetric in its two axes, so (a1, a2) and (a2, a1) are equivalent and
        # only pairs with a1 < a2 need to be examined.
        for a1 in range(num_attributes):
            for a2 in range(a1 + 1, num_attributes):
                quality, cx1, cx2, r = find_best_split(
                    X[:, a1], X[:, a2], y, initial_counts)
                if quality > self.split_quality:
                    self.split_quality = quality
                    self.attribute1_index = a1
                    self.attribute2_index = a2
                    self.x1 = cx1
                    self.x2 = cx2
                    self.r = r

    def first_subset(self, x):
        # An instance is in the first (left) subset if it lies inside the circle,
        # i.e. its distance to the centre is smaller than the radius. Written with
        # squared distance so it works for a single instance and, via broadcasting,
        # for the whole transposed feature matrix at once.
        d1 = x[self.attribute1_index] - self.x1
        d2 = x[self.attribute2_index] - self.x2
        return d1 * d1 + d2 * d2 < self.r * self.r
        
    def split_quality(self):
        return self.split_quality

    def __str__(self, left_branch, decimal_places):
        if self.attribute_names is not None:
            attr1_name = self.attribute_names[self.attribute1_index]
            attr2_name = self.attribute_names[self.attribute2_index]
        else:
            attr1_name = f"x[{self.attribute1_index}]"
            attr2_name = f"x[{self.attribute2_index}]"
        op = " inside " if left_branch else " outside "
        return "Attributes: " + attr1_name + "," + attr2_name + \
            op + "circle with center (" + \
            f"{attr1_name} = {self.x1:.{decimal_places}f}" + "," + \
            f"{attr2_name} = {self.x2:.{decimal_places}f}" + ") and radius " + \
            f"{self.r:.{decimal_places}f}"

    def plot_split(self, X, y, decimal_places):
        x_min, x_max = X[:, self.attribute1_index].min() - 1, X[:, self.attribute1_index].max() + 1
        y_min, y_max = X[:, self.attribute2_index].min() - 1, X[:, self.attribute2_index].max() + 1
        fig, ax = plt.subplots()
        ax.add_patch(plt.Circle((self.x1, self.x2), self.r))
        plt.scatter(X[:, self.attribute1_index], X[:, self.attribute2_index], c=y, s=20)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        if self.attribute_names is not None:
            attr1_name = self.attribute_names[self.attribute1_index]
            attr2_name = self.attribute_names[self.attribute2_index]
        else:
            attr1_name = f"x[{self.attribute1_index}]"
            attr2_name = f"x[{self.attribute2_index}]"
        plt.suptitle("Information gain: " + f"{self.split_quality:.{decimal_places}f}")
        ax.set_xlabel(attr1_name)
        ax.set_ylabel(attr2_name)
        plt.show()

