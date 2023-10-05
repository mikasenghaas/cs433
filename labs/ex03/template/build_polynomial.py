# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    # Create augmented features by iterating over degrees
    augmented_features = np.hstack(
        [(x**d).reshape(-1, 1) for d in range(0, degree + 1)]
    )

    return augmented_features
