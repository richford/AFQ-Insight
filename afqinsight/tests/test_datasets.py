from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from afqinsight.datasets import make_group_classification, make_group_regression

from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_raises


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("return_idx", [True, False])
def test_make_group_sparse_classification(shuffle, return_idx):
    weights = [0.10, 0.25]

    clf_res = make_group_classification(
        n_samples=100,
        n_groups=10,
        n_features_per_group=8,
        n_informative_groups=5,
        n_informative_per_group=2,
        n_redundant_per_group=1,
        n_repeated_per_group=1,
        n_classes=3,
        n_clusters_per_class=1,
        hypercube=False,
        shift=None,
        scale=None,
        weights=weights,
        random_state=0,
        shuffle=shuffle,
        useful_indices=return_idx,
    )

    if return_idx:
        X, y, groups, idx = clf_res
        assert np.sum(idx) == 20
    else:
        X, y, groups = clf_res

    assert weights == [0.1, 0.25]  # nosec
    assert X.shape == (100, 80), "X shape mismatch"  # nosec
    assert y.shape == (100,), "y shape mismatch"  # nosec
    assert groups.shape == (80,), "groups shape mismatch"  # nosec
    assert np.unique(y).shape == (3,), "Unexpected number of classes"  # nosec
    assert sum(y == 0) == 10, "Unexpected number of samples in class #0"  # nosec
    assert sum(y == 1) == 25, "Unexpected number of samples in class #1"  # nosec
    assert sum(y == 2) == 65, "Unexpected number of samples in class #2"  # nosec

    # Test for n_features > 30
    X, y, groups = make_group_classification(
        n_samples=2000,
        n_groups=11,
        n_features_per_group=3,
        n_informative_groups=11,
        n_informative_per_group=3,
        n_redundant_per_group=0,
        n_repeated_per_group=0,
        n_clusters_per_class=1,
        n_classes=3,
        hypercube=True,
        flip_y=0,
        shift=None,
        scale=0.5,
        random_state=0,
    )

    assert X.shape == (2000, 33), "X shape mismatch"  # nosec
    assert y.shape == (2000,), "y shape mismatch"  # nosec
    assert groups.shape == (33,), "groups shape mismatch"  # nosec

    assert (
        np.unique(X.view([("", X.dtype)] * X.shape[1]))
        .view(X.dtype)
        .reshape(-1, X.shape[1])
        .shape[0]
        == 2000
    ), "Unexpected number of unique rows"  # nosec

    assert (
        np.unique(X.view([("", X.dtype)] * X.shape[1]))
        .view(X.dtype)
        .reshape(-1, X.shape[1])
        .shape[0]
        == 2000
    ), "Unexpected number of unique rows"

    assert_raises(
        ValueError,
        make_group_classification,
        n_groups=20,
        n_features_per_group=2,
        n_informative_groups=20,
        n_redundant_per_group=1,
        n_repeated_per_group=2,
    )
    assert_raises(ValueError, make_group_classification, weights=weights, n_classes=5)


def test_make_group_regression():
    X, y, groups, coefs = make_group_regression(
        n_samples=100,
        n_groups=2,
        n_informative_groups=1,
        n_features_per_group=5,
        n_informative_per_group=2,
        effective_rank=5,
        coef=True,
        random_state=0,
    )

    assert X.shape == (100, 10), "X shape mismatch"  # nosec
    assert y.shape == (100,), "y shape mismatch"  # nosec
    assert groups.shape == (10,), "groups shape mismatch"  # nosec
    assert coefs.shape == (10,), "coef shape mismatch"
    assert sum(coefs != 0.0) == 2, "Unexpected number of informative features"

    X, y, groups, coefs = make_group_regression(
        n_samples=100,
        n_groups=2,
        n_informative_groups=2,
        n_features_per_group=5,
        n_informative_per_group=3,
        effective_rank=10,
        noise=1.0,
        coef=True,
        shuffle=True,
        random_state=0,
    )

    assert X.shape == (100, 10), "X shape mismatch"  # nosec
    assert y.shape == (100,), "y shape mismatch"  # nosec
    assert groups.shape == (10,), "groups shape mismatch"  # nosec
    assert coefs.shape == (10,), "coef shape mismatch"
    assert sum(coefs != 0.0) == 6, "Unexpected number of informative features"

    # Test that y ~= np.dot(X, coefs) + bias + N(0, 1.0).
    assert_array_almost_equal(np.std(y - np.dot(X, coefs)), 1.0, decimal=1)

    # Test with small number of features.
    X, y, groups = make_group_regression(
        n_samples=100,
        n_groups=1,
        n_informative_groups=1,
        n_features_per_group=1,
        n_informative_per_group=1,
    )  # n_informative=3
    assert X.shape == (100, 1)
