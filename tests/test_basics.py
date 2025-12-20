from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from clump_tracker import gradient


@pytest.fixture(params=[np.float32, np.float64, float])
def dtype(request):
    return request.param


@pytest.fixture(params=[0, 1, 2])
def axis(request):
    return request.param


def test_gradient_1D_uniform(dtype):
    field = np.arange(10, dtype=dtype) + 5
    x = np.arange(10, dtype=dtype)
    axis = 0
    assert_array_equal(gradient(field, x, axis), np.gradient(field, x, axis=axis))


def test_gradient_1D_non_uniform(dtype):
    field = np.arange(10, dtype=float) + 5
    axis = 0
    x = np.geomspace(1, 10, len(field), dtype=float)
    assert_array_almost_equal(
        gradient(field, x, axis), np.gradient(field, x, axis=axis)
    )


def test_gradient_3D_uniform(axis, dtype):
    field = np.arange(9 * 10 * 11, dtype=dtype).reshape((9, 10, 11)) + 5
    x = np.arange(field.shape[axis], dtype=dtype)
    assert_array_equal(gradient(field, x, axis), np.gradient(field, x, axis=axis))


def test_gradient_3D_non_uniform(axis, dtype):
    field = np.arange(9 * 10 * 11, dtype=dtype).reshape((9, 10, 11)) + 5
    x = np.geomspace(1, 10, field.shape[axis], dtype=dtype)
    if dtype == np.float32:
        prec = 3  # this is not a lot
    else:
        prec = 12
    assert_array_almost_equal(
        gradient(field, x, axis), np.gradient(field, x, axis=axis), decimal=prec
    )
