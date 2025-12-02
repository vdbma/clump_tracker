from clump_tracker import gradient
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_tests():
    assert True


def test_gradient_1D_uniform():
    field = np.arange(10, dtype=float) + 5
    x = np.arange(10, dtype=float)
    axis = 0
    assert_array_equal(gradient(field, x, axis), np.gradient(field, x, axis=axis))


def test_gradient_1D_non_uniform():
    field = np.arange(10, dtype=float) + 5
    axis = 0
    x = np.geomspace(1, 10, len(field), dtype=float)
    assert_array_almost_equal(
        gradient(field, x, axis), np.gradient(field, x, axis=axis)
    )


@pytest.mark.parametrize("axis", range(3))
def test_gradient_3D_uniform(axis):
    field = np.arange(9 * 10 * 11, dtype=float).reshape((9, 10, 11)) + 5
    x = np.arange(field.shape[axis], dtype=float)
    assert_array_equal(gradient(field, x, axis), np.gradient(field, x, axis=axis))


@pytest.mark.parametrize("axis", range(3))
def test_gradient_3D_non_uniform(axis):
    field = np.arange(9 * 10 * 11, dtype=float).reshape((9, 10, 11)) + 5
    x = np.geomspace(1, 10, field.shape[axis], dtype=float)
    assert_array_almost_equal(
        gradient(field, x, axis), np.gradient(field, x, axis=axis)
    )
