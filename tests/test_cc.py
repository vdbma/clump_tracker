import numpy as np
import pytest
from numpy.testing import assert_array_equal

from clump_tracker import compute_adjacency_cartesian, compute_cc


def _compute_adjacency_cartesian_ref(indexes, x, y, z, max_distance):
    """
    computes and returns adjacency matrix
    this is a reference implementation for
    testing purposes only
    """
    adj = np.zeros((len(indexes), len(indexes)), dtype=bool)

    for i in range(len(indexes)):
        for j in range(len(indexes)):
            d2 = (
                (x[indexes[i][0]] - x[indexes[j][0]]) ** 2
                + (y[indexes[i][1]] - y[indexes[j][1]]) ** 2
                + (z[indexes[i][2]] - z[indexes[j][2]]) ** 2
            )
            adj[i, j] = d2 <= max_distance * max_distance
    return adj


def _compute_cc_ref(indexes, x, y, z, max_distance):
    """
    indexes : list of coordinates (array indexes)

    this function checks the distance between all pairs in indexes,
    if it is less than `max_distance` they are of the same clump

    this is a reference implementation for
    testing purposes only
    """

    deja_vus = np.zeros((len(indexes)), dtype=bool)
    a_visiter = np.zeros((len(indexes)), dtype=bool)
    composante_connexes = []

    adj = _compute_adjacency_cartesian_ref(indexes, x, y, z, max_distance)

    # ajoute les voisins de p0 dans a_visiter
    for i, _ in enumerate(indexes):
        if not deja_vus[i]:
            composante_connexes.append([i])
        a_visiter = adj[i]
        a_visiter[i] = False
        deja_vus[i] = True
        a_visiter = np.logical_and(a_visiter, np.logical_not(deja_vus))

        while np.sum(a_visiter) > 0:  # there are still people to visit
            for j, _ in enumerate(indexes):
                if a_visiter[j] and not deja_vus[j]:
                    composante_connexes[-1].append(j)
                    a_visiter += adj[j]
                    a_visiter[j] = False
                    deja_vus[j] = True
                    a_visiter = np.logical_and(a_visiter, np.logical_not(deja_vus))

        if np.sum(deja_vus) == len(indexes):
            break

    return composante_connexes


@pytest.fixture(
    params=[
        [[0, 0, 0], [0, 1, 0], [1, 0, 1]],
        [[0, 0, 0], [49, 19, 4]],
        [[i, 0, 0] for i in range(50)] + [[i, 0, 4] for i in range(50)],
    ]
)
def indexes(request):
    return request.param


@pytest.fixture(params=[np.float32, np.float64, float])
def dtype(request):
    return request.param


def test_adjacency(indexes, dtype):
    x = np.linspace(0, 10, 50, dtype=dtype)
    y = np.linspace(0, 5, 20, dtype=dtype)
    z = np.linspace(0, 1, 5, dtype=dtype)

    assert_array_equal(
        compute_adjacency_cartesian(indexes, x, y, z, 1.0),
        _compute_adjacency_cartesian_ref(indexes, x, y, z, 1.0),
    )


def test_cc(indexes, dtype):
    x = np.linspace(0, 10, 50, dtype=dtype)
    y = np.linspace(0, 5, 20, dtype=dtype)
    z = np.linspace(0, 1, 5, dtype=dtype)

    expected = _compute_cc_ref(indexes, x, y, z, 1.0)
    actual = compute_cc(indexes, x, y, z, 1.0, "cartesian")
    assert expected == actual


@pytest.mark.xfail
def test_cc_not_implemented(indexes, dtype):
    x = np.linspace(0, 10, 50, dtype=dtype)
    y = np.linspace(0, 5, 20, dtype=dtype)
    z = np.linspace(0, 1, 5, dtype=dtype)

    assert_array_equal(
        compute_cc(indexes, x, y, z, 1.0, "polar"),
        _compute_cc_ref(indexes, x, y, z, 1.0),
    )


@pytest.fixture(
    params=[[11.0, [[0, 1, 2]]], [2.0, [[0, 1], [2]]], [0.0, [[0], [1], [2]]]]
)
def cc_params(request):
    return request.param


def test_cc_ref(cc_params):
    indexes = [[i, 0, 0] for i in range(3)]
    x = np.array([0, 1, 10])
    y = np.array([0])
    z = np.array([0])
    d, expected = cc_params

    actual = _compute_cc_ref(indexes, x, y, z, d)
    assert actual == expected


def test_cc_cartesian(cc_params, dtype):
    indexes = [[i, 0, 0] for i in range(3)]
    x = np.array([0, 1, 10], dtype=dtype)
    y = np.array([0], dtype=dtype)
    z = np.array([0], dtype=dtype)
    d, expected = cc_params

    actual = compute_cc(indexes, x, y, z, d, "cartesian")
    assert actual == expected
