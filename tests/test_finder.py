from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pytest

from clump_tracker.finder import find_clumps


@dataclass
class FakeVTK:
    xE: np.ndarray
    yE: np.ndarray
    zE: np.ndarray
    data: dict[np.ndarray]

    @cached_property
    def x(self):
        return np.convolve([0.5, 0.5], self.xE, mode="valid")

    @cached_property
    def y(self):
        return np.convolve([0.5, 0.5], self.yE, mode="valid")

    @cached_property
    def z(self):
        return np.convolve([0.5, 0.5], self.zE, mode="valid")


def _data(Nx, Ny, Nz):
    return {
        "RHO": np.zeros((Nx, Ny, Nz)),
        "VX1": np.zeros((Nx, Ny, Nz)),
        "VX2": np.zeros((Nx, Ny, Nz)),
    }


def n2Pixel(n, Nx=1024, Ny=1024, Nz=1):
    V = FakeVTK(
        np.linspace(-10, 10, Nx + 1),
        np.linspace(-10, 10, Ny + 1),
        np.linspace(-0.5, 0.5, Nz + 1),
        _data(Nx, Ny, Nz),
    )

    for k in range(n):
        for kp in range(n):
            V.data["RHO"][Nx // n * k + Nx // (2 * n), Ny // n * kp + Ny // (2 * n)] = (
                1.0
            )

    return V


def oneBump(Nx=1024, Ny=1024, Nz=1):
    V = FakeVTK(
        np.linspace(-10, 10, Nx + 1),
        np.linspace(-10, 10, Ny + 1),
        np.linspace(-0.5, 0.5, Nz + 1),
        _data(Nx, Ny, Nz),
    )

    xx, yy, zz = np.meshgrid(V.x, V.y, V.z, indexing="ij")

    V.data["RHO"] += np.exp(-(xx * xx + yy * yy)) * 2

    return V


def condition(data):
    return np.asarray(data["RHO"] > 0.5)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 10])
def test_find_clumps_pixels(n):
    V = n2Pixel(n, 128, 128)
    coords = [V.x, V.y, V.z]
    dcoords = [np.ediff1d(_) for _ in [V.xE, V.yE, V.zE]]
    max_distance = 1.1 * np.ediff1d(V.xE)[0]
    clumps = find_clumps(
        V.data, *coords, *dcoords, float(max_distance), condition=condition
    )
    assert len(clumps) == n * n


def test_find_clumps_oneBump():
    V = oneBump(128, 128)
    coords = [V.x, V.y, V.z]
    dcoords = [np.ediff1d(_) for _ in [V.xE, V.yE, V.zE]]
    max_distance = 1.1 * np.ediff1d(V.xE)[0]
    clumps = find_clumps(
        V.data, *coords, *dcoords, float(max_distance), condition=condition
    )
    assert len(clumps) == 1
