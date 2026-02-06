from __future__ import annotations

import numpy as np

from clump_tracker import compute_cc
from clump_tracker.clumps import Clump

__all__ = ["find_clumps"]

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import dtype, ndarray


def find_coordinates(
    data: dict[str, ndarray[tuple[int, int, int]]],
    x: ndarray[tuple[int, int, int]],
    y: ndarray[tuple[int, int, int]],
    z: ndarray[tuple[int, int, int]],
    mask: ndarray[tuple[int, int, int], dtype[bool]],
) -> ndarray[tuple[int, 3]]:
    out = np.array(np.nonzero(mask)).T
    return out


def find_clumps(
    data: dict[str, ndarray[tuple[int, int, int]]],
    x: ndarray[tuple[int, int, int]],
    y: ndarray[tuple[int, int, int]],
    z: ndarray[tuple[int, int, int]],
    dx: ndarray[tuple[int, int, int]],
    dy: ndarray[tuple[int, int, int]],
    dz: ndarray[tuple[int, int, int]],
    max_distance: float,
    mask: ndarray[tuple[int, int, int], dtype[bool]],
) -> list[Clump]:
    coordinates = find_coordinates(data, x, y, z, mask)

    cc = compute_cc(list(coordinates), x, y, z, max_distance, "cartesian")

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    ddx, ddy, ddz = np.meshgrid(dx, dy, dz, indexing="ij")

    if len(y) == 1:
        _dy = np.array([1.0])
    else:
        _dy = dy
    if len(z) == 1:
        _dz = np.array([1.0])
    else:
        _dz = dz
    ddx, ddy, ddz = np.meshgrid(dx, _dy, _dz, indexing="ij")

    dv = ddx * ddy * ddz

    clumps = []
    for c in cc:
        x = 0
        y = 0
        z = 0
        vx = 0
        vy = 0
        vz = 0
        mass = 0
        ncells = 0
        max_density = 0
        area = 0
        for idx in c:
            idx_data = coordinates[idx]
            m = data["RHO"][*idx_data] * dv[*idx_data]
            mass += m

            x += xx[*idx_data] * m
            y += yy[*idx_data] * m
            z += zz[*idx_data] * m

            vx += data["VX1"][*idx_data] * m
            if "VX2" in data:
                vy += data["VX2"][*idx_data] * m
            if "VX3" in data:
                vz += data["VX3"][*idx_data] * m

            max_density = max(max_density, data["RHO"][*idx_data])
            ncells += 1
            area += dv[*idx_data]
        x /= mass
        y /= mass
        z /= mass
        vx /= mass
        vy /= mass
        vz /= mass

        clumps.append(Clump(x, y, z, vx, vy, vz, mass, ncells, area, max_density))
    return clumps
