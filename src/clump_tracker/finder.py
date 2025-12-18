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
    q: float = 3 / 2,
    Omega: float = 1,
    gamma: float = -1.0,
    cs: float = -1.0,
    condition: Callable[
        [ndarray[tuple[int, int, int]]], ndarray[tuple[int, int, int], dtype[bool]]
    ]
    | None = None,
) -> ndarray[tuple[int, int, int]]:
    if gamma == -1 and "PRS" in data:
        raise ValueError("Please specify gamma.")
    elif gamma == -1 and cs == -1:
        raise ValueError("Please specify either gamma or cs.")

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    Ec = (
        0.5
        * data["RHO"]
        * (data["VX1"] * data["VX1"] + (data["VX2"] + q * Omega * xx) ** 2)
    )
    Ep = data["RHO"] * data["phiP"]

    if "PRS" in data:
        E_t = data["PRS"] / (gamma - 1)
    else:
        E_t = cs * cs * data["RHO"]
    E_tot = Ec + Ep + E_t

    div_v = np.gradient(data["VX1"], x, axis=0)
    if len(y) > 1:
        div_v += np.gradient(data["VX2"], y, axis=1)
    if len(z) > 1:
        div_v += np.gradient(data["VX3"], z, axis=2)

    mask = np.logical_and(E_tot < 0, div_v < 0)

    if condition is not None:
        mask = np.logical_and(mask, condition(data))

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
    *,
    q: float = 3 / 2,
    Omega: float = 1,
    gamma: float = -1.0,
    cs: float = -1.0,
    condition: Callable[
        [ndarray[tuple[int, int, int]]], ndarray[tuple[int, int, int], dtype[bool]]
    ]
    | None = None,
) -> list[Clump]:
    coordinates = find_coordinates(data, x, y, z, q, Omega, gamma, cs, condition)

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
