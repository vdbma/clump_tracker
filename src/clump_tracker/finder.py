import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray


def find_coordinates(
    data: dict[str, ndarray[tuple[int, int, int]]],
    x: ndarray[tuple[int, int, int]],
    y: ndarray[tuple[int, int, int]],
    z: ndarray[tuple[int, int, int]],
    q: float = 3 / 2,
    Omega: float = 1,
    gamma: float = -1.0,
    cs: float = -1.0,
    conditions=None,
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

    return np.array(np.nonzero(np.logical_and(E_tot < 0, div_v < 0))).T
