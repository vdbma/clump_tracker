from __future__ import annotations


__all__ = [
    "compute_adjacency_cartesian",
    "compute_cc",
]

from clump_tracker._core import (
    compute_adjacency_cartesian_f32,
    compute_adjacency_cartesian_f64,
    compute_cc_f32,
    compute_cc_f64,
)
import numpy as np

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from numpy import ndarray, dtype


def compute_cc(
    indexes: list,
    x: ndarray[tuple[int, int, int], dtype[Union[float, np.float32, np.float64]]],
    y: ndarray[tuple[int, int, int], dtype[Union[float, np.float32, np.float64]]],
    z: ndarray[tuple[int, int, int], dtype[Union[float, np.float32, np.float64]]],
    d: Union[float, np.float32, np.float64],
    geometry: str,
) -> list:
    if x.dtype == np.float32:
        return compute_cc_f32(indexes, x, y, z, d, geometry)
    elif x.dtype in [float, np.float64]:
        return compute_cc_f64(indexes, x, y, z, d, geometry)
    else:
        raise TypeError("Only supports f32 and f64.")


def compute_adjacency_cartesian(
    indexes: list,
    x: ndarray[tuple[int, int, int], dtype[Union[float, np.float32, np.float64]]],
    y: ndarray[tuple[int, int, int], dtype[Union[float, np.float32, np.float64]]],
    z: ndarray[tuple[int, int, int], dtype[Union[float, np.float32, np.float64]]],
    d: Union[float, np.float32, np.float64],
) -> list:
    if x.dtype == np.float32:
        return compute_adjacency_cartesian_f32(indexes, x, y, z, d)
    elif x.dtype in [float, np.float64]:
        return compute_adjacency_cartesian_f64(indexes, x, y, z, d)
    else:
        raise TypeError("Only supports f32 and f64.")
