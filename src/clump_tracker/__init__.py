from __future__ import annotations


__all__ = ["compute_adjacency_cartesian", "compute_cc", "gradient"]

from clump_tracker._core import (
    compute_adjacency_cartesian_f32,
    compute_adjacency_cartesian_f64,
    compute_cc_f32,
    compute_cc_f64,
    gradient_f32,
    gradient_f64,
)
import numpy as np

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from numpy import ndarray, dtype


def compute_cc(
    indexes: list[tuple[int, int, int], dtype[int]],
    x: ndarray[tuple[int], dtype[Union[float, np.float32, np.float64]]],
    y: ndarray[tuple[int], dtype[Union[float, np.float32, np.float64]]],
    z: ndarray[tuple[int], dtype[Union[float, np.float32, np.float64]]],
    d: Union[float, np.float32, np.float64],
    geometry: str,
) -> list[list[tuple[int, int, int], dtype[int]]]:
    if x.dtype == np.float32:
        return compute_cc_f32(indexes, x, y, z, d, geometry)
    elif x.dtype in [float, np.float64]:
        return compute_cc_f64(indexes, x, y, z, d, geometry)
    else:
        raise TypeError("Only supports f32 and f64.")


def compute_adjacency_cartesian(
    indexes: list[tuple[int, int, int], dtype[int]],
    x: ndarray[tuple[int], dtype[Union[float, np.float32, np.float64]]],
    y: ndarray[tuple[int], dtype[Union[float, np.float32, np.float64]]],
    z: ndarray[tuple[int], dtype[Union[float, np.float32, np.float64]]],
    d: Union[float, np.float32, np.float64],
) -> list[list[tuple[int, int, int], dtype[int]]]:
    if x.dtype == np.float32:
        return compute_adjacency_cartesian_f32(indexes, x, y, z, d)
    elif x.dtype in [float, np.float64]:
        return compute_adjacency_cartesian_f64(indexes, x, y, z, d)
    else:
        raise TypeError("Only supports f32 and f64.")


def gradient(
    field: ndarray[dtype[Union[float, np.float32, np.float64]]],
    x: ndarray[tuple[int], dtype[Union[float, np.float32, np.float64]]],
    axis: int,
) -> ndarray[dtype[Union[float, np.float32, np.float64]]]:
    if x.dtype == np.float32:
        return gradient_f32(field, x, axis)
    elif x.dtype in [float, np.float64]:
        return gradient_f64(field, x, axis)
    else:
        raise TypeError("Only supports f32 and f64.")
