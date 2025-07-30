# ruff: noqa: ERA001

from typing import Any, assert_type

import numpy as np

from scipy.optimize import Bounds

ints_1d: list[int]
ints_2d: list[list[int]]

floats_1d: list[float]
floats_2d: list[list[float]]

f32_1d: np.ndarray[tuple[int], np.dtype[np.float32]]
f32_2d: np.ndarray[tuple[int, int], np.dtype[np.float32]]

###
# Bounds

assert_type(Bounds(0, 1), Bounds[tuple[int], np.int_])
assert_type(Bounds(ints_1d, ints_1d), Bounds[tuple[int], np.int_])
assert_type(Bounds(ints_2d, ints_2d), Bounds[tuple[Any, ...], np.int_])

assert_type(Bounds(0.0, 1.0), Bounds[tuple[int], np.float64])
assert_type(Bounds(floats_1d, floats_1d), Bounds[tuple[int], np.float64])
assert_type(Bounds(floats_2d, floats_2d), Bounds[tuple[Any, ...], np.float64])

# NOTE: these two assertions only pass on numpy 2.1+, so we instead check for assignability
# assert_type(Bounds(f32_1d, f32_1d), Bounds[tuple[int], np.float32])
# assert_type(Bounds(f32_2d, f32_2d), Bounds[tuple[int, int], np.float32])
_0: Bounds[tuple[int], np.float32] = Bounds(f32_1d, f32_1d)
_1: Bounds[tuple[int, int], np.float32] = Bounds(f32_2d, f32_2d)

###
# LinearConstraint
# TODO(joreham): type tests

###
# NonlinearConstraint
# TODO(joreham): type tests
