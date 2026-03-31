from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

ints_1d: list[int]
ints_2d: list[list[int]]

floats_1d: list[float]
floats_2d: list[list[float]]

f32_1d: onp.Array1D[np.float32]
f32_2d: onp.Array2D[np.float32]
f32_nd: onp.ArrayND[np.float32]

###
# Bounds

assert_type(Bounds(0, 1), Bounds[tuple[int], np.int_])
assert_type(Bounds(ints_1d, ints_1d), Bounds[tuple[int], np.int_])
assert_type(Bounds(ints_2d, ints_2d), Bounds[tuple[Any, ...], np.int_])

assert_type(Bounds(0.0, 1.0), Bounds[tuple[int], np.float64])
assert_type(Bounds(floats_1d, floats_1d), Bounds[tuple[int], np.float64])
assert_type(Bounds(floats_2d, floats_2d), Bounds[tuple[Any, ...], np.float64])

assert_type(Bounds(f32_1d, f32_1d), Bounds[tuple[int], np.float32])
assert_type(Bounds(f32_1d, f32_2d), Bounds[tuple[int, int], np.float32])
assert_type(Bounds(f32_1d, f32_nd), Bounds[tuple[Any, ...], np.float32])

assert_type(Bounds(f32_2d, f32_1d), Bounds[tuple[int, int], np.float32])
assert_type(Bounds(f32_2d, f32_2d), Bounds[tuple[int, int], np.float32])
assert_type(Bounds(f32_2d, f32_nd), Bounds[tuple[Any, ...], np.float32])

assert_type(Bounds(f32_nd, f32_1d), Bounds[tuple[Any, ...], np.float32])
assert_type(Bounds(f32_nd, f32_2d), Bounds[tuple[Any, ...], np.float32])
assert_type(Bounds(f32_nd, f32_nd), Bounds[tuple[Any, ...], np.float32])

###
# LinearConstraint

assert_type(LinearConstraint([[1.0, 2.0]], 0.0, 1.0), LinearConstraint)
assert_type(LinearConstraint([[1.0, 2.0]]), LinearConstraint)

###
# NonlinearConstraint

def _con(x: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]: ...

assert_type(NonlinearConstraint(_con, 0.0, 1.0), NonlinearConstraint)
