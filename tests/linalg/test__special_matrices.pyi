from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import hankel, toeplitz

_py_i_1d: list[int]
_py_i_2d: list[list[int]]
_py_i_3d: list[list[list[int]]]
_py_f_1d: list[float]
_py_c_1d: list[complex]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

###

# toeplitz
assert_type(toeplitz(_py_i_1d), onp.Array2D[np.int_])
assert_type(toeplitz(_py_f_1d), onp.Array2D[np.float64])
assert_type(toeplitz(_py_c_1d), onp.Array2D[np.complex128])
assert_type(toeplitz(_f64_1d), onp.Array2D[np.float64])
assert_type(toeplitz(_f64_nd), onp.Array2D[np.float64])
assert_type(toeplitz(_f64_1d, _f64_1d), onp.Array2D[np.float64])
assert_type(toeplitz(_f64_1d, _f64_nd), onp.Array2D[np.float64])
assert_type(toeplitz(_f64_nd, _f64_nd), onp.Array2D[np.float64])
assert_type(toeplitz(_py_i_2d), onp.ArrayND[np.int_])
assert_type(toeplitz(_py_i_3d), onp.ArrayND[np.int_])

# hankel
assert_type(hankel([0]), onp.Array2D[np.int_])
assert_type(hankel([0.0]), onp.Array2D[np.float64])
assert_type(hankel([0j]), onp.Array2D[np.complex128])
assert_type(hankel(_f64_1d), onp.Array2D[np.float64])
assert_type(hankel(_f64_nd), onp.Array2D[np.float64])
assert_type(hankel(_f64_1d, _f64_1d), onp.Array2D[np.float64])
assert_type(hankel(_f64_1d, _f64_nd), onp.Array2D[np.float64])
assert_type(hankel(_f64_nd, _f64_nd), onp.Array2D[np.float64])
assert_type(hankel(_py_i_2d), onp.Array2D[np.int_])  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
assert_type(hankel(_py_i_3d), onp.Array2D[np.int_])  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
