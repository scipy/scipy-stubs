from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import kron, toeplitz

arr_f8_1d: onp.Array1D[np.float64]
arr_f8_2d: onp.Array2D[np.float64]
arr_f8_nd: onp.Array[tuple[int, ...], np.float64]

###
# kron
# > deprecated
f_kron_f8_2d = kron(arr_f8_2d, arr_f8_2d)  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
f_kron_f8_nd = kron(arr_f8_nd, arr_f8_nd)  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
assert_type(f_kron_f8_2d, onp.Array2D[np.float64])
assert_type(f_kron_f8_nd, onp.Array[onp.AtLeast2D, np.float64])
###

###
# toeplitz
# > non-deprecated overloads
assert_type(toeplitz([0]), onp.Array2D[np.int_])
assert_type(toeplitz([0.0]), onp.Array2D[np.float64])
assert_type(toeplitz([0j]), onp.Array2D[np.complex128])
assert_type(toeplitz(arr_f8_1d), onp.Array2D[np.float64])
assert_type(toeplitz(arr_f8_nd), onp.Array2D[np.float64])
assert_type(toeplitz(arr_f8_1d, arr_f8_1d), onp.Array2D[np.float64])
assert_type(toeplitz(arr_f8_1d, arr_f8_nd), onp.Array2D[np.float64])
assert_type(toeplitz(arr_f8_nd, arr_f8_nd), onp.Array2D[np.float64])
# > deprecated (raveled input)
assert_type(toeplitz([[0], [1]]), onp.Array2D[np.int_])  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
assert_type(toeplitz([[[0], [1]], [[2], [3]]]), onp.Array2D[np.int_])  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
###
