from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import hankel, toeplitz  # pyright: ignore[reportUnknownVariableType]

arr_f8_1d: onp.Array1D[np.float64]
arr_f8_2d: onp.Array2D[np.float64]
arr_f8_nd: onp.Array[tuple[int, ...], np.float64]

###
# toeplitz
# > 1d overloads
assert_type(toeplitz([0]), onp.Array2D[np.int_])
assert_type(toeplitz([0.0]), onp.Array2D[np.float64])
assert_type(toeplitz([0j]), onp.Array2D[np.complex128])
assert_type(toeplitz(arr_f8_1d), onp.Array2D[np.float64])
assert_type(toeplitz(arr_f8_nd), onp.Array2D[np.float64])
assert_type(toeplitz(arr_f8_1d, arr_f8_1d), onp.Array2D[np.float64])
assert_type(toeplitz(arr_f8_1d, arr_f8_nd), onp.Array2D[np.float64])
assert_type(toeplitz(arr_f8_nd, arr_f8_nd), onp.Array2D[np.float64])
# > >1d overloads
# https://github.com/python/mypy/issues/19109
# https://github.com/python/mypy/issues/20354
# https://github.com/microsoft/pyright/issues/11127
assert_type(toeplitz([[0], [1]]), onp.Array[tuple[int, int, int, tuple[Any, ...]], np.int_])  # type: ignore[assert-type, type-var]  # pyright: ignore[reportInvalidTypeForm]
assert_type(toeplitz([[[0], [1]], [[2], [3]]]), onp.Array[tuple[int, int, int, tuple[Any, ...]], np.int_])  # type: ignore[assert-type, type-var]  # pyright: ignore[reportInvalidTypeForm]
###

###
# hankel
# > non-deprecated overloads
assert_type(hankel([0]), onp.Array2D[np.int_])
assert_type(hankel([0.0]), onp.Array2D[np.float64])
assert_type(hankel([0j]), onp.Array2D[np.complex128])
assert_type(hankel(arr_f8_1d), onp.Array2D[np.float64])
assert_type(hankel(arr_f8_nd), onp.Array2D[np.float64])
assert_type(hankel(arr_f8_1d, arr_f8_1d), onp.Array2D[np.float64])
assert_type(hankel(arr_f8_1d, arr_f8_nd), onp.Array2D[np.float64])
assert_type(hankel(arr_f8_nd, arr_f8_nd), onp.Array2D[np.float64])
# > deprecated (raveled input)
assert_type(hankel([[0], [1]]), onp.Array2D[np.int_])  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
assert_type(hankel([[[0], [1]], [[2], [3]]]), onp.Array2D[np.int_])  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
###
