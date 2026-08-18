# type-tests for `linalg/_decomp_cossin.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.linalg import cossin

###

type _F64ish = np.float64 | Any
type _C128ish = np.complex128 | Any

_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]

_i64_2d: onp.Array2D[np.int64]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]
_c128_nd: onp.ArrayND[np.complex128]

_blocks: list[onp.Array2D[np.float64]]

###
# cossin

assert_type(cossin(_f32_2d), tuple[onp.Array2D[_F64ish], onp.Array2D[_F64ish], onp.Array2D[_F64ish]])
assert_type(cossin(_f64_2d), tuple[onp.Array2D[_F64ish], onp.Array2D[_F64ish], onp.Array2D[_F64ish]])
assert_type(cossin(_i64_2d), tuple[onp.Array2D[_F64ish], onp.Array2D[_F64ish], onp.Array2D[_F64ish]])
assert_type(cossin(_py_f_2d), tuple[onp.Array2D[_F64ish], onp.Array2D[_F64ish], onp.Array2D[_F64ish]])
assert_type(cossin(_blocks), tuple[onp.Array2D[_F64ish], onp.Array2D[_F64ish], onp.Array2D[_F64ish]])
assert_type(cossin(_f64_nd), tuple[onp.ArrayND[_F64ish], onp.ArrayND[_F64ish], onp.ArrayND[_F64ish]])

assert_type(cossin(_c64_2d), tuple[onp.Array2D[_C128ish], onp.Array2D[_F64ish], onp.Array2D[_C128ish]])
assert_type(cossin(_c128_2d), tuple[onp.Array2D[_C128ish], onp.Array2D[_F64ish], onp.Array2D[_C128ish]])
assert_type(cossin(_py_c_2d), tuple[onp.Array2D[_C128ish], onp.Array2D[_F64ish], onp.Array2D[_C128ish]])
assert_type(cossin(_c128_nd), tuple[onp.ArrayND[_C128ish], onp.ArrayND[_F64ish], onp.ArrayND[_C128ish]])

assert_type(
    cossin(_f64_2d, separate=True),
    tuple[
        tuple[onp.Array2D[_F64ish], onp.Array2D[_F64ish]], onp.Array1D[_F64ish], tuple[onp.Array2D[_F64ish], onp.Array2D[_F64ish]]
    ],
)
assert_type(
    cossin(_c128_2d, separate=True),
    tuple[
        tuple[onp.Array2D[_C128ish], onp.Array2D[_C128ish]],
        onp.Array1D[_F64ish],
        tuple[onp.Array2D[_C128ish], onp.Array2D[_C128ish]],
    ],
)
assert_subtype[
    tuple[
        tuple[onp.ArrayND[_C128ish], onp.ArrayND[_C128ish]],
        onp.ArrayND[_F64ish],
        tuple[onp.ArrayND[_C128ish], onp.ArrayND[_C128ish]],
    ]
](cossin(_c128_nd, separate=True))
