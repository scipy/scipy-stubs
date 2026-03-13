# type-tests for `linalg/_decomp_cholesky.pyi`

from typing import Any, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import cho_factor, cho_solve, cho_solve_banded, cholesky, cholesky_banded

_Complex2D: TypeAlias = onp.Array2D[npc.inexact]
_ComplexND: TypeAlias = onp.ArrayND[npc.inexact]

###

_bool_2d: onp.Array2D[np.bool_]
_i8_2d: onp.Array2D[np.int8]
_i16_2d: onp.Array2D[np.int16]
_i32_2d: onp.Array2D[np.int32]
_f16_2d: onp.Array2D[np.float16]
_f32_2d: onp.Array2D[np.float32]
_i64_2d: onp.Array2D[np.int64]
_f80_2d: onp.Array2D[np.float128]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_c64_2d: onp.Array2D[np.complex64]
_c160_2d: onp.Array2D[np.complex256]

_c128_1d: onp.Array1D[np.complex128]
_c128_2d: onp.Array2D[np.complex128]
_c128_3d: onp.Array3D[np.complex128]
_c128_nd: onp.ArrayND[np.complex128]

_number_2d: onp.Array2D[npc.number]

_py_f_2d: list[list[float]]
_py_f_3d: list[list[list[float]]]
_py_c_2d: list[list[complex]]
_py_c_3d: list[list[list[complex]]]

###
# cholesky

assert_type(cholesky(_bool_2d), onp.Array2D[np.float32])
assert_type(cholesky(_i8_2d), onp.Array2D[np.float32])
assert_type(cholesky(_i16_2d), onp.Array2D[np.float32])
assert_type(cholesky(_i32_2d), onp.Array2D[np.float64])
assert_type(cholesky(_i64_2d), onp.Array2D[np.float64])
assert_type(cholesky(_f16_2d), onp.Array2D[np.float32])
assert_type(cholesky(_f32_2d), onp.Array2D[np.float32])
assert_type(cholesky(_f64_2d), onp.Array2D[np.float64])
assert_type(cholesky(_f80_2d), onp.Array2D[np.float64])
assert_type(cholesky(_c64_2d), onp.Array2D[np.complex64])
assert_type(cholesky(_c128_2d), onp.Array2D[np.complex128])
assert_type(cholesky(_c160_2d), onp.Array2D[np.complex128])
assert_type(cholesky(_number_2d), onp.Array2D[Any])

cholesky(_f64_1d)  # type:ignore[type-var] # pyright:ignore[reportArgumentType,reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_type(cholesky(_f64_3d), onp.Array3D[np.float64])
assert_type(cholesky(_f64_nd), onp.ArrayND[np.float64])

assert_type(cholesky(_py_f_2d), onp.Array2D[np.float64])
assert_type(cholesky(_py_f_3d), onp.ArrayND[np.float64])
assert_type(cholesky(_py_c_2d), onp.Array2D[np.complex128])
assert_type(cholesky(_py_c_3d), onp.ArrayND[np.complex128])

###
# cho_factor (congruent with cholesky)

assert_type(cho_factor(_bool_2d), tuple[onp.Array2D[np.float32], bool])
assert_type(cho_factor(_i8_2d), tuple[onp.Array2D[np.float32], bool])
assert_type(cho_factor(_i16_2d), tuple[onp.Array2D[np.float32], bool])
assert_type(cho_factor(_i32_2d), tuple[onp.Array2D[np.float64], bool])
assert_type(cho_factor(_i64_2d), tuple[onp.Array2D[np.float64], bool])
assert_type(cho_factor(_f16_2d), tuple[onp.Array2D[np.float32], bool])
assert_type(cho_factor(_f32_2d), tuple[onp.Array2D[np.float32], bool])
assert_type(cho_factor(_f64_2d), tuple[onp.Array2D[np.float64], bool])
assert_type(cho_factor(_f80_2d), tuple[onp.Array2D[np.float64], bool])
assert_type(cho_factor(_c64_2d), tuple[onp.Array2D[np.complex64], bool])
assert_type(cho_factor(_c128_2d), tuple[onp.Array2D[np.complex128], bool])
assert_type(cho_factor(_c160_2d), tuple[onp.Array2D[np.complex128], bool])
assert_type(cho_factor(_number_2d), tuple[onp.Array2D[Any], bool])

cho_factor(_f64_1d)  # type:ignore[type-var] # pyright:ignore[reportArgumentType,reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_type(cho_factor(_f64_3d), tuple[onp.Array3D[np.float64], bool])
assert_type(cho_factor(_f64_nd), tuple[onp.ArrayND[np.float64], bool])

assert_type(cho_factor(_py_f_2d), tuple[onp.Array2D[np.float64], bool])
assert_type(cho_factor(_py_f_3d), tuple[onp.ArrayND[np.float64], bool])
assert_type(cho_factor(_py_c_2d), tuple[onp.Array2D[np.complex128], bool])
assert_type(cho_factor(_py_c_3d), tuple[onp.ArrayND[np.complex128], bool])

###
# cholesky_banded (same signature as cholesky)

assert_type(cholesky_banded(_bool_2d), onp.Array2D[np.float32])
assert_type(cholesky_banded(_i8_2d), onp.Array2D[np.float32])
assert_type(cholesky_banded(_i16_2d), onp.Array2D[np.float32])
assert_type(cholesky_banded(_i32_2d), onp.Array2D[np.float64])
assert_type(cholesky_banded(_i64_2d), onp.Array2D[np.float64])
assert_type(cholesky_banded(_f16_2d), onp.Array2D[np.float32])
assert_type(cholesky_banded(_f32_2d), onp.Array2D[np.float32])
assert_type(cholesky_banded(_f64_2d), onp.Array2D[np.float64])
assert_type(cholesky_banded(_f80_2d), onp.Array2D[np.float64])
assert_type(cholesky_banded(_c64_2d), onp.Array2D[np.complex64])
assert_type(cholesky_banded(_c128_2d), onp.Array2D[np.complex128])
assert_type(cholesky_banded(_c160_2d), onp.Array2D[np.complex128])
assert_type(cholesky_banded(_number_2d), onp.Array2D[Any])

cholesky_banded(_f64_1d)  # type:ignore[type-var] # pyright:ignore[reportArgumentType,reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_type(cholesky_banded(_f64_3d), onp.Array3D[np.float64])
assert_type(cholesky_banded(_f64_nd), onp.ArrayND[np.float64])

assert_type(cholesky_banded(_py_f_2d), onp.Array2D[np.float64])
assert_type(cholesky_banded(_py_f_3d), onp.ArrayND[np.float64])
assert_type(cholesky_banded(_py_c_2d), onp.Array2D[np.complex128])
assert_type(cholesky_banded(_py_c_3d), onp.ArrayND[np.complex128])

###
# cho_solve

assert_type(cho_solve((_bool_2d, False), _bool_2d), onp.ArrayND[np.float32])
assert_type(cho_solve((_i8_2d, False), _i8_2d), onp.ArrayND[np.float32])
assert_type(cho_solve((_i16_2d, False), _i16_2d), onp.ArrayND[np.float32])
assert_type(cho_solve((_i32_2d, False), _i32_2d), onp.ArrayND[np.float64])
assert_type(cho_solve((_i64_2d, False), _i64_2d), onp.ArrayND[np.float64])
assert_type(cho_solve((_f16_2d, False), _f16_2d), onp.ArrayND[np.float32])
assert_type(cho_solve((_f32_2d, False), _f32_2d), onp.ArrayND[np.float32])
assert_type(cho_solve((_f64_2d, False), _f64_2d), onp.ArrayND[np.float64])
assert_type(cho_solve((_f80_2d, False), _f80_2d), onp.ArrayND[np.float64])
assert_type(cho_solve((_c64_2d, False), _c64_2d), onp.ArrayND[np.complex64])
assert_type(cho_solve((_c128_2d, False), _c128_2d), onp.ArrayND[np.complex128])
assert_type(cho_solve((_c160_2d, False), _c160_2d), onp.ArrayND[np.complex128])
assert_type(cho_solve((_number_2d, False), _number_2d), onp.ArrayND[Any])

assert_type(cho_solve((_i16_2d, False), _f32_2d), onp.ArrayND[np.float32])
assert_type(cho_solve((_i16_2d, False), _f64_2d), onp.ArrayND[np.float64])
assert_type(cho_solve((_i16_2d, False), _c64_2d), onp.ArrayND[np.complex64])
assert_type(cho_solve((_i16_2d, False), _c128_2d), onp.ArrayND[np.complex128])

assert_type(cho_solve((_f32_2d, False), _i16_2d), onp.ArrayND[np.float32])
assert_type(cho_solve((_f32_2d, False), _f64_2d), onp.ArrayND[np.float64])
assert_type(cho_solve((_f32_2d, False), _c64_2d), onp.ArrayND[np.complex64])
assert_type(cho_solve((_f32_2d, False), _c128_2d), onp.ArrayND[np.complex128])

assert_type(cho_solve((_f64_2d, False), _i16_2d), onp.ArrayND[np.float64])
assert_type(cho_solve((_f64_2d, False), _f32_2d), onp.ArrayND[np.float64])
assert_type(cho_solve((_f64_2d, False), _c64_2d), onp.ArrayND[Any])
assert_type(cho_solve((_f64_2d, False), _c128_2d), onp.ArrayND[np.complex128])

assert_type(cho_solve((_c64_2d, False), _i16_2d), onp.ArrayND[np.complex64])
assert_type(cho_solve((_c64_2d, False), _f32_2d), onp.ArrayND[np.complex64])
assert_type(cho_solve((_c64_2d, False), _f64_2d), onp.ArrayND[Any])
assert_type(cho_solve((_c64_2d, False), _c128_2d), onp.ArrayND[np.complex128])

assert_type(cho_solve((_c128_2d, False), _i16_2d), onp.ArrayND[np.complex128])
assert_type(cho_solve((_c128_2d, False), _f32_2d), onp.ArrayND[np.complex128])
assert_type(cho_solve((_c128_2d, False), _f64_2d), onp.ArrayND[np.complex128])
assert_type(cho_solve((_c128_2d, False), _c64_2d), onp.ArrayND[np.complex128])

assert_type(cho_solve((_py_f_2d, False), _py_f_2d), onp.ArrayND[np.float64])
assert_type(cho_solve((_py_f_2d, False), _py_c_2d), onp.ArrayND[np.complex128])
assert_type(cho_solve((_py_c_2d, False), _py_f_2d), onp.ArrayND[np.complex128])
assert_type(cho_solve((_py_c_2d, False), _py_c_2d), onp.ArrayND[np.complex128])

###
# cho_solve_banded

assert_type(cho_solve_banded((_f64_2d, False), _c128_1d), _Complex2D)
assert_type(cho_solve_banded((_f64_3d, False), _c128_3d), _ComplexND)
assert_type(cho_solve_banded((_c128_2d, False), _c128_1d), _Complex2D)
assert_type(cho_solve_banded((_c128_3d, False), _c128_3d), _ComplexND)
