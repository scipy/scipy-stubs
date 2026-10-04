from typing import assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.linalg import diagsvd, null_space, orth, subspace_angles, svd, svdvals

###

_py_i_2d: list[list[int]]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]

_bool_nd: onp.ArrayND[np.bool]
_i8_nd: onp.ArrayND[np.int8]
_i16_nd: onp.ArrayND[np.int16]
_i32_nd: onp.ArrayND[np.int32]
_i64_nd: onp.ArrayND[np.int64]
_f16_nd: onp.ArrayND[np.float16]
_f32_nd: onp.ArrayND[np.float32]
_f64_nd: onp.ArrayND[np.float64]
_f80_nd: onp.ArrayND[np.float128]
_c64_nd: onp.ArrayND[np.complex64]
_c128_nd: onp.ArrayND[np.complex128]
_c160_nd: onp.ArrayND[np.complex256]

_bool_2d: onp.Array2D[np.bool]
_i8_2d: onp.Array2D[np.int8]
_i16_2d: onp.Array2D[np.int16]
_i32_2d: onp.Array2D[np.int32]
_i64_2d: onp.Array2D[np.int64]
_f16_2d: onp.Array2D[np.float16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]

_f32_3d: onp.Array3D[np.float32]
_f64_3d: onp.Array3D[np.float64]

###

# svd

assert_type(svd(_py_i_2d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(svd(_py_f_2d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(svd(_py_c_2d), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.float64], onp.ArrayND[np.complex128]])
assert_type(svd(_f16_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(svd(_f32_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(svd(_f64_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(svd(_f80_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(svd(_c64_nd), tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.float32], onp.ArrayND[np.complex64]])
assert_type(svd(_c128_nd), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.float64], onp.ArrayND[np.complex128]])
assert_type(svd(_c160_nd), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.float64], onp.ArrayND[np.complex128]])

assert_type(svd(_py_i_2d, compute_uv=False), onp.ArrayND[np.float64])
assert_type(svd(_py_f_2d, compute_uv=False), onp.ArrayND[np.float64])
assert_type(svd(_py_c_2d, compute_uv=False), onp.ArrayND[np.float64])
assert_type(svd(_f16_nd, compute_uv=False), onp.ArrayND[np.float32])
assert_type(svd(_f32_nd, compute_uv=False), onp.ArrayND[np.float32])
assert_type(svd(_f64_nd, compute_uv=False), onp.ArrayND[np.float64])
assert_type(svd(_f80_nd, compute_uv=False), onp.ArrayND[np.float64])
assert_type(svd(_c64_nd, compute_uv=False), onp.ArrayND[np.float32])
assert_type(svd(_c128_nd, compute_uv=False), onp.ArrayND[np.float64])
assert_type(svd(_c160_nd, compute_uv=False), onp.ArrayND[np.float64])

# svdvals

assert_type(svdvals(_bool_nd), onp.ArrayND[np.float32])
assert_type(svdvals(_i8_nd), onp.ArrayND[np.float32])
assert_type(svdvals(_i16_nd), onp.ArrayND[np.float32])
assert_type(svdvals(_i32_nd), onp.ArrayND[np.float64])
assert_type(svdvals(_i64_nd), onp.ArrayND[np.float64])
assert_type(svdvals(_f16_nd), onp.ArrayND[np.float32])
assert_type(svdvals(_f32_nd), onp.ArrayND[np.float32])
assert_type(svdvals(_f64_nd), onp.ArrayND[np.float64])
assert_type(svdvals(_c64_nd), onp.ArrayND[np.float32])
assert_type(svdvals(_c128_nd), onp.ArrayND[np.float64])
assert_type(svdvals(_py_f_2d), onp.ArrayND[np.float64])
assert_type(svdvals(_py_c_2d), onp.ArrayND[np.float64])

# diagsvd

assert_type(diagsvd(_f64_nd, 3, 4), onp.ArrayND[np.float64])
assert_type(diagsvd(_f32_nd, 3, 4), onp.ArrayND[np.float32])
assert_type(diagsvd([True, False], 2, 3), onp.ArrayND[np.bool])  # type: ignore[assert-type]  # mypy bug
assert_type(diagsvd([1, 2], 2, 3), onp.ArrayND[np.intp])  # type: ignore[assert-type]  # mypy bug
assert_type(diagsvd([1.0, 2.0], 2, 3), onp.ArrayND[np.float64])  # type: ignore[assert-type]  # mypy bug

# orth

assert_type(orth(_f64_nd), onp.ArrayND[np.float64])
assert_type(orth(_py_f_2d), onp.ArrayND[np.float64])
assert_type(orth(_py_c_2d), onp.ArrayND[np.complex128])
assert_type(orth(_f32_nd), onp.ArrayND[np.float32])
assert_type(orth(_c64_nd), onp.ArrayND[np.complex64])

# null_space

assert_type(null_space(_f64_nd), onp.ArrayND[np.float64])
assert_type(null_space(_py_f_2d), onp.ArrayND[np.float64])
assert_type(null_space(_py_c_2d), onp.ArrayND[np.complex128])
assert_type(null_space(_f32_nd), onp.ArrayND[np.float32])
assert_type(null_space(_c64_nd), onp.ArrayND[np.complex64])

# subspace_angles

assert_type(subspace_angles(_bool_2d, _bool_2d), onp.Array1D[np.float32])
assert_type(subspace_angles(_i8_2d, _i8_2d), onp.Array1D[np.float32])
assert_type(subspace_angles(_i16_2d, _i16_2d), onp.Array1D[np.float32])
assert_type(subspace_angles(_i32_2d, _i32_2d), onp.Array1D[np.float64])
assert_type(subspace_angles(_i64_2d, _i64_2d), onp.Array1D[np.float64])
assert_type(subspace_angles(_f16_2d, _f16_2d), onp.Array1D[np.float32])
assert_type(subspace_angles(_f32_2d, _f32_2d), onp.Array1D[np.float32])
assert_type(subspace_angles(_f64_2d, _f64_2d), onp.Array1D[np.float64])
assert_type(subspace_angles(_c64_2d, _c64_2d), onp.Array1D[np.float32])
assert_type(subspace_angles(_c128_2d, _c128_2d), onp.Array1D[np.float64])
assert_type(subspace_angles(_py_f_2d, _py_f_2d), onp.Array1D[np.float64])
assert_type(subspace_angles(_py_c_2d, _py_c_2d), onp.Array1D[np.float64])
assert_type(subspace_angles(_f32_2d, _f64_2d), onp.Array1D[np.float64])
assert_type(subspace_angles(_f64_3d, _f64_3d), onp.ArrayND[np.float64])
assert_type(subspace_angles(_f32_3d, _f64_3d), onp.ArrayND[np.float64])
assert_type(subspace_angles(_f32_3d, _f32_3d), onp.ArrayND[np.float32])
assert_subtype[onp.ArrayND[np.float64]](subspace_angles(_f64_nd, _f64_2d))
assert_subtype[onp.ArrayND[np.float64]](subspace_angles(_f64_2d, _f64_nd))
assert_subtype[onp.ArrayND[np.float32]](subspace_angles(_f32_nd, _f32_2d))
assert_subtype[onp.ArrayND[np.float32]](subspace_angles(_f32_2d, _f32_nd))
