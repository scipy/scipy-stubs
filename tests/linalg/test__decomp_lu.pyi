# type-tests for `linalg/_decomp_lu.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import lu, lu_factor, lu_solve

###

_bool_2d: onp.Array2D[np.bool_]
_i8_2d: onp.Array2D[np.int8]
_i16_2d: onp.Array2D[np.int16]
_i32_2d: onp.Array2D[np.int32]
_i64_2d: onp.Array2D[np.int64]
_f16_2d: onp.Array2D[np.float16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_f80_2d: onp.Array2D[np.float128]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]
_c160_2d: onp.Array2D[np.complex256]

_bool_3d: onp.Array3D[np.bool_]
_i8_3d: onp.Array3D[np.int8]
_i16_3d: onp.Array3D[np.int16]
_i32_3d: onp.Array3D[np.int32]
_i64_3d: onp.Array3D[np.int64]
_f16_3d: onp.Array3D[np.float16]
_f32_3d: onp.Array3D[np.float32]
_f64_3d: onp.Array3D[np.float64]
_f80_3d: onp.Array3D[np.float128]
_c64_3d: onp.Array3D[np.complex64]
_c128_3d: onp.Array3D[np.complex128]
_c160_3d: onp.Array3D[np.complex256]

_bool_nd: onp.ArrayND[np.bool_]
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

_f64_1d: onp.Array1D[np.float64]
_c128_1d: onp.Array1D[np.complex128]

###
# lu_factor

assert_type(lu_factor(_bool_2d), tuple[onp.Array2D[np.float32], onp.Array1D[np.int32]])
assert_type(lu_factor(_i8_2d), tuple[onp.Array2D[np.float32], onp.Array1D[np.int32]])
assert_type(lu_factor(_i16_2d), tuple[onp.Array2D[np.float32], onp.Array1D[np.int32]])
assert_type(lu_factor(_i32_2d), tuple[onp.Array2D[np.float64], onp.Array1D[np.int32]])
assert_type(lu_factor(_i64_2d), tuple[onp.Array2D[np.float64], onp.Array1D[np.int32]])
assert_type(lu_factor(_f16_2d), tuple[onp.Array2D[np.float32], onp.Array1D[np.int32]])
assert_type(lu_factor(_f32_2d), tuple[onp.Array2D[np.float32], onp.Array1D[np.int32]])
assert_type(lu_factor(_f64_2d), tuple[onp.Array2D[np.float64], onp.Array1D[np.int32]])
assert_type(lu_factor(_f80_2d), tuple[onp.Array2D[np.float64], onp.Array1D[np.int32]])
assert_type(lu_factor(_c64_2d), tuple[onp.Array2D[np.complex64], onp.Array1D[np.int32]])
assert_type(lu_factor(_c128_2d), tuple[onp.Array2D[np.complex128], onp.Array1D[np.int32]])
assert_type(lu_factor(_c160_2d), tuple[onp.Array2D[np.complex128], onp.Array1D[np.int32]])

assert_type(lu_factor(_bool_3d), tuple[onp.Array3D[np.float32], onp.Array2D[np.int32]])
assert_type(lu_factor(_i8_3d), tuple[onp.Array3D[np.float32], onp.Array2D[np.int32]])
assert_type(lu_factor(_i16_3d), tuple[onp.Array3D[np.float32], onp.Array2D[np.int32]])
assert_type(lu_factor(_i32_3d), tuple[onp.Array3D[np.float64], onp.Array2D[np.int32]])
assert_type(lu_factor(_i64_3d), tuple[onp.Array3D[np.float64], onp.Array2D[np.int32]])
assert_type(lu_factor(_f16_3d), tuple[onp.Array3D[np.float32], onp.Array2D[np.int32]])
assert_type(lu_factor(_f32_3d), tuple[onp.Array3D[np.float32], onp.Array2D[np.int32]])
assert_type(lu_factor(_f64_3d), tuple[onp.Array3D[np.float64], onp.Array2D[np.int32]])
assert_type(lu_factor(_f80_3d), tuple[onp.Array3D[np.float64], onp.Array2D[np.int32]])
assert_type(lu_factor(_c64_3d), tuple[onp.Array3D[np.complex64], onp.Array2D[np.int32]])
assert_type(lu_factor(_c128_3d), tuple[onp.Array3D[np.complex128], onp.Array2D[np.int32]])
assert_type(lu_factor(_c160_3d), tuple[onp.Array3D[np.complex128], onp.Array2D[np.int32]])

assert_type(lu_factor(_bool_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.int32]])
assert_type(lu_factor(_i8_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.int32]])
assert_type(lu_factor(_i16_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.int32]])
assert_type(lu_factor(_i32_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]])
assert_type(lu_factor(_i64_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]])
assert_type(lu_factor(_f16_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.int32]])
assert_type(lu_factor(_f32_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.int32]])
assert_type(lu_factor(_f64_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]])
assert_type(lu_factor(_f80_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]])
assert_type(lu_factor(_c64_nd), tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.int32]])
assert_type(lu_factor(_c128_nd), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.int32]])
assert_type(lu_factor(_c160_nd), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.int32]])

###
# lu_solve

assert_type(lu_solve((_f64_2d, _f64_1d), _f64_1d), onp.Array2D[npc.floating])
assert_type(lu_solve((_f64_3d, _f64_3d), _f64_3d), onp.ArrayND[npc.floating])
assert_type(lu_solve((_c128_2d, _c128_1d), _c128_1d), onp.Array2D[npc.complexfloating])
assert_type(lu_solve((_c128_3d, _c128_3d), _c128_3d), onp.ArrayND[npc.complexfloating])

###
# lu

assert_type(lu(_bool_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_i8_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_i16_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_i32_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_i64_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_f16_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_f32_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_f64_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_f80_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_c64_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.complex64], onp.ArrayND[np.complex64]])
assert_type(lu(_c128_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])
assert_type(lu(_c160_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])

assert_type(lu(_bool_nd, p_indices=True), tuple[onp.ArrayND[np.int32], onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_i8_nd, p_indices=True), tuple[onp.ArrayND[np.int32], onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_i16_nd, p_indices=True), tuple[onp.ArrayND[np.int32], onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_i32_nd, p_indices=True), tuple[onp.ArrayND[np.int32], onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_i64_nd, p_indices=True), tuple[onp.ArrayND[np.int32], onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_f16_nd, p_indices=True), tuple[onp.ArrayND[np.int32], onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_f32_nd, p_indices=True), tuple[onp.ArrayND[np.int32], onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_f64_nd, p_indices=True), tuple[onp.ArrayND[np.int32], onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_f80_nd, p_indices=True), tuple[onp.ArrayND[np.int32], onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_c64_nd, p_indices=True), tuple[onp.ArrayND[np.int32], onp.ArrayND[np.complex64], onp.ArrayND[np.complex64]])
assert_type(lu(_c128_nd, p_indices=True), tuple[onp.ArrayND[np.int32], onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])
assert_type(lu(_c160_nd, p_indices=True), tuple[onp.ArrayND[np.int32], onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])

assert_type(lu(_bool_nd, permute_l=True), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_i8_nd, permute_l=True), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_i16_nd, permute_l=True), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_i32_nd, permute_l=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_i64_nd, permute_l=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_f16_nd, permute_l=True), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_f32_nd, permute_l=True), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_f64_nd, permute_l=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_f80_nd, permute_l=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_c64_nd, permute_l=True), tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.complex64]])
assert_type(lu(_c128_nd, permute_l=True), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])
assert_type(lu(_c160_nd, permute_l=True), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])

assert_type(lu(_bool_nd, permute_l=True, p_indices=True), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_i8_nd, permute_l=True, p_indices=True), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_i16_nd, permute_l=True, p_indices=True), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_i32_nd, permute_l=True, p_indices=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_i64_nd, permute_l=True, p_indices=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_f16_nd, permute_l=True, p_indices=True), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_f32_nd, permute_l=True, p_indices=True), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(lu(_f64_nd, permute_l=True, p_indices=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_f80_nd, permute_l=True, p_indices=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(lu(_c64_nd, permute_l=True, p_indices=True), tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.complex64]])
assert_type(lu(_c128_nd, permute_l=True, p_indices=True), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])
assert_type(lu(_c160_nd, permute_l=True, p_indices=True), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])
