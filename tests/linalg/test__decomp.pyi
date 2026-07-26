# type-tests for `linalg/_decomp.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import (
    cdf2rdf,
    eig,
    eig_banded,
    eigh,
    eigh_tridiagonal,
    eigvals,
    eigvals_banded,
    eigvalsh,
    eigvalsh_tridiagonal,
    hessenberg,
)

###

type _FloatND = onp.ArrayND[np.float64 | np.float32]
type _ComplexND = onp.ArrayND[np.complex128 | np.complex64]
type _InexactND = onp.ArrayND[np.complex128 | np.complex64 | np.float64 | np.float32]

###
# Input arrays

_i8_nd: onp.ArrayND[np.int8]
_i32_nd: onp.ArrayND[np.int32]
_f16_nd: onp.ArrayND[np.float16]
_f32_nd: onp.ArrayND[np.float32]
_f64_nd: onp.ArrayND[np.float64]
_f128_nd: onp.ArrayND[np.float128]
_c64_nd: onp.ArrayND[np.complex64]
_c128_nd: onp.ArrayND[np.complex128]
_c256_nd: onp.ArrayND[np.complex256]

_py_i_1d: list[int]
_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]

###
# eigvals

assert_type(eigvals(_i8_nd), onp.ArrayND[np.complex64])
assert_type(eigvals(_f32_nd), onp.ArrayND[np.complex64])
assert_type(eigvals(_c64_nd), onp.ArrayND[np.complex64])
assert_type(eigvals(_i32_nd), onp.ArrayND[np.complex128])
assert_type(eigvals(_f64_nd), onp.ArrayND[np.complex128])
assert_type(eigvals(_c128_nd), onp.ArrayND[np.complex128])
assert_type(eigvals(_py_f_2d), onp.ArrayND[np.complex128])
assert_type(eigvals(_py_c_2d), onp.ArrayND[np.complex128])
assert_type(eigvals(_f32_nd, homogeneous_eigvals=True), onp.ArrayND[np.complex64])
assert_type(eigvals(_c128_nd, homogeneous_eigvals=True), onp.ArrayND[np.complex128])

# `b` widens the result to `c128` unless it fits in `c64` as well
assert_type(eigvals(_f32_nd, _c64_nd), onp.ArrayND[np.complex64])
assert_type(eigvals(_f32_nd, _f64_nd), onp.ArrayND[np.complex128])
assert_type(eigvals(_f64_nd, _f32_nd), onp.ArrayND[np.complex128])
assert_type(eigvals(_f32_nd, None), onp.ArrayND[np.complex64])

# deprecated input dtypes
assert_type(eigvals(_f16_nd), onp.ArrayND[np.complex64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigvals(_f16_nd, _f64_nd), onp.ArrayND[np.complex128])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigvals(_f128_nd), onp.ArrayND[np.complex128])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigvals(_f32_nd, _f128_nd), onp.ArrayND[np.complex128])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

###
# eigvalsh

assert_type(eigvalsh(_i8_nd), onp.ArrayND[np.float32])
assert_type(eigvalsh(_f32_nd), onp.ArrayND[np.float32])
assert_type(eigvalsh(_c64_nd), onp.ArrayND[np.float32])
assert_type(eigvalsh(_i32_nd), onp.ArrayND[np.float64])
assert_type(eigvalsh(_f64_nd), onp.ArrayND[np.float64])
assert_type(eigvalsh(_c128_nd), onp.ArrayND[np.float64])
assert_type(eigvalsh(_py_f_2d), onp.ArrayND[np.float64])
assert_type(eigvalsh(_py_c_2d), onp.ArrayND[np.float64])
assert_type(eigvalsh(_f32_nd, subset_by_index=[0, 1], driver="evx"), onp.ArrayND[np.float32])

# `b` widens the result to `f64` unless it fits in `c64` as well
assert_type(eigvalsh(_f32_nd, _c64_nd), onp.ArrayND[np.float32])
assert_type(eigvalsh(_f32_nd, _f64_nd), onp.ArrayND[np.float64])
assert_type(eigvalsh(_f64_nd, _f32_nd), onp.ArrayND[np.float64])
assert_type(eigvalsh(_f32_nd, None), onp.ArrayND[np.float32])

# deprecated input dtypes
assert_type(eigvalsh(_f16_nd), onp.ArrayND[np.float32])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigvalsh(_f16_nd, _f64_nd), onp.ArrayND[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigvalsh(_f128_nd), onp.ArrayND[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigvalsh(_f32_nd, _f128_nd), onp.ArrayND[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

###
# eigvalsh_tridiagonal

assert_type(eigvalsh_tridiagonal(_i8_nd, _i8_nd), onp.ArrayND[np.float32])
assert_type(eigvalsh_tridiagonal(_f32_nd, _f32_nd), onp.ArrayND[np.float32])
assert_type(eigvalsh_tridiagonal(_i32_nd, _i32_nd), onp.ArrayND[np.float64])
assert_type(eigvalsh_tridiagonal(_f64_nd, _f64_nd), onp.ArrayND[np.float64])
assert_type(eigvalsh_tridiagonal(_py_f_1d, _py_f_1d), onp.ArrayND[np.float64])

assert_type(eigvalsh_tridiagonal(_f32_nd, _i8_nd), onp.ArrayND[np.float32])
assert_type(eigvalsh_tridiagonal(_f32_nd, _f64_nd), onp.ArrayND[np.float64])
assert_type(eigvalsh_tridiagonal(_f64_nd, _f32_nd), onp.ArrayND[np.float64])

assert_type(eigvalsh_tridiagonal(_f32_nd, _f32_nd, "v", _py_f_1d), onp.ArrayND[np.float32])
assert_type(eigvalsh_tridiagonal(_py_f_1d, _py_f_1d, "v", _py_f_1d), onp.ArrayND[np.float64])
assert_type(eigvalsh_tridiagonal(_f32_nd, _f32_nd, "i", _py_i_1d), onp.ArrayND[np.float32])
assert_type(eigvalsh_tridiagonal(_py_f_1d, _py_f_1d, "i", _py_i_1d), onp.ArrayND[np.float64])

assert_type(eigvalsh_tridiagonal(_f16_nd, _f16_nd), onp.ArrayND[np.float32])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigvalsh_tridiagonal(_f16_nd, _f64_nd), onp.ArrayND[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigvalsh_tridiagonal(_f64_nd, _f16_nd), onp.ArrayND[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigvalsh_tridiagonal(_f128_nd, _f32_nd), onp.ArrayND[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigvalsh_tridiagonal(_f32_nd, _f128_nd), onp.ArrayND[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

eigvalsh_tridiagonal(_c128_nd, _c128_nd)  # type: ignore[arg-type]  # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]

###
# eigvals_banded

assert_type(eigvals_banded(_i8_nd), onp.ArrayND[np.float32])
assert_type(eigvals_banded(_f32_nd), onp.ArrayND[np.float32])
assert_type(eigvals_banded(_c64_nd), onp.ArrayND[np.float32])
assert_type(eigvals_banded(_i32_nd), onp.ArrayND[np.float64])
assert_type(eigvals_banded(_f64_nd), onp.ArrayND[np.float64])
assert_type(eigvals_banded(_c128_nd), onp.ArrayND[np.float64])
assert_type(eigvals_banded(_py_f_2d), onp.ArrayND[np.float64])
assert_type(eigvals_banded(_py_c_2d), onp.ArrayND[np.float64])

assert_type(eigvals_banded(_f32_nd, select="v", select_range=[0.5, 1.5]), onp.ArrayND[np.float32])
assert_type(eigvals_banded(_py_f_2d, select="v", select_range=[0.5, 1.5]), onp.ArrayND[np.float64])
assert_type(eigvals_banded(_f32_nd, select="i", select_range=[0, 2]), onp.ArrayND[np.float32])
assert_type(eigvals_banded(_py_f_2d, select="i", select_range=[0, 2]), onp.ArrayND[np.float64])
assert_type(eigvals_banded(_c64_nd, select="i", select_range=[0, 2]), onp.ArrayND[np.float32])
assert_type(eigvals_banded(_c128_nd, select="v", select_range=[0.5, 1.5]), onp.ArrayND[np.float64])

assert_type(eigvals_banded(_f16_nd), onp.ArrayND[np.float32])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigvals_banded(_f128_nd), onp.ArrayND[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigvals_banded(_c256_nd), onp.ArrayND[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

###
# eigh_tridiagonal

assert_type(eigh_tridiagonal(_i8_nd, _i8_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(eigh_tridiagonal(_f32_nd, _f32_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(eigh_tridiagonal(_i32_nd, _i32_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(eigh_tridiagonal(_f64_nd, _f64_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(eigh_tridiagonal(_py_f_1d, _py_f_1d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])

assert_type(eigh_tridiagonal(_f32_nd, _i8_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(eigh_tridiagonal(_f32_nd, _f64_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(eigh_tridiagonal(_f64_nd, _f32_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])

assert_type(eigh_tridiagonal(_f32_nd, _f32_nd, True), onp.ArrayND[np.float32])
assert_type(eigh_tridiagonal(_f64_nd, _f32_nd, True), onp.ArrayND[np.float64])
assert_type(eigh_tridiagonal(_py_f_1d, _py_f_1d, True), onp.ArrayND[np.float64])

assert_type(eigh_tridiagonal(_f32_nd, _f32_nd, False, "v", _py_f_1d), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(eigh_tridiagonal(_py_f_1d, _py_f_1d, False, "i", _py_i_1d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(eigh_tridiagonal(_f32_nd, _f32_nd, True, "v", _py_f_1d), onp.ArrayND[np.float32])
assert_type(eigh_tridiagonal(_py_f_1d, _py_f_1d, True, "i", _py_i_1d), onp.ArrayND[np.float64])

assert_type(eigh_tridiagonal(_f16_nd, _f16_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigh_tridiagonal(_f128_nd, _f32_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigh_tridiagonal(_f32_nd, _f128_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eigh_tridiagonal(_f16_nd, _f16_nd, True), onp.ArrayND[np.float32])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

eigh_tridiagonal(_c128_nd, _c128_nd)  # type: ignore[arg-type]  # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]

###
# eigh

assert_type(eigh(_f32_nd), tuple[_FloatND, _FloatND])
assert_type(eigh(_c128_nd), tuple[_FloatND, _ComplexND])
assert_type(eigh(_f32_nd, eigvals_only=True), _FloatND)
assert_type(eigh(_c128_nd, eigvals_only=True), onp.ArrayND[np.float64])

###
# eig

assert_type(eig(_f32_nd), tuple[_ComplexND, _InexactND])
assert_type(eig(_f64_nd), tuple[_ComplexND, _InexactND])
assert_type(eig(_c64_nd), tuple[_ComplexND, _ComplexND])
assert_type(eig(_c128_nd), tuple[_ComplexND, _ComplexND])

assert_type(eig(_f32_nd, left=False, right=False), _ComplexND)
assert_type(eig(_f64_nd, left=False, right=False), _ComplexND)
assert_type(eig(_c64_nd, left=False, right=False), _ComplexND)
assert_type(eig(_c128_nd, left=False, right=False), _ComplexND)

assert_type(eig(_f32_nd, left=False, right=True), tuple[_ComplexND, _InexactND])
assert_type(eig(_f64_nd, left=False, right=True), tuple[_ComplexND, _InexactND])
assert_type(eig(_c64_nd, left=False, right=True), tuple[_ComplexND, _ComplexND])
assert_type(eig(_c128_nd, left=False, right=True), tuple[_ComplexND, _ComplexND])

assert_type(eig(_f32_nd, left=True, right=False), tuple[_ComplexND, _InexactND])
assert_type(eig(_f64_nd, left=True, right=False), tuple[_ComplexND, _InexactND])
assert_type(eig(_c64_nd, left=True, right=False), tuple[_ComplexND, _ComplexND])
assert_type(eig(_c128_nd, left=True, right=False), tuple[_ComplexND, _ComplexND])

assert_type(eig(_f32_nd, left=True, right=True), tuple[_ComplexND, _InexactND, _InexactND])
assert_type(eig(_f64_nd, left=True, right=True), tuple[_ComplexND, _InexactND, _InexactND])
assert_type(eig(_c64_nd, left=True, right=True), tuple[_ComplexND, _ComplexND, _ComplexND])
assert_type(eig(_c128_nd, left=True, right=True), tuple[_ComplexND, _ComplexND, _ComplexND])

###
# eig_banded

assert_type(eig_banded(_i8_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(eig_banded(_f32_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(eig_banded(_i32_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(eig_banded(_f64_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(eig_banded(_py_f_2d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(eig_banded(_c64_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.complex64]])
assert_type(eig_banded(_c128_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.complex128]])
assert_type(eig_banded(_py_c_2d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.complex128]])

# the eigenvalues are always real, so complex input does not widen the result here
assert_type(eig_banded(_f32_nd, eigvals_only=True), onp.ArrayND[np.float32])
assert_type(eig_banded(_c64_nd, eigvals_only=True), onp.ArrayND[np.float32])
assert_type(eig_banded(_f64_nd, eigvals_only=True), onp.ArrayND[np.float64])
assert_type(eig_banded(_py_f_2d, eigvals_only=True), onp.ArrayND[np.float64])
assert_type(eig_banded(_c128_nd, True, True), onp.ArrayND[np.float64])

assert_type(eig_banded(_f32_nd, select="v", select_range=_py_f_1d), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(eig_banded(_c128_nd, select="i", select_range=_py_i_1d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.complex128]])
assert_type(eig_banded(_c64_nd, eigvals_only=True, select="v", select_range=_py_f_1d), onp.ArrayND[np.float32])
assert_type(eig_banded(_f64_nd, True, True, select="i", select_range=_py_i_1d), onp.ArrayND[np.float64])

assert_type(eig_banded(_f16_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eig_banded(_f128_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eig_banded(_c256_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.complex128]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eig_banded(_f16_nd, eigvals_only=True), onp.ArrayND[np.float32])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(eig_banded(_c256_nd, eigvals_only=True), onp.ArrayND[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

###
# hessenberg

assert_type(hessenberg(_i8_nd), onp.ArrayND[np.float32])
assert_type(hessenberg(_f32_nd), onp.ArrayND[np.float32])
assert_type(hessenberg(_i32_nd), onp.ArrayND[np.float64])
assert_type(hessenberg(_f64_nd), onp.ArrayND[np.float64])
assert_type(hessenberg(_py_f_2d), onp.ArrayND[np.float64])
assert_type(hessenberg(_c64_nd), onp.ArrayND[np.complex64])
assert_type(hessenberg(_c128_nd), onp.ArrayND[np.complex128])
assert_type(hessenberg(_py_c_2d), onp.ArrayND[np.complex128])

assert_type(hessenberg(_f32_nd, True), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(hessenberg(_f64_nd, True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(hessenberg(_c64_nd, True), tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.complex64]])
assert_type(hessenberg(_c128_nd, calc_q=True), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])

assert_type(hessenberg(_f16_nd), onp.ArrayND[np.float32])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(hessenberg(_f128_nd), onp.ArrayND[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(hessenberg(_c256_nd), onp.ArrayND[np.complex128])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

###
# cdf2rdf

assert_type(cdf2rdf(_f64_nd, _f64_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(cdf2rdf(_f32_nd, _f32_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(cdf2rdf(_f64_nd, _f32_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float32]])
assert_type(cdf2rdf(_f64_nd, _c128_nd), tuple[onp.ArrayND[np.float64], _FloatND])
assert_type(cdf2rdf(_c128_nd, _f64_nd), tuple[_FloatND, onp.ArrayND[np.float64]])
assert_type(cdf2rdf(_c128_nd, _c128_nd), tuple[_FloatND, _FloatND])
