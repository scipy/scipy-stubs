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
type _Complex64ND = onp.ArrayND[np.complex64]
type _Complex128ND = onp.ArrayND[np.complex128]
type _Vec32ND = onp.ArrayND[np.complex64 | np.float32]
type _Vec64ND = onp.ArrayND[np.complex128 | np.float64]

###
# Input arrays

_f32_nd: onp.ArrayND[np.float32]
_f64_nd: onp.ArrayND[np.float64]
_c64_nd: onp.ArrayND[np.complex64]
_c128_nd: onp.ArrayND[np.complex128]

_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]

###
# eigvals

assert_type(eigvals(_f32_nd), _ComplexND)
assert_type(eigvals(_c128_nd), _ComplexND)
assert_type(eigvals(_f32_nd, homogeneous_eigvals=True), _ComplexND)
assert_type(eigvals(_c128_nd, homogeneous_eigvals=True), _ComplexND)

###
# eigvalsh

assert_type(eigvalsh(_f32_nd), _FloatND)
assert_type(eigvalsh(_c128_nd), _FloatND)

###
# eigvalsh_tridiagonal

assert_type(eigvalsh_tridiagonal(_py_f_1d, _py_f_1d), _FloatND)
assert_type(eigvalsh_tridiagonal(_py_f_1d, _py_f_1d, "v", [0.5, 1.5]), _FloatND)
assert_type(eigvalsh_tridiagonal(_py_f_1d, _py_f_1d, "i", [0, 2]), _FloatND)

###
# eigvals_banded

assert_type(eigvals_banded(_py_f_2d), _FloatND)
assert_type(eigvals_banded(_py_f_2d, select="v", select_range=[0.5, 1.5]), _FloatND)
assert_type(eigvals_banded(_py_f_2d, select="i", select_range=[0, 2]), _FloatND)

###
# eigh_tridiagonal

assert_type(eigh_tridiagonal(_py_f_1d, _py_f_1d), tuple[_FloatND, _FloatND])
assert_type(eigh_tridiagonal(_py_f_1d, _py_f_1d, False, "v", [0.5, 1.5]), tuple[_FloatND, _FloatND])
assert_type(eigh_tridiagonal(_py_f_1d, _py_f_1d, False, "i", [0, 2]), tuple[_FloatND, _FloatND])
assert_type(eigh_tridiagonal(_py_f_1d, _py_f_1d, True), _FloatND)
assert_type(eigh_tridiagonal(_py_f_1d, _py_f_1d, True, "v", [0.5, 1.5]), _FloatND)
assert_type(eigh_tridiagonal(_py_f_1d, _py_f_1d, True, "i", [0, 2]), _FloatND)

###
# eigh

assert_type(eigh(_f32_nd), tuple[_FloatND, _FloatND])
assert_type(eigh(_c128_nd), tuple[_FloatND, _ComplexND])
assert_type(eigh(_f32_nd, eigvals_only=True), _FloatND)
assert_type(eigh(_c128_nd, eigvals_only=True), onp.ArrayND[np.float64])

###
# eig

assert_type(eig(_f32_nd), tuple[_Complex64ND, _Vec32ND])
assert_type(eig(_f64_nd), tuple[_Complex128ND, _Vec64ND])
assert_type(eig(_c64_nd), tuple[_Complex64ND, _Complex64ND])
assert_type(eig(_c128_nd), tuple[_Complex128ND, _Complex128ND])

assert_type(eig(_f32_nd, left=False, right=False), _Complex64ND)
assert_type(eig(_f64_nd, left=False, right=False), _Complex128ND)
assert_type(eig(_c64_nd, left=False, right=False), _Complex64ND)
assert_type(eig(_c128_nd, left=False, right=False), _Complex128ND)

assert_type(eig(_f32_nd, left=False, right=True), tuple[_Complex64ND, _Vec32ND])
assert_type(eig(_f64_nd, left=False, right=True), tuple[_Complex128ND, _Vec64ND])
assert_type(eig(_c64_nd, left=False, right=True), tuple[_Complex64ND, _Complex64ND])
assert_type(eig(_c128_nd, left=False, right=True), tuple[_Complex128ND, _Complex128ND])

assert_type(eig(_f32_nd, left=True, right=False), tuple[_Complex64ND, _Vec32ND])
assert_type(eig(_f64_nd, left=True, right=False), tuple[_Complex128ND, _Vec64ND])
assert_type(eig(_c64_nd, left=True, right=False), tuple[_Complex64ND, _Complex64ND])
assert_type(eig(_c128_nd, left=True, right=False), tuple[_Complex128ND, _Complex128ND])

assert_type(eig(_f32_nd, left=True, right=True), tuple[_Complex64ND, _Vec32ND, _Vec32ND])
assert_type(eig(_f64_nd, left=True, right=True), tuple[_Complex128ND, _Vec64ND, _Vec64ND])
assert_type(eig(_c64_nd, left=True, right=True), tuple[_Complex64ND, _Complex64ND, _Complex64ND])
assert_type(eig(_c128_nd, left=True, right=True), tuple[_Complex128ND, _Complex128ND, _Complex128ND])

###
# eig_banded

assert_type(eig_banded(_py_f_2d), tuple[_FloatND, _FloatND])
assert_type(eig_banded(_c128_nd), tuple[_FloatND, _InexactND])
assert_type(eig_banded(_py_f_2d, eigvals_only=True), _FloatND)
assert_type(eig_banded(_c128_nd, True, True), _FloatND)

###
# hessenberg

assert_type(hessenberg(_f32_nd), _FloatND)
assert_type(hessenberg(_f32_nd, True), tuple[_FloatND, _FloatND])

assert_type(hessenberg(_c128_nd), _InexactND)
assert_type(hessenberg(_c128_nd, True), tuple[_InexactND, _InexactND])

###
# cdf2rdf

assert_type(cdf2rdf(_f64_nd, _f64_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(cdf2rdf(_f32_nd, _f32_nd), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(cdf2rdf(_f64_nd, _f32_nd), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float32]])
assert_type(cdf2rdf(_f64_nd, _c128_nd), tuple[onp.ArrayND[np.float64], _FloatND])
assert_type(cdf2rdf(_c128_nd, _f64_nd), tuple[_FloatND, onp.ArrayND[np.float64]])
assert_type(cdf2rdf(_c128_nd, _c128_nd), tuple[_FloatND, _FloatND])
