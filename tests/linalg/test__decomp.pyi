# type-tests for `linalg/_decomp.pyi`

from typing import TypeAlias, assert_type

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

_FloatND: TypeAlias = onp.ArrayND[np.float32 | np.float64]
_ComplexND: TypeAlias = onp.ArrayND[np.complex64 | np.complex128]
_InexactND: TypeAlias = onp.ArrayND[np.float32 | np.float64 | np.complex64 | np.complex128]

###
# Input arrays

f32_nd: onp.ArrayND[np.float32]
c128_nd: onp.ArrayND[np.complex128]

py_f_1d: list[float]
py_f_2d: list[list[float]]

w_f64: onp.ArrayND[np.float64]
v_f64: onp.ArrayND[np.float64]
w_f32: onp.ArrayND[np.float32]
v_f32: onp.ArrayND[np.float32]

###
# eigvals

assert_type(eigvals(f32_nd), _ComplexND)
assert_type(eigvals(c128_nd), _ComplexND)
assert_type(eigvals(f32_nd, homogeneous_eigvals=True), _ComplexND)
assert_type(eigvals(c128_nd, homogeneous_eigvals=True), _ComplexND)

###
# eigvalsh

assert_type(eigvalsh(f32_nd), _FloatND)
assert_type(eigvalsh(c128_nd), _FloatND)

###
# eigvalsh_tridiagonal

assert_type(eigvalsh_tridiagonal(py_f_1d, py_f_1d), _FloatND)
assert_type(eigvalsh_tridiagonal(py_f_1d, py_f_1d, "v", [0.5, 1.5]), _FloatND)
assert_type(eigvalsh_tridiagonal(py_f_1d, py_f_1d, "i", [0, 2]), _FloatND)

###
# eigvals_banded

assert_type(eigvals_banded(py_f_2d), _FloatND)
assert_type(eigvals_banded(py_f_2d, select="v", select_range=[0.5, 1.5]), _FloatND)
assert_type(eigvals_banded(py_f_2d, select="i", select_range=[0, 2]), _FloatND)

###
# eigh_tridiagonal

assert_type(eigh_tridiagonal(py_f_1d, py_f_1d), tuple[_FloatND, _FloatND])
assert_type(eigh_tridiagonal(py_f_1d, py_f_1d, False, "v", [0.5, 1.5]), tuple[_FloatND, _FloatND])
assert_type(eigh_tridiagonal(py_f_1d, py_f_1d, False, "i", [0, 2]), tuple[_FloatND, _FloatND])
assert_type(eigh_tridiagonal(py_f_1d, py_f_1d, True), _FloatND)
assert_type(eigh_tridiagonal(py_f_1d, py_f_1d, True, "v", [0.5, 1.5]), _FloatND)
assert_type(eigh_tridiagonal(py_f_1d, py_f_1d, True, "i", [0, 2]), _FloatND)

###
# eigh

assert_type(eigh(f32_nd), tuple[_FloatND, _FloatND])
assert_type(eigh(c128_nd), tuple[_FloatND, _ComplexND])
assert_type(eigh(f32_nd, eigvals_only=True), _FloatND)
assert_type(eigh(c128_nd, eigvals_only=True), onp.ArrayND[np.float64])

###
# eig

assert_type(eig(c128_nd), _ComplexND)
assert_type(eig(c128_nd, None, True), tuple[_ComplexND, _InexactND])
assert_type(eig(c128_nd, None, True, False), tuple[_ComplexND, _InexactND, _InexactND])

assert_type(eig(f32_nd), _ComplexND)
assert_type(eig(f32_nd, None, True), tuple[_ComplexND, _FloatND])

###
# eig_banded

assert_type(eig_banded(py_f_2d), tuple[_FloatND, _FloatND])
assert_type(eig_banded(c128_nd), tuple[_FloatND, _InexactND])
assert_type(eig_banded(py_f_2d, eigvals_only=True), _FloatND)
assert_type(eig_banded(c128_nd, True, True), _FloatND)

###
# hessenberg

assert_type(hessenberg(f32_nd), _FloatND)
assert_type(hessenberg(f32_nd, True), tuple[_FloatND, _FloatND])

assert_type(hessenberg(c128_nd), _InexactND)
assert_type(hessenberg(c128_nd, True), tuple[_InexactND, _InexactND])

###
# cdf2rdf

assert_type(cdf2rdf(w_f64, v_f64), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(cdf2rdf(w_f32, v_f32), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(cdf2rdf(w_f64, v_f32), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float32]])
assert_type(cdf2rdf(w_f64, c128_nd), tuple[onp.ArrayND[np.float64], _FloatND])
assert_type(cdf2rdf(c128_nd, v_f64), tuple[_FloatND, onp.ArrayND[np.float64]])
assert_type(cdf2rdf(c128_nd, c128_nd), tuple[_FloatND, _FloatND])
