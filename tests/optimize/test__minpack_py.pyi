from typing import Literal, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import curve_fit, fixed_point, fsolve, leastsq
from scipy.optimize._minpack_py import _InfoDictCurveFit, _InfoDictLSQ, _InfoDictSolve

###

_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]
_IERFlag: TypeAlias = Literal[1, 2, 3, 4, 5, 6, 7, 8]

###

def _func(x: _Float1D, /) -> list[float]: ...
def _jac(x: _Float1D, /) -> list[list[float]]: ...
def _model(x: _Float1D, a: float, b: float) -> list[float]: ...

###
# fsolve

assert_type(fsolve(_func, [1.0, 2.0]), _Float1D)
assert_type(fsolve(_func, [1.0, 2.0], (), None, True), tuple[_Float1D, _InfoDictSolve, _IERFlag, str])
assert_type(fsolve(_func, [1.0, 2.0], full_output=True), tuple[_Float1D, _InfoDictSolve, _IERFlag, str])

###
# leastsq

assert_type(leastsq(_func, [1.0, 2.0]), tuple[_Float1D, _IERFlag])
assert_type(leastsq(_func, [1.0, 2.0], (), None, True), tuple[_Float1D, _Float2D, _InfoDictLSQ, str, _IERFlag])
assert_type(leastsq(_func, [1.0, 2.0], full_output=True), tuple[_Float1D, _Float2D, _InfoDictLSQ, str, _IERFlag])

###
# curve_fit, 1-d xdata

assert_type(curve_fit(_model, [1.0, 2.0], [3.0, 4.0]), tuple[_Float1D, _Float2D])
assert_type(
    curve_fit(_model, [1.0, 2.0], [3.0, 4.0], full_output=True), tuple[_Float1D, _Float2D, _InfoDictCurveFit, str, _IERFlag]
)

def _model_2d(x: _Float2D, a: float, b: float) -> list[float]: ...

assert_type(curve_fit(_model_2d, [[1.0, 2.0], [3.0, 4.0]], [3.0, 4.0]), tuple[_Float2D, _Float2D])
assert_type(
    curve_fit(_model_2d, [[1.0, 2.0], [3.0, 4.0]], [3.0, 4.0], full_output=True),
    tuple[_Float2D, _Float2D, _InfoDictCurveFit, str, _IERFlag],
)

###
# fixed_point

def _fp(x: float, /) -> float: ...
def _fp_c(x: complex, /) -> complex: ...
def _fp1d(x: _Float1D, /) -> list[float]: ...
def _fp1d_c(x: onp.Array1D[np.complex128], /) -> list[complex]: ...
def _fp2d(x: _Float2D, /) -> list[list[float]]: ...

assert_type(fixed_point(_fp, 0.5), np.float64)
assert_type(fixed_point(_fp_c, 0.5 + 0j), np.complex128)
assert_type(fixed_point(_fp1d, [0.5, 1.0]), _Float1D)
assert_type(fixed_point(_fp1d_c, [0.5 + 0j, 1.0 + 0j]), onp.Array1D[np.complex128])
assert_type(fixed_point(_fp2d, [[0.5, 1.0], [1.5, 2.0]]), _Float2D)
