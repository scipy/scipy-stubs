from typing import Any, Literal, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import (
    approx_fprime,
    bracket,
    brent,
    brute,
    check_grad,
    fmin,
    fmin_bfgs,
    fmin_cg,
    fmin_ncg,
    fmin_powell,
    fminbound,
    golden,
    line_search,
    rosen,
    rosen_der,
    rosen_hess,
    rosen_hess_prod,
    show_options,
)

_Float: TypeAlias = float | np.float64
_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]
_AllVecs: TypeAlias = list[onp.Array1D[np.intp] | _Float1D]
_WarnFlag: TypeAlias = Literal[0, 1, 2, 3, 4]
_BracketInfo: TypeAlias = tuple[_Float, _Float, _Float, _Float, _Float, _Float, int]

_x0: list[float]
_arr_1d: _Float1D

def _f(x: _Float1D, /) -> float: ...
def _f0d(x: float, /) -> float: ...
def _fprime(x: _Float1D, /) -> _Float1D: ...

###
# rosen, rosen_der, rosen_hess, rosen_hess_prod

assert_type(rosen([1.0, 2.0, 3.0]), _Float)
assert_type(rosen_der([1.0, 2.0, 3.0]), _Float1D)
assert_type(rosen_hess([1.0, 2.0, 3.0]), _Float2D)
assert_type(rosen_hess_prod([1.0, 2.0, 3.0], [1.0, 0.0, 0.0]), _Float1D)

###
# approx_fprime

assert_type(approx_fprime([1.0, 2.0], _f, 1e-8), _Float1D)

###
# check_grad

assert_type(check_grad(_f, _fprime, [1.0, 2.0]), _Float)

###
# bracket

assert_type(bracket(_f0d), _BracketInfo)
assert_type(bracket(_f0d, xa=0.0, xb=1.0), _BracketInfo)
assert_type(bracket(_f0d, args=()), _BracketInfo)

###
# brent

assert_type(brent(_f0d), _Float)
assert_type(brent(_f0d, (), None, 1.48e-08, True), tuple[_Float, _Float, int, int])
assert_type(brent(_f0d, full_output=True), tuple[_Float, _Float, int, int])

###
# golden

assert_type(golden(_f0d), _Float)
assert_type(golden(_f0d, (), None, 1e-5, True), tuple[_Float, _Float, int])
assert_type(golden(_f0d, full_output=True), tuple[_Float, _Float, int])

###
# fminbound

assert_type(fminbound(_f0d, 0.0, 1.0), _Float)
assert_type(fminbound(_f0d, 0.0, 1.0, full_output=True), tuple[_Float, _Float, _WarnFlag, int])

###
# brute

assert_type(brute(_f, ((-1.0, 1.0), (-1.0, 1.0))), _Float1D)
assert_type(
    brute(_f, ((-1.0, 1.0), (-1.0, 1.0)), full_output=True),
    tuple[_Float1D, np.float64, onp.Array3D[np.float64], onp.Array2D[np.floating[Any]]],
)

###
# fmin

assert_type(fmin(_f, _x0), _Float1D)
assert_type(fmin(_f, _x0, retall=True), tuple[_Float1D, _AllVecs])
assert_type(fmin(_f, _x0, full_output=True), tuple[_Float1D, onp.ToFloat, int, int, _WarnFlag])
assert_type(fmin(_f, _x0, full_output=True, retall=True), tuple[_Float1D, onp.ToFloat, int, int, _WarnFlag, _AllVecs])

###
# fmin_bfgs

assert_type(fmin_bfgs(_f, _x0), _Float1D)
assert_type(fmin_bfgs(_f, _x0, retall=True), tuple[_Float1D, _AllVecs])
assert_type(
    fmin_bfgs(_f, _x0, full_output=True),
    tuple[_Float1D, _Float, _Float1D, _Float2D, int, int, _WarnFlag],
)
assert_type(
    fmin_bfgs(_f, _x0, full_output=True, retall=True),
    tuple[_Float1D, _Float, _Float1D, _Float2D, int, int, _WarnFlag, _AllVecs],
)

###
# fmin_cg

assert_type(fmin_cg(_f, _x0), _Float1D)
assert_type(fmin_cg(_f, _x0, retall=True), tuple[_Float1D, _AllVecs])
assert_type(fmin_cg(_f, _x0, full_output=True), tuple[_Float1D, _Float, int, int, _WarnFlag])
assert_type(
    fmin_cg(_f, _x0, full_output=True, retall=True),
    tuple[_Float1D, _Float, int, int, _WarnFlag, _AllVecs],
)

###
# fmin_ncg

assert_type(fmin_ncg(_f, _x0, _fprime), _Float1D)
assert_type(fmin_ncg(_f, _x0, _fprime, retall=True), tuple[_Float1D, _AllVecs])
assert_type(fmin_ncg(_f, _x0, _fprime, full_output=True), tuple[_Float1D, _Float, int, int, int, _WarnFlag])
assert_type(
    fmin_ncg(_f, _x0, _fprime, full_output=True, retall=True),
    tuple[_Float1D, _Float, int, int, int, _WarnFlag, _AllVecs],
)

###
# fmin_powell

assert_type(fmin_powell(_f, _x0), _Float1D)
assert_type(fmin_powell(_f, _x0, retall=True), tuple[_Float1D, _AllVecs])
assert_type(
    fmin_powell(_f, _x0, full_output=True),
    tuple[_Float1D, _Float, _Float2D, int, int, _WarnFlag],
)
assert_type(
    fmin_powell(_f, _x0, full_output=True, retall=True),
    tuple[_Float1D, _Float, _Float2D, int, int, _WarnFlag, _AllVecs],
)

###
# show_options

assert_type(show_options(), None)
assert_type(show_options(None, None, False), str)
assert_type(show_options(disp=False), str)

###
# line_search

assert_type(
    line_search(_f, _fprime, _arr_1d, _arr_1d, _arr_1d),
    tuple[_Float | None, int, int, _Float | None, _Float, _Float | None],
)
