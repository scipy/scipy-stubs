from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import RootResults, bisect, brenth, brentq, newton, ridder, root_scalar, toms748

_Float: TypeAlias = float | np.float64
_RR: TypeAlias = RootResults[float | np.float64]

def f(x: float) -> float: ...
def g(x: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]: ...
def g2(x: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]: ...
def h(x: float) -> tuple[float, float]: ...
def k(x: float) -> tuple[float, float, float]: ...

arr_1d: onp.Array1D[np.float64]
arr_2d: onp.Array2D[np.float64]

# bisect
assert_type(bisect(f, 0.0, 1.0), float)
assert_type(bisect(f, 0.0, 1.0, full_output=True), tuple[float, _RR])

# ridder
assert_type(ridder(f, 0.0, 1.0), float)
assert_type(ridder(f, 0.0, 1.0, full_output=True), tuple[float, _RR])

# brentq
assert_type(brentq(f, 0.0, 1.0), float)
assert_type(brentq(f, 0.0, 1.0, full_output=True), tuple[float, _RR])

# brenth
assert_type(brenth(f, 0.0, 1.0), float)
assert_type(brenth(f, 0.0, 1.0, full_output=True), tuple[float, _RR])

# toms748
assert_type(toms748(f, 0.0, 1.0), np.float64)
assert_type(toms748(f, 0.0, 1.0, full_output=True), tuple[np.float64, _RR])

# newton
assert_type(newton(f, 0.5), _Float)
assert_type(newton(f, 0.5, full_output=True), tuple[_Float, RootResults[_Float]])
assert_type(newton(g, arr_1d), onp.Array1D[np.float64])
assert_type(newton(g, arr_1d, full_output=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.bool_], onp.Array1D[np.bool_]])
assert_type(newton(g2, arr_2d), onp.Array2D[np.float64])
assert_type(newton(g2, arr_2d, full_output=True), tuple[onp.Array2D[np.float64], onp.Array2D[np.bool_], onp.Array2D[np.bool_]])

# root_scalar
assert_type(root_scalar(f, bracket=(0.0, 1.0)), RootResults[float])
assert_type(root_scalar(f, method="secant", x0=0.5), RootResults[float])
assert_type(root_scalar(f, method="newton", fprime=f, x0=0.5), RootResults[float])
assert_type(root_scalar(h, method="newton", fprime=True, x0=0.5), RootResults[float])
assert_type(root_scalar(f, method="halley", fprime=f, fprime2=f, x0=0.5), RootResults[float])
assert_type(root_scalar(k, method="halley", fprime=True, fprime2=True, x0=0.5), RootResults[float])
