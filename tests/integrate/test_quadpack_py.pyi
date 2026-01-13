from collections.abc import Callable
from typing import Literal, assert_type

import numpy as np

from scipy.integrate import dblquad, nquad, quad, tplquad
from scipy.integrate._quadpack_py import _QuadExplain, _QuadOutputNC
from scipy.integrate._typing import QuadInfoDict

TRUE: Literal[True] = True

###
# quad

# ufunc
assert_type(quad(np.exp, 0, 1), tuple[float, float])

# (float) -> float
f0_float_float: Callable[[float], float]
assert_type(quad(f0_float_float, 0, 1), tuple[float, float])

# (float) -> np.float64
f0_float_f8: Callable[[float], np.float64]
assert_type(quad(f0_float_f8, 0, 1), tuple[float, float])

# (np.float64) -> float
f0_f8_float: Callable[[np.float64], float]
assert_type(quad(f0_f8_float, 0, 1), tuple[float, float])

# (float, str) -> float
f1_float_float: Callable[[float, str], float]
assert_type(quad(f1_float_float, 0, 1, args=("",)), tuple[float, float])

# (float, str, str) -> float
f2_float_float: Callable[[float, str, str], float]
assert_type(quad(f2_float_float, 0, 1, args=("", "")), tuple[float, float])

# (float) -> float, full output
# NOTE: this test fails (only) in mypy due to some mypy bug
assert_type(
    quad(f0_float_float, 0, 1, full_output=TRUE),
    tuple[float, float, QuadInfoDict]
    | tuple[float, float, QuadInfoDict, str]
    | tuple[float, float, QuadInfoDict, str, _QuadExplain],
)

# (float) -> complex
# NOTE: this test fails (only) in mypy due to some mypy bug
z0_float_complex: Callable[[float], complex]
assert_type(quad(z0_float_complex, 0, 1, complex_func=TRUE), tuple[complex, complex])

###
# dblquad

def _f2_0(x: float, y: float) -> float: ...
def _f2_1(x: float, y: float, arg1: int) -> float: ...

assert_type(dblquad(_f2_0, 0, 1, 0, 1), tuple[float, float])
assert_type(dblquad(_f2_1, 0, 1, 0, 1, args=(1,)), tuple[float, float])

###
# tplquad

def _f3_0(x: float, y: float, z: float) -> float: ...
def _f3_1(x: float, y: float, z: float, arg1: int) -> float: ...

assert_type(tplquad(_f3_0, 0, 1, 0, 1, 0, 1), tuple[float, float])
assert_type(tplquad(_f3_1, 0, 1, 0, 1, 0, 1, args=(1,)), tuple[float, float])

###
# nquad

assert_type(nquad(_f2_0, [(0, 1), (0, 1)]), tuple[float, float])
assert_type(nquad(_f2_1, [(0, 1), (0, 1)], args=(2,)), tuple[float, float])
assert_type(nquad(_f3_0, [(0, 1), (0, 1), (0, 1)]), tuple[float, float])
assert_type(nquad(_f3_1, [(0, 1), (0, 1), (0, 1)], args=(2,)), tuple[float, float])

assert_type(nquad(_f2_0, [(0, 1), (0, 1)], full_output=True), tuple[float, float, _QuadOutputNC])
assert_type(nquad(_f2_1, [(0, 1), (0, 1)], args=(2,), full_output=True), tuple[float, float, _QuadOutputNC])
assert_type(nquad(_f3_0, [(0, 1), (0, 1), (0, 1)], full_output=True), tuple[float, float, _QuadOutputNC])
assert_type(nquad(_f3_1, [(0, 1), (0, 1), (0, 1)], args=(2,), full_output=True), tuple[float, float, _QuadOutputNC])
