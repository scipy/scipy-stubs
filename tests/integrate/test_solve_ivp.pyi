from typing import Any, TypeAlias, assert_type, type_check_only

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.integrate import solve_ivp

_VecF64: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
_MatF64: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
_ArrF64: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
_VecC128: TypeAlias = np.ndarray[tuple[Any], np.dtype[np.complex128]]
_MatC128: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.complex128]]
_ArrC128: TypeAlias = np.ndarray[tuple[Any, ...], np.dtype[np.complex128]]

list_float: list[float] = ...
list_complex: list[complex] = ...

vec_f64: _VecF64 = ...
arr_f64: _ArrF64 = ...

vec_c128: _VecC128 = ...
arr_c128: _ArrC128 = ...

# NOTE: these examples are based on the `solve_ivp` docstring, and use common (suboptimal) type annotation patterns.
###

@type_check_only
def exponential_decay(t: float, y: _ArrF64) -> _ArrF64: ...

assert_type(solve_ivp(exponential_decay, list_float, list_float).t, _VecF64)
assert_type(solve_ivp(exponential_decay, list_float, list_float).y, _MatF64)
assert_type(solve_ivp(exponential_decay, list_float, list_float, args=()).y, _MatF64)

###

@type_check_only
def upward_cannon(t: np.float64, y: _VecF64) -> list[float]: ...
@type_check_only
def hit_ground(t: np.float64, y: _VecF64) -> np.float64: ...

assert_type(solve_ivp(upward_cannon, list_float, list_float).y, _MatF64)
assert_type(solve_ivp(upward_cannon, list_float, list_float, events=hit_ground).y, _MatF64)
assert_type(solve_ivp(upward_cannon, list_float, list_float, events=hit_ground, args=()).y, _MatF64)
assert_type(solve_ivp(upward_cannon, list_float, list_float, events=hit_ground, dense_output=True).y, _MatF64)

###

@type_check_only
def lotkavolterra(t: float, z: npt.NDArray[np.float64], a: float, b: float, c: float, d: float) -> _VecF64: ...

assert_type(solve_ivp(lotkavolterra, list_float, list_float, args=(1.5, 1, 3, 1)).y, _MatF64)
assert_type(solve_ivp(lotkavolterra, list_float, list_float, args=(1.5, 1, 3, 1), dense_output=True).y, _MatF64)

###

@type_check_only
def deriv_vec(t: float, y: onp.ArrayND[np.float64 | np.complex128]) -> onp.ArrayND[np.float64 | np.complex128]: ...

assert_type(solve_ivp(deriv_vec, list_float, list_complex).y, _MatC128)
assert_type(solve_ivp(deriv_vec, list_float, vec_c128).y, _MatC128)
assert_type(solve_ivp(deriv_vec, list_float, arr_c128).y, _MatC128)

assert_type(solve_ivp(deriv_vec, list_float, arr_c128, t_eval=list_float).y, _MatC128)
assert_type(solve_ivp(deriv_vec, list_float, list_complex, t_eval=vec_f64).y, _MatC128)
assert_type(solve_ivp(deriv_vec, list_float, vec_c128, t_eval=arr_f64).y, _MatC128)
