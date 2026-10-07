from collections.abc import Callable
from typing import Any, assert_type, type_check_only

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.integrate import DenseOutput, OdeSolution, OdeSolver, solve_ivp

type _VecF64 = onp.Array1D[np.float64]
type _MatF64 = onp.Array2D[np.float64]
type _ArrF64 = onp.ArrayND[np.float64]
type _VecC128 = onp.Array1D[np.complex128]
type _MatC128 = onp.Array2D[np.complex128]
type _ArrC128 = onp.ArrayND[np.complex128]

list_float: list[float] = ...
list_complex: list[complex] = ...

vec_f64: _VecF64 = ...
arr_f64: _ArrF64 = ...

vec_c128: _VecC128 = ...
arr_c128: _ArrC128 = ...

@type_check_only
def _lv_vec(t: _VecF64, z: _MatF64, a: float, b: float, c: float, d: float) -> _MatF64: ...
@type_check_only
def _lv_event(t: float, z: npt.NDArray[np.float64], a: float, b: float, c: float, d: float) -> float: ...
@type_check_only
def _lv_jac(t: float, z: npt.NDArray[np.float64], a: float, b: float, c: float, d: float) -> _MatF64: ...
@type_check_only
def _rot(t: float, y: _ArrC128, omega: float) -> _ArrC128: ...
@type_check_only
def _rot_vec(t: _VecF64, y: _MatC128, omega: float) -> _MatC128: ...
@type_check_only
def _rot_event(t: float, y: _ArrC128, omega: float) -> float: ...

@type_check_only
class _Euler(OdeSolver[np.float64]):
    def __init__(
        self, fun: Callable[..., Any], t0: float, y0: list[float], t_bound: float, vectorized: bool = False, num_t_steps: int = 15
    ) -> None: ...

# NOTE: these examples are based on the `solve_ivp` docstring, and use common (suboptimal) type annotation patterns.
###

@type_check_only
def exponential_decay(t: float, y: _ArrF64) -> _ArrF64: ...

assert_type(solve_ivp(exponential_decay, list_float, list_float).t, _VecF64)
assert_type(solve_ivp(exponential_decay, list_float, list_float).y, _MatF64)
assert_type(solve_ivp(exponential_decay, list_float, list_float, args=()).y, _MatF64)
assert_type(solve_ivp(exponential_decay, list_float, list_float, method=_Euler, num_t_steps=16).y, _MatF64)

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

assert_type(solve_ivp(lotkavolterra, list_float, list_float, events=_lv_event, args=(1.5, 1, 3, 1)).y, _MatF64)
assert_type(solve_ivp(_lv_vec, list_float, list_float, events=_lv_event, vectorized=True, args=(1.5, 1, 3, 1)).y, _MatF64)
assert_type(solve_ivp(lotkavolterra, list_float, list_float, method="Radau", jac=_lv_jac, args=(1.5, 1, 3, 1)).y, _MatF64)

###

@type_check_only
def deriv_vec(t: float, y: onp.ArrayND[np.float64 | np.complex128]) -> onp.ArrayND[np.float64 | np.complex128]: ...

assert_type(solve_ivp(deriv_vec, list_float, list_complex).y, _MatC128)
assert_type(solve_ivp(deriv_vec, list_float, vec_c128).y, _MatC128)
assert_type(solve_ivp(deriv_vec, list_float, arr_c128).y, _MatC128)

assert_type(solve_ivp(deriv_vec, list_float, arr_c128, t_eval=list_float).y, _MatC128)
assert_type(solve_ivp(deriv_vec, list_float, list_complex, t_eval=vec_f64).y, _MatC128)
assert_type(solve_ivp(deriv_vec, list_float, vec_c128, t_eval=arr_f64).y, _MatC128)

assert_type(solve_ivp(_rot, list_float, list_complex, events=_rot_event, args=(1.0,)).y, _MatC128)
assert_type(solve_ivp(_rot_vec, list_float, list_complex, events=_rot_event, vectorized=True, args=(1.0,)).y, _MatC128)
assert_type(solve_ivp(deriv_vec, list_float, list_complex, dense_output=True).sol, OdeSolution[DenseOutput[np.complex128]] | Any)
