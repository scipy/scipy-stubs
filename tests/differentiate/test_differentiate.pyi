from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.differentiate import derivative, hessian, jacobian
from scipy.differentiate._differentiate import _DerivativeResult0D, _DerivativeResultND, _HessianResult, _JacobianResult

###

_i32_0d: np.int32
_i64_0d: np.int64
_f32_0d: np.float32
_f64_0d: np.float64

_i64_1d: onp.Array1D[np.int64]
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]

_py_f_1d: list[float]

def _f_f32_0d(x: np.float32) -> np.float32: ...
def _f_f64_0d(x: np.float64) -> np.float64: ...
def _f_f64_1d(x: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]: ...
def _f_f64_nd(x: onp.ArrayND[np.float64]) -> onp.ArrayND[np.float64]: ...
def _f_f64_nd_0d(x: onp.ArrayND[np.float64]) -> np.float64: ...
def _f_f64_nd_nd(x: onp.ArrayND[np.float64]) -> onp.ArrayND[np.float64]: ...
def _f_f64_0d_arg(x: np.float64, a: float) -> np.float64: ...

###
# derivative

assert_type(derivative(_f_f64_0d, 1.0), _DerivativeResult0D[np.float64])
assert_type(derivative(_f_f64_0d, _f64_0d), _DerivativeResult0D[np.float64])
assert_type(derivative(_f_f64_0d, _i32_0d), _DerivativeResult0D[np.float64])
assert_type(derivative(_f_f32_0d, _f32_0d), _DerivativeResult0D[np.float32])

assert_type(derivative(_f_f64_1d, _f64_1d), _DerivativeResultND[np.float64, tuple[int]])
assert_type(derivative(_f_f64_nd, _f64_2d), _DerivativeResultND[np.float64, tuple[Any, ...]])
assert_type(
    derivative(_f_f64_1d, _f64_1d, initial_step=_f64_1d, step_direction=_i64_1d), _DerivativeResultND[np.float64, tuple[int]]
)

# the dtype follows `x`, not `f`
assert_type(derivative(np.exp, _f64_1d), _DerivativeResultND[np.float64, tuple[int]])
assert_type(derivative(np.exp, _i64_1d), _DerivativeResultND[np.float64, tuple[int]])
assert_type(derivative(np.exp, _py_f_1d), _DerivativeResultND[np.float64, tuple[int]])
assert_type(jacobian(np.exp, _f64_1d), _JacobianResult[np.float64, tuple[int, *tuple[Any, ...]]])
assert_type(hessian(np.exp, _f64_1d), _HessianResult[np.float64, tuple[int, int, *tuple[Any, ...]]])

assert_type(derivative(_f_f64_0d_arg, 1.0, args=(2.0,)), _DerivativeResult0D[np.float64])
assert_type(derivative(_f_f64_0d, 1.0, tolerances={"atol": 0.1}), _DerivativeResult0D[np.float64])
assert_type(
    derivative(
        _f_f64_0d,
        1.0,
        args=(),
        tolerances={"atol": 0.1},
        maxiter=20,
        order=4,
        initial_step=0.1,
        step_factor=1.5,
        step_direction=1,
        preserve_shape=False,
        callback=None,
    ),
    _DerivativeResult0D[np.float64],
)

res_der_0d = derivative(_f_f64_0d, 1.0)
assert_type(res_der_0d.success, np.bool)
assert_type(res_der_0d.status, np.int32)
assert_type(res_der_0d.nfev, np.int32)
assert_type(res_der_0d.nit, np.int32)
assert_type(res_der_0d.x, np.float64)
assert_type(res_der_0d.df, np.float64)
assert_type(res_der_0d.error, np.float64)

res_der_nd = derivative(_f_f64_1d, _f64_1d)
assert_type(res_der_nd.success, onp.Array1D[np.bool])
assert_type(res_der_nd.status, onp.Array1D[np.int32])
assert_type(res_der_nd.nfev, onp.Array1D[np.int32])
assert_type(res_der_nd.nit, onp.Array1D[np.int32])
assert_type(res_der_nd.x, onp.Array1D[np.float64])
assert_type(res_der_nd.df, onp.Array1D[np.float64])
assert_type(res_der_nd.error, onp.Array1D[np.float64])

###
# jacobian

assert_type(jacobian(_f_f64_nd_0d, _f64_1d), _JacobianResult[np.float64, tuple[int, *tuple[Any, ...]]])
assert_type(jacobian(_f_f64_nd_nd, _f64_1d), _JacobianResult[np.float64, tuple[int, *tuple[Any, ...]]])
assert_type(
    jacobian(
        _f_f64_nd_0d, _f64_1d, tolerances={"atol": 0.1}, maxiter=15, order=6, initial_step=0.1, step_factor=1.8, step_direction=0
    ),
    _JacobianResult[np.float64, tuple[int, *tuple[Any, ...]]],
)
assert_type(
    jacobian(_f_f64_nd_nd, _f64_2d, initial_step=_f64_1d, step_direction=_i64_1d),
    _JacobianResult[np.float64, tuple[int, *tuple[Any, ...]]],
)

_res_jac = jacobian(_f_f64_nd_0d, _f64_1d)
assert_type(_res_jac.status, onp.ArrayND[np.int32, tuple[int, *tuple[Any, ...]]])
assert_type(_res_jac.df, onp.ArrayND[np.float64, tuple[int, *tuple[Any, ...]]])
assert_type(_res_jac.error, onp.ArrayND[np.float64, tuple[int, *tuple[Any, ...]]])
assert_type(_res_jac.nit, onp.ArrayND[np.int32, tuple[int, *tuple[Any, ...]]])
assert_type(_res_jac.nfev, onp.ArrayND[np.int32, tuple[int, *tuple[Any, ...]]])
assert_type(_res_jac.success, onp.ArrayND[np.bool, tuple[int, *tuple[Any, ...]]])

###
# hessian

assert_type(hessian(_f_f64_nd_0d, _f64_1d), _HessianResult[np.float64, tuple[int, int, *tuple[Any, ...]]])
assert_type(
    hessian(_f_f64_nd_0d, _f64_1d, tolerances={"atol": 0.1}, maxiter=25, order=10, initial_step=0.05, step_factor=2.5),
    _HessianResult[np.float64, tuple[int, int, *tuple[Any, ...]]],
)
assert_type(hessian(_f_f64_nd_0d, _f64_2d, initial_step=_f64_1d), _HessianResult[np.float64, tuple[int, int, *tuple[Any, ...]]])

_res_hes = hessian(_f_f64_nd_0d, _f64_1d)
assert_type(_res_hes.status, onp.ArrayND[np.int32, tuple[int, int, *tuple[Any, ...]]])
assert_type(_res_hes.error, onp.ArrayND[np.float64, tuple[int, int, *tuple[Any, ...]]])
assert_type(_res_hes.nfev, onp.ArrayND[np.int64, tuple[int, int, *tuple[Any, ...]]])
assert_type(_res_hes.success, onp.ArrayND[np.bool, tuple[int, int, *tuple[Any, ...]]])
assert_type(_res_hes.ddf, onp.ArrayND[np.float64, tuple[int, int, *tuple[Any, ...]]])
