from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy import differentiate
from scipy.differentiate._differentiate import _DerivativeResult0D, _DerivativeResultND, _HessianResult, _JacobianResult

# Test scalar
i32_0d: np.int32
i64_0d: np.int64
f32_0d: np.float32
f64_0d: np.float64

i64_1d: onp.Array1D[np.int64]
f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]

def f_f64_0d(x: np.float64) -> np.float64: ...
def f_f32_0d(x: np.float32) -> np.float32: ...
def f_f64_1d(x: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]: ...
def f_f64_nd(x: onp.ArrayND[np.float64]) -> onp.ArrayND[np.float64]: ...
def f_f64_1nd_0d(x: onp.Array[onp.AtLeast1D, np.float64]) -> np.float64: ...
def f_f64_1nd_nd(x: onp.Array[onp.AtLeast1D, np.float64]) -> onp.ArrayND[np.float64]: ...
def f_f64_0d_arg(x: np.float64, a: float) -> np.float64: ...

###
# derivative

assert_type(differentiate.derivative(f_f64_0d, 1.0), _DerivativeResult0D[np.float64])
assert_type(differentiate.derivative(f_f64_0d, f64_0d), _DerivativeResult0D[np.float64])
assert_type(differentiate.derivative(f_f64_0d, i32_0d), _DerivativeResult0D[np.float64])
assert_type(differentiate.derivative(f_f32_0d, f32_0d), _DerivativeResult0D[np.float32])
assert_type(differentiate.derivative(f_f64_1d, f64_1d), _DerivativeResultND[np.float64, tuple[int]])
assert_type(differentiate.derivative(f_f64_nd, f64_2d), _DerivativeResultND[np.float64, tuple[int, int]])
assert_type(differentiate.derivative(f_f64_0d_arg, 1.0, args=(2.0,)), _DerivativeResult0D[np.float64])
assert_type(differentiate.derivative(f_f64_0d, 1.0, tolerances={"atol": 0.1}), _DerivativeResult0D[np.float64])
assert_type(
    differentiate.derivative(
        f_f64_0d,
        1.0,
        args=(),
        tolerances={"atol": 0.1},
        maxiter=20,
        order=4,
        initial_step=0.1,
        step_factor=1.5,
        step_direction=1,
        preserve_shape=True,
        callback=None,
    ),
    _DerivativeResult0D[np.float64],
)
assert_type(
    differentiate.derivative(f_f64_1d, f64_1d, initial_step=f64_1d, step_direction=i64_1d),
    _DerivativeResultND[np.float64, tuple[int]],
)

res_der_0d = differentiate.derivative(f_f64_0d, 1.0)
assert_type(res_der_0d.success, np.bool_)
assert_type(res_der_0d.status, np.int32)
assert_type(res_der_0d.nfev, np.int32)
assert_type(res_der_0d.nit, np.int32)
assert_type(res_der_0d.x, np.float64)
assert_type(res_der_0d.df, np.float64)
assert_type(res_der_0d.error, np.float64)

res_der_nd = differentiate.derivative(f_f64_1d, f64_1d)
assert_type(res_der_nd.success, onp.Array1D[np.bool_])
assert_type(res_der_nd.status, onp.Array1D[np.int32])
assert_type(res_der_nd.nfev, onp.Array1D[np.int32])
assert_type(res_der_nd.nit, onp.Array1D[np.int32])
assert_type(res_der_nd.x, onp.Array1D[np.float64])
assert_type(res_der_nd.df, onp.Array1D[np.float64])
assert_type(res_der_nd.error, onp.Array1D[np.float64])

###
# jacobian

assert_type(differentiate.jacobian(f_f64_1nd_0d, f64_1d), _JacobianResult[np.float64, onp.AtLeast1D])
assert_type(differentiate.jacobian(f_f64_1nd_nd, f64_1d), _JacobianResult[np.float64, onp.AtLeast1D])
assert_type(
    differentiate.jacobian(
        f_f64_1nd_0d, f64_1d, tolerances={"atol": 0.1}, maxiter=15, order=6, initial_step=0.1, step_factor=1.8, step_direction=0
    ),
    _JacobianResult[np.float64, onp.AtLeast1D],
)
assert_type(
    differentiate.jacobian(f_f64_1nd_nd, f64_2d, initial_step=f64_1d, step_direction=i64_1d),
    _JacobianResult[np.float64, onp.AtLeast1D],
)

res_jac = differentiate.jacobian(f_f64_1nd_0d, f64_1d)
assert_type(res_jac.status, onp.Array[onp.AtLeast1D, np.int32])
assert_type(res_jac.df, onp.Array[onp.AtLeast1D, np.float64])
assert_type(res_jac.error, onp.Array[onp.AtLeast1D, np.float64])
assert_type(res_jac.nit, onp.Array[onp.AtLeast1D, np.int32])
assert_type(res_jac.nfev, onp.Array[onp.AtLeast1D, np.int32])
assert_type(res_jac.success, onp.Array[onp.AtLeast1D, np.bool_])

###
# hessian

assert_type(differentiate.hessian(f_f64_1nd_0d, f64_1d), _HessianResult[np.float64, onp.AtLeast2D])
assert_type(
    differentiate.hessian(
        f_f64_1nd_0d, f64_1d, tolerances={"atol": 0.1}, maxiter=25, order=10, initial_step=0.05, step_factor=2.5
    ),
    _HessianResult[np.float64, onp.AtLeast2D],
)
assert_type(differentiate.hessian(f_f64_1nd_0d, f64_2d, initial_step=f64_1d), _HessianResult[np.float64, onp.AtLeast2D])

res_hes = differentiate.hessian(f_f64_1nd_0d, f64_1d)
assert_type(res_hes.status, onp.Array[onp.AtLeast2D, np.int32])
assert_type(res_hes.error, onp.Array[onp.AtLeast2D, np.float64])
assert_type(res_hes.nfev, onp.Array[onp.AtLeast2D, np.int32])
assert_type(res_hes.success, onp.Array[onp.AtLeast2D, np.bool_])
assert_type(res_hes.ddf, onp.Array[onp.AtLeast2D, np.float64])
