from typing import Literal, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy.compat as npc

from scipy.optimize import fmin_l_bfgs_b

i64_1d: np.ndarray[tuple[int], np.dtype[np.int64]]
f64_1d: np.ndarray[tuple[int], np.dtype[np.float64]]

def f2(theta: npt.NDArray[np.float64], arg1: npt.NDArray[np.float64], arg2: int) -> float: ...
def g2(theta: npt.NDArray[np.float64], arg1: npt.NDArray[np.float64], arg2: int) -> npt.NDArray[npc.floating]: ...
def f0(theta: npt.NDArray[np.float64]) -> float: ...
def g0(theta: npt.NDArray[np.float64]) -> list[float]: ...
def fg0(theta: npt.NDArray[np.float64]) -> tuple[float, list[float]]: ...

###

res_f2_fprime = fmin_l_bfgs_b(f2, f64_1d, g2, (f64_1d, True))
assert_type(res_f2_fprime[0], np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(res_f2_fprime[1], float)
assert_type(res_f2_fprime[2]["grad"], np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(res_f2_fprime[2]["task"], str)
assert_type(res_f2_fprime[2]["funcalls"], int)
assert_type(res_f2_fprime[2]["nit"], int)
assert_type(res_f2_fprime[2]["warnflag"], Literal[0, 1, 2])

res_f2_approx = fmin_l_bfgs_b(f2, f64_1d, args=(f64_1d, True), approx_grad=True)
assert_type(res_f2_approx[0], np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(res_f2_approx[1], float)

fmin_l_bfgs_b(fg0, f64_1d)
fmin_l_bfgs_b(fg0, f64_1d, fprime=g0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
fmin_l_bfgs_b(fg0, f64_1d, (1,))  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
fmin_l_bfgs_b(f0, f64_1d)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
fmin_l_bfgs_b(f0, f64_1d, fprime=g0)
fmin_l_bfgs_b(f0, f64_1d, approx_grad=1)
