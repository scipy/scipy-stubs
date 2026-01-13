from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

from scipy.integrate import BDF, DOP853, LSODA, RK23, RK45, DenseOutput, OdeSolution, OdeSolver, Radau

###

def _f_f64(t: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
def _f_c128(t: float, y: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]: ...

_f64_nd: npt.NDArray[np.float64]
_c128_nd: npt.NDArray[np.complex128]

###
# BDF
assert_type(BDF(_f_f64, 0.0, _f64_nd, 10.0), BDF[np.float64])
assert_type(BDF(_f_c128, 0.0, _c128_nd, 10.0), BDF[np.complex128])
# DOP853
assert_type(DOP853(_f_f64, 0.0, _f64_nd, 10.0), DOP853[np.float64])
assert_type(DOP853(_f_c128, 0.0, _c128_nd, 10.0), DOP853[np.complex128])
# RK23
assert_type(RK23(_f_f64, 0.0, _f64_nd, 10.0), RK23[np.float64])
assert_type(RK23(_f_c128, 0.0, _c128_nd, 10.0), RK23[np.complex128])
# RK45
assert_type(RK45(_f_f64, 0.0, _f64_nd, 10.0), RK45[np.float64])
assert_type(RK45(_f_c128, 0.0, _c128_nd, 10.0), RK45[np.complex128])

###
# OdeSolver
assert_type(OdeSolver(_f_f64, 0.0, _f64_nd, 10.0, False), OdeSolver[np.float64])
assert_type(OdeSolver(_f_c128, 0.0, _c128_nd, 10.0, False, support_complex=True), OdeSolver[np.complex128])
# LSODA
assert_type(LSODA(_f_f64, 0.0, _f64_nd, 10.0), LSODA)
# LSODA
assert_type(Radau(_f_f64, 0.0, _f64_nd, 10.0), Radau)

###
# DenseOutput
assert_type(DenseOutput(0, 1), DenseOutput[np.float64 | Any])
assert_type(DenseOutput[np.float64](0, 1), DenseOutput[np.float64])
# OdeSolution
interpolants_f64: list[DenseOutput[np.float64]]
interpolants_c128: list[DenseOutput[np.complex128]]
assert_type(OdeSolution([0.0, 1.0], interpolants_f64), OdeSolution[DenseOutput[np.float64]])
assert_type(OdeSolution([0.0, 1.0], interpolants_c128), OdeSolution[DenseOutput[np.complex128]])
