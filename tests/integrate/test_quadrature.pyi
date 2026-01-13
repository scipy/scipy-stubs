# tests for `scipy.integrate._quadrature`

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.integrate import fixed_quad, newton_cotes, qmc_quad, romb, simpson, trapezoid

###

float_1d: list[float]
float_2d: list[list[float]]

complex_1d: list[complex]
complex_2d: list[list[complex]]

i64_1d: onp.Array1D[np.int64]
i64_2d: onp.Array2D[np.int64]
i64_nd: onp.ArrayND[np.int64]

f32_1d: onp.Array1D[np.float32]
f32_2d: onp.Array2D[np.float32]
f32_nd: onp.ArrayND[np.float32]

f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
f64_nd: onp.ArrayND[np.float64]

c64_1d: onp.Array1D[np.complex64]
c64_2d: onp.Array2D[np.complex64]
c64_nd: onp.ArrayND[np.complex64]

c128_1d: onp.Array1D[np.complex128]
c128_2d: onp.Array2D[np.complex128]
c128_nd: onp.ArrayND[np.complex128]

###
# trapezoid

assert_type(trapezoid(float_1d), np.float64)
assert_type(trapezoid(float_2d), onp.Array1D[np.float64])

assert_type(trapezoid(complex_1d), np.complex128)
assert_type(trapezoid(complex_2d), onp.Array1D[np.complex128])

assert_type(trapezoid(i64_1d), np.float64)
assert_type(trapezoid(i64_2d), onp.Array1D[np.float64])
assert_type(trapezoid(i64_nd), Any)

assert_type(trapezoid(f32_1d), np.float32)
assert_type(trapezoid(f32_2d), onp.Array1D[np.float32])
assert_type(trapezoid(f32_nd), Any)

assert_type(trapezoid(f64_1d), np.float64)
assert_type(trapezoid(f64_2d), onp.Array1D[np.float64])
assert_type(trapezoid(f64_nd), Any)

assert_type(trapezoid(c64_1d), np.complex64)
assert_type(trapezoid(c64_2d), onp.Array1D[np.complex64])
assert_type(trapezoid(c64_nd), Any)

assert_type(trapezoid(c128_1d), np.complex128)
assert_type(trapezoid(c128_2d), onp.Array1D[np.complex128])
assert_type(trapezoid(c128_nd), Any)

###
# simpson

assert_type(simpson(float_1d), np.float64)
assert_type(simpson(float_2d), onp.Array1D[np.float64])

assert_type(simpson(complex_1d), np.complex128)
assert_type(simpson(complex_2d), onp.Array1D[np.complex128])

assert_type(simpson(i64_1d), np.float64)
assert_type(simpson(i64_2d), onp.Array1D[np.float64])
assert_type(simpson(i64_nd), Any)

assert_type(simpson(f32_1d), np.float64)  # weird but true
assert_type(simpson(f32_2d), onp.Array1D[np.float32])
assert_type(simpson(f32_nd), Any)

assert_type(simpson(f64_1d), np.float64)
assert_type(simpson(f64_2d), onp.Array1D[np.float64])
assert_type(simpson(f64_nd), Any)

assert_type(simpson(c64_1d), np.complex128)  # weird but true
assert_type(simpson(c64_2d), onp.Array1D[np.complex64])
assert_type(simpson(c64_nd), Any)

assert_type(simpson(c128_1d), np.complex128)
assert_type(simpson(c128_2d), onp.Array1D[np.complex128])
assert_type(simpson(c128_nd), Any)

###
# romb

assert_type(romb(float_1d), np.float64)
assert_type(romb(float_2d), onp.Array1D[np.float64])

assert_type(romb(complex_1d), np.complex128)
assert_type(romb(complex_2d), onp.Array1D[np.complex128])

assert_type(romb(i64_1d), np.float64)
assert_type(romb(i64_2d), onp.Array1D[np.float64])
assert_type(romb(i64_nd), Any)

assert_type(romb(f32_1d), np.float64)
assert_type(romb(f32_2d), onp.Array1D[np.float64])
assert_type(romb(f32_nd), Any)

assert_type(romb(f64_1d), np.float64)
assert_type(romb(f64_2d), onp.Array1D[np.float64])
assert_type(romb(f64_nd), Any)

assert_type(romb(c64_1d), np.complex128)
assert_type(romb(c64_2d), onp.Array1D[np.complex128])
assert_type(romb(c64_nd), Any)

assert_type(romb(c128_1d), np.complex128)
assert_type(romb(c128_2d), onp.Array1D[np.complex128])
assert_type(romb(c128_nd), Any)

###
# cumulative_trapezoid
# TODO(@jorenham): tests

###
# cumulative_simpson
# TODO(@jorenham): tests

###
# fixed_quad

def _f_f16(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float16]: ...
def _f_f32(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]: ...
def _f_f64(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
def _f_f80(x: npt.NDArray[np.float64]) -> npt.NDArray[np.longdouble]: ...
def _f_c64(x: npt.NDArray[np.float64]) -> npt.NDArray[np.complex64]: ...
def _f_c128(x: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]: ...
def _f_c160(x: npt.NDArray[np.float64]) -> npt.NDArray[np.clongdouble]: ...
def _f_m64(x: npt.NDArray[np.float64]) -> npt.NDArray[np.timedelta64]: ...
def _f_obj(x: npt.NDArray[np.float64]) -> npt.NDArray[np.object_]: ...

assert_type(fixed_quad(_f_f16, 0.0, 1.0), tuple[np.float64, None])
assert_type(fixed_quad(_f_f32, 0.0, 1.0), tuple[np.float64, None])
assert_type(fixed_quad(_f_f64, 0.0, 1.0), tuple[np.float64, None])
assert_type(fixed_quad(_f_f80, 0.0, 1.0), tuple[np.longdouble, None])
assert_type(fixed_quad(_f_c64, 0.0, 1.0), tuple[np.complex128, None])
assert_type(fixed_quad(_f_c128, 0.0, 1.0), tuple[np.complex128, None])
assert_type(fixed_quad(_f_c160, 0.0, 1.0), tuple[np.clongdouble, None])
assert_type(fixed_quad(_f_m64, 0.0, 1.0), tuple[np.timedelta64, None])
assert_type(fixed_quad(_f_obj, 0.0, 1.0), tuple[Any, None])

###
# qmc_quad

_r = qmc_quad(lambda x: x, 0, 1)
_r = qmc_quad(lambda x: x, [0, -1], [1, 1])
assert_type(_r.integral, np.float64)
assert_type(_r.standard_error, np.float64)

###
# newton-cotes

assert_type(newton_cotes(5), tuple[onp.Array1D[np.float64], float])
assert_type(newton_cotes(rn=5), tuple[onp.Array1D[np.float64], float])
assert_type(newton_cotes(5, 1), tuple[onp.Array1D[np.float64], float])
assert_type(newton_cotes(5, equal=1), tuple[onp.Array1D[np.float64], float])
assert_type(newton_cotes(rn=5, equal=1), tuple[onp.Array1D[np.float64], float])
