# tests for `scipy.integrate._quadrature`

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import optype.numpy.compat as npc
from optype.test import assert_subtype

from scipy.integrate import cumulative_simpson, cumulative_trapezoid, fixed_quad, newton_cotes, qmc_quad, romb, simpson, trapezoid

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

assert_type(cumulative_trapezoid(float_1d), onp.Array1D[np.float64])
assert_type(cumulative_trapezoid(float_2d), onp.Array2D[np.float64])

assert_type(cumulative_trapezoid(complex_1d), onp.Array1D[np.complex128])
assert_type(cumulative_trapezoid(complex_2d), onp.Array2D[np.complex128])

assert_type(cumulative_trapezoid(i64_1d), onp.Array1D[np.float64])
assert_type(cumulative_trapezoid(i64_2d), onp.Array2D[np.float64])
assert_type(cumulative_trapezoid(i64_nd), onp.ArrayND[np.float64])

assert_type(cumulative_trapezoid(f32_1d), onp.Array1D[np.float32])
assert_type(cumulative_trapezoid(f32_2d), onp.Array2D[np.float32])
assert_type(cumulative_trapezoid(f32_nd), onp.ArrayND[np.float32])

assert_type(cumulative_trapezoid(f64_1d), onp.Array1D[np.float64])
assert_type(cumulative_trapezoid(f64_2d), onp.Array2D[np.float64])
assert_type(cumulative_trapezoid(f64_nd), onp.ArrayND[np.float64])

assert_type(cumulative_trapezoid(c64_1d), onp.Array1D[np.complex64])
assert_type(cumulative_trapezoid(c64_2d), onp.Array2D[np.complex64])
assert_type(cumulative_trapezoid(c64_nd), onp.ArrayND[np.complex64])

assert_type(cumulative_trapezoid(c128_1d), onp.Array1D[np.complex128])
assert_type(cumulative_trapezoid(c128_2d), onp.Array2D[np.complex128])
assert_type(cumulative_trapezoid(c128_nd), onp.ArrayND[np.complex128])

###
# cumulative_simpson (same as above)

assert_type(cumulative_simpson(float_1d), onp.Array1D[np.float64])
assert_type(cumulative_simpson(float_2d), onp.Array2D[np.float64])

assert_type(cumulative_simpson(complex_1d), onp.Array1D[np.complex128])
assert_type(cumulative_simpson(complex_2d), onp.Array2D[np.complex128])

assert_type(cumulative_simpson(i64_1d), onp.Array1D[np.float64])
assert_type(cumulative_simpson(i64_2d), onp.Array2D[np.float64])
assert_type(cumulative_simpson(i64_nd), onp.ArrayND[np.float64])

assert_type(cumulative_simpson(f32_1d), onp.Array1D[np.float32])
assert_type(cumulative_simpson(f32_2d), onp.Array2D[np.float32])
assert_type(cumulative_simpson(f32_nd), onp.ArrayND[np.float32])

assert_type(cumulative_simpson(f64_1d), onp.Array1D[np.float64])
assert_type(cumulative_simpson(f64_2d), onp.Array2D[np.float64])
assert_type(cumulative_simpson(f64_nd), onp.ArrayND[np.float64])

assert_type(cumulative_simpson(c64_1d), onp.Array1D[np.complex64])
assert_type(cumulative_simpson(c64_2d), onp.Array2D[np.complex64])
assert_type(cumulative_simpson(c64_nd), onp.ArrayND[np.complex64])

assert_type(cumulative_simpson(c128_1d), onp.Array1D[np.complex128])
assert_type(cumulative_simpson(c128_2d), onp.Array2D[np.complex128])
assert_type(cumulative_simpson(c128_nd), onp.ArrayND[np.complex128])

###
# fixed_quad

def _f_f16_1(x: npt.NDArray[np.float64]) -> onp.Array1D[np.float16]: ...
def _f_f32_1(x: npt.NDArray[np.float64]) -> onp.Array1D[np.float32]: ...
def _f_f64_1(x: npt.NDArray[np.float64]) -> onp.Array1D[np.float64]: ...
def _f_f80_1(x: npt.NDArray[np.float64]) -> onp.Array1D[npc.floating80]: ...
def _f_c64_1(x: npt.NDArray[np.float64]) -> onp.Array1D[np.complex64]: ...
def _f_c128_1(x: npt.NDArray[np.float64]) -> onp.Array1D[np.complex128]: ...
def _f_c160_1(x: npt.NDArray[np.float64]) -> onp.Array1D[npc.complexfloating160]: ...
def _f_m64_1(x: npt.NDArray[np.float64]) -> onp.Array1D[np.timedelta64]: ...
def _f_f16_2(x: npt.NDArray[np.float64]) -> onp.Array2D[np.float16]: ...
def _f_f32_2(x: npt.NDArray[np.float64]) -> onp.Array2D[np.float32]: ...
def _f_f64_2(x: npt.NDArray[np.float64]) -> onp.Array2D[np.float64]: ...
def _f_f80_2(x: npt.NDArray[np.float64]) -> onp.Array2D[npc.floating80]: ...
def _f_c64_2(x: npt.NDArray[np.float64]) -> onp.Array2D[np.complex64]: ...
def _f_c128_2(x: npt.NDArray[np.float64]) -> onp.Array2D[np.complex128]: ...
def _f_c160_2(x: npt.NDArray[np.float64]) -> onp.Array2D[npc.complexfloating160]: ...
def _f_m64_2(x: npt.NDArray[np.float64]) -> onp.Array2D[np.timedelta64]: ...
def _f_f16_n(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float16]: ...
def _f_f32_n(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]: ...
def _f_f64_n(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
def _f_f80_n(x: npt.NDArray[np.float64]) -> npt.NDArray[npc.floating80]: ...
def _f_c64_n(x: npt.NDArray[np.float64]) -> npt.NDArray[np.complex64]: ...
def _f_c128_n(x: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]: ...
def _f_c160_n(x: npt.NDArray[np.float64]) -> npt.NDArray[npc.complexfloating160]: ...
def _f_m64_n(x: npt.NDArray[np.float64]) -> npt.NDArray[np.timedelta64]: ...
def _f_obj_n(x: npt.NDArray[np.float64]) -> npt.NDArray[np.object_]: ...

# https://github.com/facebook/pyrefly/issues/3974
assert_type(fixed_quad(_f_f16_1, 0.0, 1.0), tuple[np.float64, None])  # pyrefly:ignore[assert-type]
assert_type(fixed_quad(_f_f32_1, 0.0, 1.0), tuple[np.float64, None])  # pyrefly:ignore[assert-type]
assert_type(fixed_quad(_f_f64_1, 0.0, 1.0), tuple[np.float64, None])  # pyrefly:ignore[assert-type]
assert_type(fixed_quad(_f_c64_1, 0.0, 1.0), tuple[np.complex128, None])  # pyrefly:ignore[assert-type]
assert_type(fixed_quad(_f_f80_1, 0.0, 1.0), tuple[npc.floating80, None])  # pyrefly:ignore[assert-type]
assert_type(fixed_quad(_f_c128_1, 0.0, 1.0), tuple[np.complex128, None])  # pyrefly:ignore[assert-type]
assert_type(fixed_quad(_f_c160_1, 0.0, 1.0), tuple[npc.complexfloating160, None])  # pyrefly:ignore[assert-type]
assert_subtype[tuple[np.timedelta64, None]](fixed_quad(_f_m64_1, 0.0, 1.0))  # pyrefly:ignore[bad-argument-type]
assert_type(fixed_quad(_f_f16_2, 0.0, 1.0), tuple[onp.Array1D[np.float64], None])  # pyrefly:ignore[assert-type]
assert_type(fixed_quad(_f_f32_2, 0.0, 1.0), tuple[onp.Array1D[np.float64], None])  # pyrefly:ignore[assert-type]
assert_type(fixed_quad(_f_f64_2, 0.0, 1.0), tuple[onp.Array1D[np.float64], None])  # pyrefly:ignore[assert-type]
assert_type(fixed_quad(_f_c64_2, 0.0, 1.0), tuple[onp.Array1D[np.complex128], None])  # pyrefly:ignore[assert-type]
assert_type(fixed_quad(_f_f80_2, 0.0, 1.0), tuple[onp.Array1D[npc.floating80], None])  # pyrefly:ignore[assert-type]
assert_type(fixed_quad(_f_c128_2, 0.0, 1.0), tuple[onp.Array1D[np.complex128], None])  # pyrefly:ignore[assert-type]
assert_type(fixed_quad(_f_c160_2, 0.0, 1.0), tuple[onp.Array1D[npc.complexfloating160], None])  # pyrefly:ignore[assert-type]
assert_subtype[tuple[onp.Array1D[np.timedelta64], None]](fixed_quad(_f_m64_2, 0.0, 1.0))
assert_subtype[tuple[onp.ArrayND[np.float64] | np.float64, None]](fixed_quad(_f_f16_n, 0.0, 1.0))
assert_subtype[tuple[onp.ArrayND[np.float64] | np.float64, None]](fixed_quad(_f_f32_n, 0.0, 1.0))
assert_subtype[tuple[onp.ArrayND[np.float64] | np.float64, None]](fixed_quad(_f_f64_n, 0.0, 1.0))
assert_subtype[tuple[onp.ArrayND[np.complex128] | np.complex128, None]](fixed_quad(_f_c64_n, 0.0, 1.0))
assert_subtype[tuple[onp.ArrayND[npc.floating80] | npc.floating80, None]](fixed_quad(_f_f80_n, 0.0, 1.0))
assert_subtype[tuple[onp.ArrayND[np.complex128] | np.complex128, None]](fixed_quad(_f_c128_n, 0.0, 1.0))
assert_subtype[tuple[onp.ArrayND[npc.complexfloating160] | npc.complexfloating160, None]](fixed_quad(_f_c160_n, 0.0, 1.0))
assert_subtype[tuple[onp.ArrayND[np.timedelta64] | np.timedelta64, None]](fixed_quad(_f_m64_n, 0.0, 1.0))
assert_type(fixed_quad(_f_obj_n, 0.0, 1.0), tuple[Any, None])

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
