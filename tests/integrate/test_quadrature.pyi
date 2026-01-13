# tests for `scipy.integrate._quadrature`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.integrate import simpson, trapezoid

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
