from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import chirp, gausspulse, sawtooth, square, sweep_poly, unit_impulse

###

_bool: bool
_shape_2d: tuple[int, int]

_f32_0d: np.float32
_c128_0d: np.complex128
_i64_1d: onp.Array1D[np.int64]
_i64_2d: onp.Array2D[np.int64]
_f16_1d: onp.Array1D[np.float16]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f80_1d: onp.Array1D[np.float128]
_c64_1d: onp.Array1D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]
_c128_2d: onp.Array2D[np.complex128]

_f32_0d_or_2d: np.float32 | onp.Array2D[np.float32]

type _WA = tuple[int] | tuple[Any, ...]

###

# sawtooth

assert_type(sawtooth(_f32_0d), onp.Array0D[np.float32])
assert_type(sawtooth(_f16_1d), onp.Array1D[np.float16])
assert_type(sawtooth(_f32_1d), onp.Array1D[np.float32])
assert_type(sawtooth(_f64_1d), onp.Array1D[np.float64])
assert_type(sawtooth(_f64_1d, 0.5), onp.Array1D[np.float64])
assert_type(sawtooth(_f64_2d), onp.Array2D[np.float64])
assert_type(sawtooth(_f80_1d), onp.Array1D[np.float128])
assert_type(sawtooth(_f32_0d_or_2d), onp.ArrayND[np.float32, _WA])
assert_type(sawtooth(1), onp.Array0D[np.float64])
assert_type(sawtooth(1.0), onp.Array0D[np.float64])
assert_type(sawtooth(_i64_1d), onp.Array1D[np.float64])
assert_type(sawtooth(_i64_2d), onp.Array2D[np.float64])
assert_type(sawtooth([1, 2]), onp.ArrayND[np.float64, _WA])
assert_type(sawtooth([1.0, 2.0]), onp.ArrayND[np.float64, _WA])
assert_type(sawtooth(_f64_1d, _f64_1d), onp.ArrayND[np.float64 | Any, _WA])

# square

assert_type(square(_f32_0d), onp.Array0D[np.float32])
assert_type(square(_f16_1d), onp.Array1D[np.float16])
assert_type(square(_f32_1d), onp.Array1D[np.float32])
assert_type(square(_f64_1d), onp.Array1D[np.float64])
assert_type(square(_f64_1d, 0.5), onp.Array1D[np.float64])
assert_type(square(_f64_2d), onp.Array2D[np.float64])
assert_type(square(_f80_1d), onp.Array1D[np.float128])
assert_type(square(_f32_0d_or_2d), onp.ArrayND[np.float32, _WA])
assert_type(square(1), onp.Array0D[np.float64])
assert_type(square(1.0), onp.Array0D[np.float64])
assert_type(square(_i64_1d), onp.Array1D[np.float64])
assert_type(square(_i64_2d), onp.Array2D[np.float64])
assert_type(square([1, 2]), onp.ArrayND[np.float64, _WA])
assert_type(square([1.0, 2.0]), onp.ArrayND[np.float64, _WA])
assert_type(square(_f64_1d, _f64_1d), onp.ArrayND[np.float64 | Any, _WA])

# chirp

assert_type(chirp(1.0, 0.0, 1.0, 1.0), np.float64)
assert_type(chirp(1.0, 0.0, 1.0, 1.0, complex=False), np.float64)
assert_type(chirp(_f64_1d, 0.0, 1.0, 1.0), onp.Array1D[np.float64])
assert_type(chirp(_f64_1d, 0.0, 1.0, 1.0, complex=False), onp.Array1D[np.float64])
assert_type(chirp(_f64_2d, 0.0, 1.0, 1.0), onp.Array2D[np.float64])
assert_type(chirp([0.0, 1.0], 0.0, 1.0, 1.0), onp.ArrayND[np.float64, _WA])
assert_type(chirp(1.0, 0.0, 1.0, 1.0, complex=True), np.complex128)
assert_type(chirp(_f64_1d, 0.0, 1.0, 1.0, complex=True), onp.Array1D[np.complex128])
assert_type(chirp(_f64_2d, 0.0, 1.0, 1.0, complex=True), onp.Array2D[np.complex128])
assert_type(chirp([0.0, 1.0], 0.0, 1.0, 1.0, complex=True), onp.ArrayND[np.complex128, _WA])
assert_type(chirp(_c128_0d, 0.0, 1.0, 1.0), np.complex128)
assert_type(chirp(_c64_1d, 0.0, 1.0, 1.0), onp.Array1D[np.complex128])
assert_type(chirp(_c128_1d, 0.0, 1.0, 1.0), onp.Array1D[np.complex128])
assert_type(chirp(_c128_2d, 0.0, 1.0, 1.0), onp.Array2D[np.complex128])
assert_type(chirp(_f64_2d, 0.0, 1.0, 1.0, complex=_bool), onp.Array2D[Any])
assert_type(chirp([0.0, 1.0], 0.0, 1.0, 1.0, complex=_bool), onp.ArrayND[Any, _WA])

# sweep_poly

assert_type(sweep_poly(1.0, _f64_1d), np.float64)
assert_type(sweep_poly(_f64_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(sweep_poly(_f64_1d, np.poly1d([1.0, 0.0])), onp.Array1D[np.float64])
assert_type(sweep_poly(_f64_2d, _f64_1d), onp.Array2D[np.float64])
assert_type(sweep_poly([1.0, 2.0], _f64_1d), onp.ArrayND[np.float64, _WA])

# unit_impulse

assert_type(unit_impulse(10), onp.Array1D[np.float64])
assert_type(unit_impulse(10, 3), onp.Array1D[np.float64])
assert_type(unit_impulse(10, "mid"), onp.Array1D[np.float64])
assert_type(unit_impulse(_shape_2d), onp.Array2D[np.float64])
assert_type(unit_impulse([3, 3]), onp.ArrayND[np.float64, _WA])
assert_type(unit_impulse(_i64_1d), onp.ArrayND[np.float64, _WA])
assert_type(unit_impulse(10, 3, np.dtype(np.int32)), onp.Array1D[np.int32])
assert_type(unit_impulse(_shape_2d, 3, np.dtype(np.complex128)), onp.Array2D[np.complex128])
assert_type(unit_impulse([3, 3], 3, np.dtype(np.complex128)), onp.ArrayND[np.complex128, _WA])

# gausspulse

assert_type(gausspulse("cutoff"), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, False, False), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, False, True), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, True, False), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, True, True), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, retquad=False, retenv=False), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, retquad=False, retenv=True), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, retquad=True, retenv=False), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, retquad=True, retenv=True), np.float64)
assert_type(gausspulse(t="cutoff"), np.float64)
assert_type(gausspulse(t="cutoff", retquad=False, retenv=False), np.float64)
assert_type(gausspulse(t="cutoff", retquad=False, retenv=True), np.float64)
assert_type(gausspulse(t="cutoff", retquad=True, retenv=False), np.float64)
assert_type(gausspulse(t="cutoff", retquad=True, retenv=True), np.float64)

assert_type(gausspulse(1.0), np.float64)
assert_type(gausspulse(1.0, 1.0, 1.0, 1.0, 1.0, False, False), np.float64)
assert_type(gausspulse(1.0, 1.0, 1.0, 1.0, 1.0, retquad=False, retenv=False), np.float64)
assert_type(gausspulse(1.0, 1.0, 1.0, 1.0, 1.0, retquad=False, retenv=True), tuple[np.float64, np.float64])
assert_type(gausspulse(1.0, 1.0, 1.0, 1.0, 1.0, retquad=True, retenv=False), tuple[np.float64, np.float64])
assert_type(gausspulse(1.0, 1.0, 1.0, 1.0, 1.0, retquad=True, retenv=True), tuple[np.float64, np.float64, np.float64])
assert_type(gausspulse(t=1.0), np.float64)
assert_type(gausspulse(t=1.0, retquad=False, retenv=False), np.float64)
assert_type(gausspulse(t=1.0, retquad=False, retenv=True), tuple[np.float64, np.float64])
assert_type(gausspulse(t=1.0, retquad=True, retenv=False), tuple[np.float64, np.float64])
assert_type(gausspulse(t=1.0, retquad=True, retenv=True), tuple[np.float64, np.float64, np.float64])

assert_type(gausspulse(_f64_1d), onp.Array1D[np.float64])
assert_type(gausspulse(_f64_1d, 1.0, 1.0, 1.0, 1.0, False, False), onp.Array1D[np.float64])
assert_type(gausspulse(_f64_1d, 1.0, 1.0, 1.0, 1.0, retquad=False, retenv=False), onp.Array1D[np.float64])
assert_type(
    gausspulse(_f64_1d, 1.0, 1.0, 1.0, 1.0, retquad=False, retenv=True), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]]
)
assert_type(
    gausspulse(_f64_1d, 1.0, 1.0, 1.0, 1.0, retquad=True, retenv=False), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]]
)
assert_type(
    gausspulse(_f64_1d, 1.0, 1.0, 1.0, 1.0, retquad=True, retenv=True),
    tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64]],
)
assert_type(gausspulse(t=_f64_2d), onp.Array2D[np.float64])
assert_type(gausspulse(t=_f64_2d, retquad=False, retenv=False), onp.Array2D[np.float64])
assert_type(gausspulse(t=_f64_2d, retquad=False, retenv=True), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(gausspulse(t=_f64_2d, retquad=True, retenv=False), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(
    gausspulse(t=_f64_2d, retquad=True, retenv=True),
    tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64]],
)
