from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import chirp, gausspulse, sawtooth, square, sweep_poly, unit_impulse

###

_f64_1d: onp.Array1D[np.float64]
_c64_1d: onp.Array1D[np.complex64]
_c128_0d: np.complex128
_c128_1d: onp.Array1D[np.complex128]

###

# sawtooth

assert_type(sawtooth(1.0), onp.ArrayND[np.float64])
assert_type(sawtooth(_f64_1d), onp.ArrayND[np.float64])
assert_type(sawtooth(_f64_1d, 0.5), onp.ArrayND[np.float64])

# square

assert_type(square(1.0), onp.ArrayND[np.float64])
assert_type(square(_f64_1d), onp.ArrayND[np.float64])
assert_type(square(_f64_1d, 0.5), onp.ArrayND[np.float64])

# chirp

assert_type(chirp(1.0, 0.0, 1.0, 1.0), np.float64)
assert_type(chirp(1.0, 0.0, 1.0, 1.0, complex=False), np.float64)
assert_type(chirp(_f64_1d, 0.0, 1.0, 1.0), onp.ArrayND[np.float64])
assert_type(chirp(_f64_1d, 0.0, 1.0, 1.0, complex=False), onp.ArrayND[np.float64])
assert_type(chirp(1.0, 0.0, 1.0, 1.0, complex=True), np.complex128)
assert_type(chirp(_f64_1d, 0.0, 1.0, 1.0, complex=True), onp.ArrayND[np.complex128])
assert_type(chirp(_c128_0d, 0.0, 1.0, 1.0), np.complex128)
assert_type(chirp(_c128_1d, 0.0, 1.0, 1.0), onp.ArrayND[np.complex128])
assert_type(chirp(_c64_1d, 0.0, 1.0, 1.0), onp.ArrayND[np.complex128])

# sweep_poly

assert_type(sweep_poly(1.0, _f64_1d), onp.ArrayND[np.float64])
assert_type(sweep_poly(_f64_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(sweep_poly(_f64_1d, np.poly1d([1.0, 0.0])), onp.ArrayND[np.float64])

# unit_impulse

assert_type(unit_impulse(10), onp.ArrayND[np.float64])
assert_type(unit_impulse((3, 3)), onp.ArrayND[np.float64])
assert_type(unit_impulse(10, 3), onp.ArrayND[np.float64])
assert_type(unit_impulse(10, "mid"), onp.ArrayND[np.float64])
assert_type(unit_impulse(10, 3, np.dtype(np.int32)), onp.ArrayND[np.int32])
assert_type(unit_impulse(10, 3, np.dtype(np.complex128)), onp.ArrayND[np.complex128])

# gausspulse

assert_type(gausspulse(_f64_1d), onp.ArrayND[np.float64])
assert_type(gausspulse(_f64_1d, 1.0, 1.0, 1.0, 1.0, False, False), onp.ArrayND[np.float64])
assert_type(gausspulse(_f64_1d, 1.0, 1.0, 1.0, 1.0, True, False), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(gausspulse(_f64_1d, 1.0, 1.0, 1.0, 1.0, False, True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(
    gausspulse(_f64_1d, 1.0, 1.0, 1.0, 1.0, True, True),
    tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]],
)
assert_type(gausspulse(t=_f64_1d), onp.ArrayND[np.float64])
assert_type(gausspulse(t=_f64_1d, retquad=False, retenv=False), onp.ArrayND[np.float64])
assert_type(gausspulse(t=_f64_1d, retquad=True, retenv=False), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(gausspulse(t=_f64_1d, retquad=False, retenv=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(
    gausspulse(t=_f64_1d, retquad=True, retenv=True),
    tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]],
)

assert_type(gausspulse(_f64_1d, 1.0, 1.0, 1.0, 1.0, retquad=False, retenv=False), onp.ArrayND[np.float64])
assert_type(
    gausspulse(_f64_1d, 1.0, 1.0, 1.0, 1.0, retquad=True, retenv=False), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]
)
assert_type(
    gausspulse(_f64_1d, 1.0, 1.0, 1.0, 1.0, retquad=False, retenv=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]
)
assert_type(
    gausspulse(_f64_1d, 1.0, 1.0, 1.0, 1.0, retquad=True, retenv=True),
    tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]],
)

assert_type(gausspulse(1.0), np.float64)
assert_type(gausspulse(1.0, 1.0, 1.0, 1.0, 1.0, False, False), np.float64)
assert_type(gausspulse(1.0, 1.0, 1.0, 1.0, 1.0, True, False), tuple[np.float64, np.float64])
assert_type(gausspulse(1.0, 1.0, 1.0, 1.0, 1.0, False, True), tuple[np.float64, np.float64])
assert_type(gausspulse(1.0, 1.0, 1.0, 1.0, 1.0, True, True), tuple[np.float64, np.float64, np.float64])

assert_type(gausspulse(t=1.0), np.float64)
assert_type(gausspulse(t=1.0, retquad=False, retenv=False), np.float64)
assert_type(gausspulse(t=1.0, retquad=True, retenv=False), tuple[np.float64, np.float64])
assert_type(gausspulse(t=1.0, retquad=False, retenv=True), tuple[np.float64, np.float64])
assert_type(gausspulse(t=1.0, retquad=True, retenv=True), tuple[np.float64, np.float64, np.float64])

# Mixed positional and keyword arguments
assert_type(gausspulse(1.0, 1.0, 1.0, 1.0, 1.0, retquad=False, retenv=False), np.float64)
assert_type(gausspulse(1.0, 1.0, 1.0, 1.0, 1.0, retquad=True, retenv=False), tuple[np.float64, np.float64])
assert_type(gausspulse(1.0, 1.0, 1.0, 1.0, 1.0, retquad=False, retenv=True), tuple[np.float64, np.float64])
assert_type(gausspulse(1.0, 1.0, 1.0, 1.0, 1.0, retquad=True, retenv=True), tuple[np.float64, np.float64, np.float64])

assert_type(gausspulse("cutoff"), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, False, False), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, True, False), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, False, True), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, True, True), np.float64)
assert_type(gausspulse(t="cutoff"), np.float64)
assert_type(gausspulse(t="cutoff", retquad=False, retenv=False), np.float64)
assert_type(gausspulse(t="cutoff", retquad=True, retenv=False), np.float64)
assert_type(gausspulse(t="cutoff", retquad=False, retenv=True), np.float64)
assert_type(gausspulse(t="cutoff", retquad=True, retenv=True), np.float64)

assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, retquad=False, retenv=False), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, retquad=True, retenv=False), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, retquad=False, retenv=True), np.float64)
assert_type(gausspulse("cutoff", 1.0, 1.0, 1.0, 1.0, retquad=True, retenv=True), np.float64)
