# type-tests for `interpolate/_ndbspline.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import NdBSpline

###

_i64_1d: onp.Array1D[np.int64]
_f64_1d: onp.Array1D[np.float64]
_f64_nd: onp.ArrayND[np.float64]
_c128_nd: onp.ArrayND[np.complex128]

_knots: tuple[onp.Array1D[np.float64], ...]

_ndbspl_f64: NdBSpline[np.float64]
_ndbspl_c128: NdBSpline[np.complex128]

assert_type(NdBSpline(_knots, _f64_nd, 3), NdBSpline[np.float64])
assert_type(_ndbspl_f64(_f64_nd), onp.ArrayND[np.float64])
assert_type(_ndbspl_f64.derivative(_i64_1d), NdBSpline[np.float64])
assert_type(_ndbspl_f64.k, tuple[np.int64, ...])
assert_type(_ndbspl_f64.t, tuple[onp.Array1D[np.float64], ...])
assert_type(_ndbspl_f64.c, onp.ArrayND[np.float64])

assert_type(NdBSpline(_knots, _c128_nd, 3), NdBSpline[np.complex128])
assert_type(_ndbspl_c128(_f64_nd), onp.ArrayND[np.complex128])
assert_type(_ndbspl_c128.derivative(_i64_1d), NdBSpline[np.complex128])
