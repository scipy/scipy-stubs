# type-tests for `signal/_spline_filters.pyi`

from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.signal import cspline1d, cspline1d_eval, cspline2d
from scipy.signal._spline_filters import _FloatQ

###

_F64_1D: TypeAlias = onp.Array1D[np.float64]
_F64_2D: TypeAlias = onp.Array2D[np.float64]
_FQ_1D: TypeAlias = onp.Array1D[_FloatQ]
_F80_1D: TypeAlias = onp.Array1D[npc.floating80]
_C128_1D: TypeAlias = onp.Array1D[np.complex128]
_C128_2D: TypeAlias = onp.Array2D[np.complex128]
_C160_1D: TypeAlias = onp.Array1D[npc.complexfloating160]

###

_i64_1d: onp.Array1D[np.int64]
_f64_1d: _F64_1D
_f80_1d: _F80_1D
_c128_1d: _C128_1D
_c160_1d: _C160_1D

_f64_2d: _F64_2D
_c128_2d: _C128_2D

###

# cspline1d

assert_type(cspline1d(_i64_1d), _FQ_1D)
assert_type(cspline1d(_f64_1d), _F64_1D)
assert_type(cspline1d(_f80_1d), _F80_1D)
assert_type(cspline1d(_c128_1d), _C128_1D)
assert_type(cspline1d(_c160_1d), _C160_1D)

# cspline1d_eval

assert_type(cspline1d_eval(_f64_1d, _i64_1d), _F64_1D)
assert_type(cspline1d_eval(_f64_1d, _f64_1d), _F64_1D)
assert_type(cspline1d_eval(_f80_1d, _f64_1d), _F80_1D)
assert_type(cspline1d_eval(_c128_1d, _f64_1d), _C128_1D)
assert_type(cspline1d_eval(_c160_1d, _f64_1d), _C160_1D)

# cspline2d

assert_type(cspline2d(_i64_1d), _F64_1D)
assert_type(cspline2d(_f64_1d), _F64_1D)
assert_type(cspline2d(_f64_2d), _F64_1D)
assert_type(cspline2d(_c128_1d), _C128_1D)
assert_type(cspline2d(_c128_2d), _C128_1D)
