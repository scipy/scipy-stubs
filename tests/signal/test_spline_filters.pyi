# type-tests for `signal/_spline_filters.pyi`

from typing import Any, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.signal import (
    cspline1d,
    cspline1d_eval,
    cspline2d,
    gauss_spline,
    spline_filter,
)
from scipy.signal._spline_filters import _ComplexQ, _FloatQ

###

_F64_1D: TypeAlias = onp.Array1D[np.float64]
_F64_2D: TypeAlias = onp.Array2D[np.float64]
_F32_2D: TypeAlias = onp.Array2D[np.float32]
_FQ_1D: TypeAlias = onp.Array1D[_FloatQ]
_FQ_2D: TypeAlias = onp.Array2D[_FloatQ]
_FQ_3D: TypeAlias = onp.Array3D[_FloatQ]
_FQ_ND: TypeAlias = onp.ArrayND[_FloatQ, tuple[int] | tuple[Any, ...]]
_F80_1D: TypeAlias = onp.Array1D[npc.floating80]
_C128_1D: TypeAlias = onp.Array1D[np.complex128]
_C128_2D: TypeAlias = onp.Array2D[np.complex128]
_CQ_1D: TypeAlias = onp.Array1D[_ComplexQ]
_CQ_2D: TypeAlias = onp.Array2D[_ComplexQ]
_CQ_3D: TypeAlias = onp.Array3D[_ComplexQ]
_CQ_ND: TypeAlias = onp.ArrayND[_ComplexQ]
_C160_1D: TypeAlias = onp.Array1D[npc.complexfloating160]

###

_i64_1d: onp.Array1D[np.int64]
_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f64_1d: _F64_1D
_f80_1d: _F80_1D
_c128_1d: _C128_1D
_c160_1d: _C160_1D

_f64_2d: _F64_2D
_c128_2d: _C128_2D
_n: onp.ToFloat
_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_f_3d: list[list[list[float]]]
_py_c_1d: list[complex]
_py_c_2d: list[list[complex]]
_py_c_3d: list[list[list[complex]]]
_f_nd_like: onp.ToFloatND
_c_nd_like: onp.ToJustComplexND

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

# gauss_spline

assert_type(gauss_spline(_i64_1d, _n), _F64_1D)
assert_type(gauss_spline(_f64_1d, _n), _F64_1D)
assert_type(gauss_spline(_f80_1d, _n), _F80_1D)
assert_type(gauss_spline(_f64_2d, _n), _F64_2D)
assert_type(gauss_spline(_c128_1d, _n), _C128_1D)
assert_type(gauss_spline(_c128_2d, _n), _C128_2D)
assert_type(gauss_spline(_c160_1d, _n), _C160_1D)
assert_type(gauss_spline(_py_f_1d, _n), _FQ_1D)
assert_type(gauss_spline(_py_f_2d, _n), _FQ_2D)
assert_type(gauss_spline(_py_f_3d, _n), _FQ_3D)
assert_type(gauss_spline(_f_nd_like, _n), _FQ_ND)
assert_type(gauss_spline(_py_c_1d, _n), _CQ_1D)
assert_type(gauss_spline(_py_c_2d, _n), _CQ_2D)
assert_type(gauss_spline(_py_c_3d, _n), _CQ_3D)
assert_type(gauss_spline(_c_nd_like, _n), _CQ_ND)

# spline_filter

assert_type(spline_filter(_f32_1d, _n), _F32_2D)
assert_type(spline_filter(_f32_2d, _n), _F32_2D)
assert_type(spline_filter(_f64_1d, _n), _F64_2D)
assert_type(spline_filter(_f64_2d, _n), _F64_2D)
