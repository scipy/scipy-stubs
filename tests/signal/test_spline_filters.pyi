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
    qspline1d,
    qspline1d_eval,
    qspline2d,
    spline_filter,
    symiirorder1,
    symiirorder2,
)

###

_F64_1D: TypeAlias = onp.Array1D[np.float64]
_F64_2D: TypeAlias = onp.Array2D[np.float64]
_F32_1D: TypeAlias = onp.Array1D[np.float32]
_F32_2D: TypeAlias = onp.Array2D[np.float32]
_FQ_1D: TypeAlias = onp.Array1D[np.float64 | np.longdouble]
_FQ_2D: TypeAlias = onp.Array2D[np.float64 | np.longdouble]
_FQ_3D: TypeAlias = onp.Array3D[np.float64 | np.longdouble]
_FQ_ND: TypeAlias = onp.ArrayND[np.float64 | np.longdouble, tuple[int] | tuple[Any, ...]]
_F80_1D: TypeAlias = onp.Array1D[npc.floating80]
_C64_1D: TypeAlias = onp.Array1D[np.complex64]
_C64_2D: TypeAlias = onp.Array2D[np.complex64]
_C128_1D: TypeAlias = onp.Array1D[np.complex128]
_C128_2D: TypeAlias = onp.Array2D[np.complex128]
_CQ_1D: TypeAlias = onp.Array1D[np.complex128 | np.clongdouble]
_CQ_2D: TypeAlias = onp.Array2D[np.complex128 | np.clongdouble]
_CQ_3D: TypeAlias = onp.Array3D[np.complex128 | np.clongdouble]
_CQ_ND: TypeAlias = onp.ArrayND[np.complex128 | np.clongdouble]
_C160_1D: TypeAlias = onp.Array1D[npc.complexfloating160]

###

_i64_1d: onp.Array1D[np.int64]
_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_c64_1d: _C64_1D
_c64_2d: _C64_2D
_f64_1d: _F64_1D
_f80_1d: _F80_1D
_c128_1d: _C128_1D
_c160_1d: _C160_1D

_f64_2d: _F64_2D
_c128_2d: _C128_2D
_n: onp.ToFloat
_c: onp.ToComplex
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

# qspline1d

assert_type(qspline1d(_i64_1d, _n), _FQ_1D)
assert_type(qspline1d(_f64_1d, _n), _F64_1D)
assert_type(qspline1d(_f80_1d, _n), _F80_1D)
assert_type(qspline1d(_c128_1d, _n), _C128_1D)
assert_type(qspline1d(_c160_1d, _n), _C160_1D)
assert_type(qspline1d(_py_f_1d, _n), _FQ_1D)
assert_type(qspline1d(_py_c_1d, _n), _CQ_1D)
assert_type(qspline1d(_f_nd_like, _n), _FQ_1D)
assert_type(qspline1d(_c_nd_like, _n), _CQ_1D)
assert_type(qspline1d(_f32_2d, _n), _FQ_1D)
assert_type(qspline1d(_c64_2d, _n), _CQ_1D)

# qspline1d_eval

assert_type(qspline1d_eval(_f64_1d, _i64_1d), _F64_1D)
assert_type(qspline1d_eval(_f64_1d, _f64_1d), _F64_1D)
assert_type(qspline1d_eval(_f80_1d, _f64_1d), _F80_1D)
assert_type(qspline1d_eval(_c128_1d, _f64_1d), _C128_1D)
assert_type(qspline1d_eval(_c160_1d, _f64_1d), _C160_1D)

# qspline2d

assert_type(qspline2d(_i64_1d, _n), _F64_1D)
assert_type(qspline2d(_f64_1d, _n), _F64_1D)
assert_type(qspline2d(_f64_2d, _n), _F64_1D)
assert_type(qspline2d(_c128_1d, _n), _C128_1D)
assert_type(qspline2d(_c128_2d, _n), _C128_1D)
assert_type(qspline2d(_py_f_2d, _n), _F64_1D)
assert_type(qspline2d(_py_c_2d, _n), _C128_1D)

# symiirorder1

assert_type(symiirorder1(_f32_1d, _c, _c, _n), _F32_1D)
assert_type(symiirorder1(_f64_1d, _c, _c, _n), _F64_1D)
assert_type(symiirorder1(_c64_2d, _c, _c, _n), _C64_2D)
assert_type(symiirorder1(_c128_1d, _c, _c, _n), _C128_1D)

# symiirorder2

assert_type(symiirorder2(_f32_2d, _n, _n, _n), _F32_2D)
assert_type(symiirorder2(_f64_1d, _n, _n, _n), _F64_1D)
assert_type(symiirorder2(_f64_2d, _n, _n, _n), _F64_2D)
