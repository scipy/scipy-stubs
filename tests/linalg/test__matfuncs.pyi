# type-tests for `linalg/_matfuncs.pyi`

from collections.abc import Callable
from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import (
    coshm,
    cosm,
    expm,
    fractional_matrix_power,
    funm,
    khatri_rao,
    logm,
    signm,
    sinhm,
    sinm,
    sqrtm,
    tanhm,
    tanm,
)

###

py_f_2d: list[list[float]]
py_f_3d: list[list[list[float]]]
py_c_2d: list[list[complex]]
py_c_3d: list[list[list[complex]]]

b1_2d: onp.Array2D[np.bool_]

i32_2d: onp.Array2D[np.int32]
i32_nd: onp.ArrayND[np.int32]

f16_2d: onp.Array2D[np.float16]

f32_2d: onp.Array2D[np.float32]
f32_nd: onp.ArrayND[np.float32]

f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]
f64_nd: onp.ArrayND[np.float64]

f80_2d: onp.Array2D[np.float128]

c64_2d: onp.Array2D[np.complex64]
c64_nd: onp.ArrayND[np.complex64]

c128_2d: onp.Array2D[np.complex128]
c128_3d: onp.Array3D[np.complex128]
c128_nd: onp.ArrayND[np.complex128]

c160_nd: onp.ArrayND[np.complex256]

func_f64: Callable[[onp.Array[Any, np.float64]], onp.ToComplexND]
func_c128: Callable[[onp.Array[Any, np.complex128]], onp.ToComplexND]

###
# expm / cosm / sinm / tanm / coshm / sinhm / tanhm
# (same overload structure, except that the hyperbolic ones reject `bool` input)

assert_type(expm(py_f_2d), onp.Array2D[np.float64])
assert_type(expm(py_c_2d), onp.Array2D[np.complex128])
assert_type(expm(b1_2d), onp.Array2D[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(expm(f16_2d), onp.Array2D[np.float32])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(expm(i32_2d), onp.Array2D[np.float64])
assert_type(expm(f32_2d), onp.Array2D[np.float32])
assert_type(expm(f64_2d), onp.Array2D[np.float64])
assert_type(expm(c64_2d), onp.Array2D[np.complex64])
assert_type(expm(c128_2d), onp.Array2D[np.complex128])
assert_type(expm(py_f_3d), onp.Array3D[np.float64])
assert_type(expm(f64_3d), onp.Array3D[np.float64])
assert_type(expm(c128_3d), onp.Array3D[np.complex128])
assert_type(expm(i32_nd), onp.ArrayND[np.float64])
assert_type(expm(f32_nd), onp.ArrayND[np.float32])
assert_type(expm(f64_nd), onp.ArrayND[np.float64])
assert_type(expm(c64_nd), onp.ArrayND[np.complex64])
assert_type(expm(c128_nd), onp.ArrayND[np.complex128])

assert_type(cosm(py_f_2d), onp.Array2D[np.float64])
assert_type(cosm(py_c_2d), onp.Array2D[np.complex128])
assert_type(cosm(b1_2d), onp.Array2D[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(cosm(f16_2d), onp.Array2D[np.float32])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(cosm(i32_2d), onp.Array2D[np.float64])
assert_type(cosm(f32_2d), onp.Array2D[np.float32])
assert_type(cosm(f64_2d), onp.Array2D[np.float64])
assert_type(cosm(c64_2d), onp.Array2D[np.complex64])
assert_type(cosm(c128_2d), onp.Array2D[np.complex128])
assert_type(cosm(py_f_3d), onp.Array3D[np.float64])
assert_type(cosm(f64_3d), onp.Array3D[np.float64])
assert_type(cosm(c128_3d), onp.Array3D[np.complex128])
assert_type(cosm(i32_nd), onp.ArrayND[np.float64])
assert_type(cosm(f32_nd), onp.ArrayND[np.float32])
assert_type(cosm(f64_nd), onp.ArrayND[np.float64])
assert_type(cosm(c64_nd), onp.ArrayND[np.complex64])
assert_type(cosm(c128_nd), onp.ArrayND[np.complex128])

assert_type(sinm(py_f_2d), onp.Array2D[np.float64])
assert_type(sinm(py_c_2d), onp.Array2D[np.complex128])
assert_type(sinm(b1_2d), onp.Array2D[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(sinm(f16_2d), onp.Array2D[np.float32])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(sinm(i32_2d), onp.Array2D[np.float64])
assert_type(sinm(f32_2d), onp.Array2D[np.float32])
assert_type(sinm(f64_2d), onp.Array2D[np.float64])
assert_type(sinm(c64_2d), onp.Array2D[np.complex64])
assert_type(sinm(c128_2d), onp.Array2D[np.complex128])
assert_type(sinm(py_f_3d), onp.Array3D[np.float64])
assert_type(sinm(f64_3d), onp.Array3D[np.float64])
assert_type(sinm(c128_3d), onp.Array3D[np.complex128])
assert_type(sinm(i32_nd), onp.ArrayND[np.float64])
assert_type(sinm(f32_nd), onp.ArrayND[np.float32])
assert_type(sinm(f64_nd), onp.ArrayND[np.float64])
assert_type(sinm(c64_nd), onp.ArrayND[np.complex64])
assert_type(sinm(c128_nd), onp.ArrayND[np.complex128])

assert_type(tanm(py_f_2d), onp.Array2D[np.float64])
assert_type(tanm(py_c_2d), onp.Array2D[np.complex128])
assert_type(tanm(b1_2d), onp.Array2D[np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(tanm(f16_2d), onp.Array2D[np.float32])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(tanm(i32_2d), onp.Array2D[np.float64])
assert_type(tanm(f32_2d), onp.Array2D[np.float32])
assert_type(tanm(f64_2d), onp.Array2D[np.float64])
assert_type(tanm(c64_2d), onp.Array2D[np.complex64])
assert_type(tanm(c128_2d), onp.Array2D[np.complex128])
assert_type(tanm(py_f_3d), onp.Array3D[np.float64])
assert_type(tanm(f64_3d), onp.Array3D[np.float64])
assert_type(tanm(c128_3d), onp.Array3D[np.complex128])
assert_type(tanm(i32_nd), onp.ArrayND[np.float64])
assert_type(tanm(f32_nd), onp.ArrayND[np.float32])
assert_type(tanm(f64_nd), onp.ArrayND[np.float64])
assert_type(tanm(c64_nd), onp.ArrayND[np.complex64])
assert_type(tanm(c128_nd), onp.ArrayND[np.complex128])

assert_type(coshm(py_f_2d), onp.Array2D[np.float64])
assert_type(coshm(py_c_2d), onp.Array2D[np.complex128])
assert_type(coshm(f16_2d), onp.Array2D[np.float32])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(coshm(i32_2d), onp.Array2D[np.float64])
assert_type(coshm(f32_2d), onp.Array2D[np.float32])
assert_type(coshm(f64_2d), onp.Array2D[np.float64])
assert_type(coshm(c64_2d), onp.Array2D[np.complex64])
assert_type(coshm(c128_2d), onp.Array2D[np.complex128])
assert_type(coshm(py_f_3d), onp.Array3D[np.float64])
assert_type(coshm(f64_3d), onp.Array3D[np.float64])
assert_type(coshm(c128_3d), onp.Array3D[np.complex128])
assert_type(coshm(i32_nd), onp.ArrayND[np.float64])
assert_type(coshm(f32_nd), onp.ArrayND[np.float32])
assert_type(coshm(f64_nd), onp.ArrayND[np.float64])
assert_type(coshm(c64_nd), onp.ArrayND[np.complex64])
assert_type(coshm(c128_nd), onp.ArrayND[np.complex128])

assert_type(sinhm(py_f_2d), onp.Array2D[np.float64])
assert_type(sinhm(py_c_2d), onp.Array2D[np.complex128])
assert_type(sinhm(f16_2d), onp.Array2D[np.float32])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(sinhm(i32_2d), onp.Array2D[np.float64])
assert_type(sinhm(f32_2d), onp.Array2D[np.float32])
assert_type(sinhm(f64_2d), onp.Array2D[np.float64])
assert_type(sinhm(c64_2d), onp.Array2D[np.complex64])
assert_type(sinhm(c128_2d), onp.Array2D[np.complex128])
assert_type(sinhm(py_f_3d), onp.Array3D[np.float64])
assert_type(sinhm(f64_3d), onp.Array3D[np.float64])
assert_type(sinhm(c128_3d), onp.Array3D[np.complex128])
assert_type(sinhm(i32_nd), onp.ArrayND[np.float64])
assert_type(sinhm(f32_nd), onp.ArrayND[np.float32])
assert_type(sinhm(f64_nd), onp.ArrayND[np.float64])
assert_type(sinhm(c64_nd), onp.ArrayND[np.complex64])
assert_type(sinhm(c128_nd), onp.ArrayND[np.complex128])

assert_type(tanhm(py_f_2d), onp.Array2D[np.float64])
assert_type(tanhm(py_c_2d), onp.Array2D[np.complex128])
assert_type(tanhm(f16_2d), onp.Array2D[np.float32])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(tanhm(i32_2d), onp.Array2D[np.float64])
assert_type(tanhm(f32_2d), onp.Array2D[np.float32])
assert_type(tanhm(f64_2d), onp.Array2D[np.float64])
assert_type(tanhm(c64_2d), onp.Array2D[np.complex64])
assert_type(tanhm(c128_2d), onp.Array2D[np.complex128])
assert_type(tanhm(py_f_3d), onp.Array3D[np.float64])
assert_type(tanhm(f64_3d), onp.Array3D[np.float64])
assert_type(tanhm(c128_3d), onp.Array3D[np.complex128])
assert_type(tanhm(i32_nd), onp.ArrayND[np.float64])
assert_type(tanhm(f32_nd), onp.ArrayND[np.float32])
assert_type(tanhm(f64_nd), onp.ArrayND[np.float64])
assert_type(tanhm(c64_nd), onp.ArrayND[np.complex64])
assert_type(tanhm(c128_nd), onp.ArrayND[np.complex128])

###
# sqrtm

assert_type(sqrtm(py_f_2d), onp.Array2D[np.float64 | np.complex128])
assert_type(sqrtm(py_c_2d), onp.Array2D[np.complex128])
assert_type(sqrtm(b1_2d), onp.Array2D[np.float64 | np.complex128])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(sqrtm(i32_2d), onp.Array2D[np.float64 | np.complex128])
assert_type(sqrtm(f32_2d), onp.Array2D[np.float32 | np.complex64])
assert_type(sqrtm(f64_2d), onp.Array2D[np.float64 | np.complex128])
assert_type(sqrtm(c64_2d), onp.Array2D[np.complex64])
assert_type(sqrtm(c128_2d), onp.Array2D[np.complex128])

assert_type(sqrtm(py_f_3d), onp.Array3D[np.float64 | np.complex128])
assert_type(sqrtm(py_c_3d), onp.Array3D[np.complex128])
assert_type(sqrtm(f64_3d), onp.Array3D[np.float64 | np.complex128])
assert_type(sqrtm(c128_3d), onp.Array3D[np.complex128])

assert_type(sqrtm(i32_nd), onp.ArrayND[np.float64 | np.complex128])
assert_type(sqrtm(f32_nd), onp.ArrayND[np.float32 | np.complex64])
assert_type(sqrtm(f64_nd), onp.ArrayND[np.float64 | np.complex128])
assert_type(sqrtm(c64_nd), onp.ArrayND[np.complex64])
assert_type(sqrtm(c128_nd), onp.ArrayND[np.complex128])
assert_type(sqrtm(c160_nd), onp.ArrayND[np.complex128])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

###
# logm

assert_type(logm(py_f_2d), onp.Array2D[np.float64 | np.complex128])
assert_type(logm(py_c_2d), onp.Array2D[np.complex128])
assert_type(logm(b1_2d), onp.Array2D[np.float64 | np.complex128])
assert_type(logm(i32_2d), onp.Array2D[np.float64 | np.complex128])
assert_type(logm(f32_2d), onp.Array2D[np.float64 | np.complex128])
assert_type(logm(f64_2d), onp.Array2D[np.float64 | np.complex128])
assert_type(logm(c64_2d), onp.Array2D[np.complex128])
assert_type(logm(c128_2d), onp.Array2D[np.complex128])

assert_type(logm(py_f_3d), onp.Array3D[np.float64 | np.complex128])
assert_type(logm(py_c_3d), onp.Array3D[np.complex128])
assert_type(logm(f64_3d), onp.Array3D[np.float64 | np.complex128])
assert_type(logm(c128_3d), onp.Array3D[np.complex128])

assert_type(logm(i32_nd), onp.ArrayND[np.float64 | np.complex128])
assert_type(logm(f32_nd), onp.ArrayND[np.float64 | np.complex128])
assert_type(logm(f64_nd), onp.ArrayND[np.float64 | np.complex128])
assert_type(logm(c64_nd), onp.ArrayND[np.complex128])
assert_type(logm(c128_nd), onp.ArrayND[np.complex128])
assert_type(logm(c160_nd), onp.ArrayND[np.complex128])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

###
# signm

assert_type(signm(py_f_2d), onp.Array2D[np.float64])
assert_type(signm(py_c_2d), onp.Array2D[np.complex128])
assert_type(signm(b1_2d), onp.Array2D[np.float32])
assert_type(signm(i32_2d), onp.Array2D[np.float64])
assert_type(signm(f32_2d), onp.Array2D[np.float32])
assert_type(signm(f64_2d), onp.Array2D[np.float64])
assert_type(signm(c64_2d), onp.Array2D[np.complex64])
assert_type(signm(c128_2d), onp.Array2D[np.complex128])

assert_type(signm(py_f_3d), onp.Array3D[np.float64])
assert_type(signm(py_c_3d), onp.Array3D[np.complex128])
assert_type(signm(f64_3d), onp.Array3D[np.float64])
assert_type(signm(c128_3d), onp.Array3D[np.complex128])

assert_type(signm(i32_nd), onp.ArrayND[np.float64])
assert_type(signm(f32_nd), onp.ArrayND[np.float32])
assert_type(signm(f64_nd), onp.ArrayND[np.float64])
assert_type(signm(c64_nd), onp.ArrayND[np.complex64])
assert_type(signm(c128_nd), onp.ArrayND[np.complex128])
assert_type(signm(c160_nd), onp.ArrayND[np.complex128])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

###
# funm

assert_type(funm(f64_nd, func_f64), onp.ArrayND[np.float64])
assert_type(funm(f64_nd, func_f64, False), tuple[onp.ArrayND[np.float64], float])
assert_type(funm(c128_nd, func_c128), onp.ArrayND[np.complex128])
assert_type(funm(c128_nd, func_c128, False), tuple[onp.ArrayND[np.complex128], float])

###
# khatri_rao

assert_type(khatri_rao(i32_nd, i32_nd), onp.ArrayND[npc.integer])
assert_type(khatri_rao(f64_nd, f64_nd), onp.ArrayND[np.float64])
assert_type(khatri_rao(c128_nd, c128_nd), onp.ArrayND[np.complex128])
assert_type(khatri_rao(c64_nd, c64_nd), onp.ArrayND[npc.complexfloating])

###
# fractional_matrix_power

assert_type(fractional_matrix_power(i32_2d, 2), onp.Array2D[np.int32])
assert_type(fractional_matrix_power(f32_2d, 2), onp.Array2D[np.float32])
assert_type(fractional_matrix_power(f64_2d, 2), onp.Array2D[np.float64])
assert_type(fractional_matrix_power(c64_2d, 2), onp.Array2D[np.complex64])
assert_type(fractional_matrix_power(c128_2d, 2), onp.Array2D[np.complex128])
assert_type(fractional_matrix_power(b1_2d, 2), onp.Array2D[np.bool_])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(fractional_matrix_power(f16_2d, 2), onp.Array2D[np.float16])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(fractional_matrix_power(f80_2d, 2), onp.Array2D[np.float128])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

assert_type(fractional_matrix_power(f32_2d, -1), onp.Array2D[np.float32])
assert_type(fractional_matrix_power(c128_2d, -1), onp.Array2D[np.complex128])
assert_type(fractional_matrix_power(i32_2d, -1), onp.Array2D[np.int32 | np.float64])
assert_type(fractional_matrix_power(b1_2d, -1), onp.Array2D[np.bool_ | np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

assert_type(fractional_matrix_power(i32_2d, 0.5), onp.Array2D[np.float64 | np.complex128])
assert_type(fractional_matrix_power(f32_2d, 0.5), onp.Array2D[np.float64 | np.complex128])
assert_type(fractional_matrix_power(f64_2d, 0.5), onp.Array2D[np.float64 | np.complex128])
assert_type(fractional_matrix_power(c64_2d, 0.5), onp.Array2D[np.complex128])
assert_type(fractional_matrix_power(c128_2d, 0.5), onp.Array2D[np.complex128])
assert_type(fractional_matrix_power(b1_2d, 0.5), onp.Array2D[np.float64 | np.complex128])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(fractional_matrix_power(f16_2d, 0.5), onp.Array2D[np.float64 | np.complex128])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(fractional_matrix_power(f80_2d, 0.5), onp.Array2D[np.longdouble | np.clongdouble])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

assert_type(fractional_matrix_power(py_f_2d, 2), onp.Array2D[np.float64])
assert_type(fractional_matrix_power(py_f_2d, 0.5), onp.Array2D[np.float64 | np.complex128])
assert_type(fractional_matrix_power(py_c_2d, 0.5), onp.Array2D[np.complex128])
assert_type(fractional_matrix_power(py_f_3d, 2), onp.Array3D[np.float64])
assert_type(fractional_matrix_power(py_f_3d, 0.5), onp.Array3D[np.float64 | np.complex128])
assert_type(fractional_matrix_power(py_c_3d, 2), onp.Array3D[np.complex128])

assert_type(fractional_matrix_power(i32_nd, 2), onp.ArrayND[np.int32])
assert_type(fractional_matrix_power(f32_nd, 2), onp.ArrayND[np.float32])
assert_type(fractional_matrix_power(f64_nd, 2), onp.ArrayND[np.float64])
assert_type(fractional_matrix_power(f64_nd, 0.5), onp.ArrayND[np.float64 | np.complex128])
assert_type(fractional_matrix_power(c128_nd, np.float64(1.0)), onp.ArrayND[np.complex128])
assert_type(fractional_matrix_power(c160_nd, 2), onp.ArrayND[np.complex256])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
