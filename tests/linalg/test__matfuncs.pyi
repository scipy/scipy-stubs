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

i32_nd: onp.ArrayND[np.int32]
f32_nd: onp.ArrayND[np.float32]
f64_nd: onp.ArrayND[np.float64]
c64_nd: onp.ArrayND[np.complex64]
c128_nd: onp.ArrayND[np.complex128]

func_f64: Callable[[onp.Array[Any, np.float64]], onp.ToComplexND]
func_c128: Callable[[onp.Array[Any, np.complex128]], onp.ToComplexND]

###
# expm / cosm / sinm / tanm / coshm / sinhm / tanhm (all share the same overload structure)

assert_type(expm(py_f_2d), onp.ArrayND[np.float64])
assert_type(expm(f32_nd), onp.ArrayND[npc.floating])
assert_type(expm(c128_nd), onp.ArrayND[np.complex128])
assert_type(expm(c64_nd), onp.ArrayND[npc.complexfloating])

assert_type(cosm(py_f_2d), onp.ArrayND[np.float64])
assert_type(cosm(f32_nd), onp.ArrayND[npc.floating])
assert_type(cosm(c128_nd), onp.ArrayND[np.complex128])
assert_type(cosm(c64_nd), onp.ArrayND[npc.complexfloating])

assert_type(sinm(py_f_2d), onp.ArrayND[np.float64])
assert_type(sinm(f32_nd), onp.ArrayND[npc.floating])
assert_type(sinm(c128_nd), onp.ArrayND[np.complex128])
assert_type(sinm(c64_nd), onp.ArrayND[npc.complexfloating])

assert_type(tanm(py_f_2d), onp.ArrayND[np.float64])
assert_type(tanm(f32_nd), onp.ArrayND[npc.floating])
assert_type(tanm(c128_nd), onp.ArrayND[np.complex128])
assert_type(tanm(c64_nd), onp.ArrayND[npc.complexfloating])

assert_type(coshm(py_f_2d), onp.ArrayND[np.float64])
assert_type(coshm(f32_nd), onp.ArrayND[npc.floating])
assert_type(coshm(c128_nd), onp.ArrayND[np.complex128])
assert_type(coshm(c64_nd), onp.ArrayND[npc.complexfloating])

assert_type(sinhm(py_f_2d), onp.ArrayND[np.float64])
assert_type(sinhm(f32_nd), onp.ArrayND[npc.floating])
assert_type(sinhm(c128_nd), onp.ArrayND[np.complex128])
assert_type(sinhm(c64_nd), onp.ArrayND[npc.complexfloating])

assert_type(tanhm(py_f_2d), onp.ArrayND[np.float64])
assert_type(tanhm(f32_nd), onp.ArrayND[npc.floating])
assert_type(tanhm(c128_nd), onp.ArrayND[np.complex128])
assert_type(tanhm(c64_nd), onp.ArrayND[npc.complexfloating])

###
# sqrtm

assert_type(sqrtm(py_f_2d), onp.ArrayND[np.float64])
assert_type(sqrtm(f32_nd), onp.ArrayND[npc.floating])
assert_type(sqrtm(c128_nd), onp.ArrayND[np.complex128])
assert_type(sqrtm(c64_nd), onp.ArrayND[npc.complexfloating])

###
# logm

assert_type(logm(py_f_2d), onp.ArrayND[np.float64] | onp.ArrayND[np.complex128])
assert_type(logm(f32_nd), onp.ArrayND[npc.inexact])
assert_type(logm(c128_nd), onp.ArrayND[np.complex128])
assert_type(logm(c64_nd), onp.ArrayND[npc.complexfloating])

###
# signm

assert_type(signm(py_f_2d), onp.ArrayND[np.float64])
assert_type(signm(f32_nd), onp.ArrayND[npc.floating])
assert_type(signm(c128_nd), onp.ArrayND[np.complex128])
assert_type(signm(c64_nd), onp.ArrayND[npc.complexfloating])

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

assert_type(fractional_matrix_power(i32_nd, 2), onp.ArrayND[npc.integer])
assert_type(fractional_matrix_power(f64_nd, 2), onp.ArrayND[np.float64])
assert_type(fractional_matrix_power(f32_nd, 2), onp.ArrayND[npc.floating])
assert_type(fractional_matrix_power(c128_nd, np.float64(1.0)), onp.ArrayND[np.complex128])
