# type-tests for `sem` from `stats/_stats_py.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.stats import sem

py_i_1d: list[int]
py_i_2d: list[list[int]]

py_f_1d: list[float]
py_f_2d: list[list[float]]

bool_1d: onp.Array1D[np.bool_]
bool_2d: onp.Array2D[np.bool_]

i16_1d: onp.Array1D[np.int16]
i16_2d: onp.Array2D[np.int16]

f32_1d: onp.Array1D[np.float32]
f32_2d: onp.Array2D[np.float32]

f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]

c64_1d: onp.Array1D[np.complex64]
c64_2d: onp.Array2D[np.complex64]

c128_1d: onp.Array1D[np.complex128]
c128_2d: onp.Array2D[np.complex128]

###

assert_type(sem(py_i_1d), np.float64)
assert_type(sem(py_f_1d), np.float64)
assert_type(sem(bool_1d), np.float64)
assert_type(sem(i16_1d), np.float64)
assert_type(sem(f32_1d), npc.floating)
assert_type(sem(f64_1d), np.float64)
assert_type(sem(c64_1d), npc.floating)
assert_type(sem(c128_1d), np.float64)

assert_type(sem(py_i_2d), onp.ArrayND[np.float64])
assert_type(sem(py_f_2d), onp.ArrayND[np.float64])
assert_type(sem(bool_2d), onp.ArrayND[np.float64])
assert_type(sem(i16_2d), onp.ArrayND[np.float64])
assert_type(sem(f32_2d), onp.ArrayND[npc.floating])
assert_type(sem(f64_2d), onp.ArrayND[np.float64])
assert_type(sem(c64_2d), onp.ArrayND[npc.floating])
assert_type(sem(c128_2d), onp.ArrayND[np.float64])

assert_type(sem(py_i_1d, axis=None), np.float64)
assert_type(sem(py_f_1d, axis=None), np.float64)
assert_type(sem(bool_1d, axis=None), np.float64)
assert_type(sem(i16_1d, axis=None), np.float64)
assert_type(sem(f32_1d, axis=None), npc.floating)
assert_type(sem(f64_1d, axis=None), np.float64)
assert_type(sem(c64_1d, axis=None), npc.floating)
assert_type(sem(c128_1d, axis=None), np.float64)

assert_type(sem(py_i_2d, axis=None), np.float64)
assert_type(sem(py_f_2d, axis=None), np.float64)
assert_type(sem(bool_2d, axis=None), np.float64)
assert_type(sem(i16_2d, axis=None), np.float64)
assert_type(sem(f32_2d, axis=None), npc.floating)
assert_type(sem(f64_2d, axis=None), np.float64)
assert_type(sem(c64_2d, axis=None), npc.floating)
assert_type(sem(c128_2d, axis=None), np.float64)

assert_type(sem(py_i_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(sem(py_f_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(sem(bool_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(sem(i16_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(sem(f32_1d, keepdims=True), onp.ArrayND[npc.floating])
assert_type(sem(f64_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(sem(c64_1d, keepdims=True), onp.ArrayND[npc.floating])
assert_type(sem(c128_1d, keepdims=True), onp.ArrayND[np.float64])

assert_type(sem(py_i_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(sem(py_f_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(sem(bool_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(sem(i16_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(sem(f32_2d, keepdims=True), onp.ArrayND[npc.floating])
assert_type(sem(f64_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(sem(c64_2d, keepdims=True), onp.ArrayND[npc.floating])
assert_type(sem(c128_2d, keepdims=True), onp.ArrayND[np.float64])
