# type-tests for `linalg/_basic.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import inv, solve, solve_banded, solve_circulant, solve_toeplitz, solve_triangular

b1_nd: onp.ArrayND[np.bool_]

i8_1d: onp.Array1D[np.int8]
i8_2d: onp.Array2D[np.int8]
i8_3d: onp.Array3D[np.int8]
i8_nd: onp.ArrayND[np.int8]

i32_1d: onp.Array1D[np.int32]
i32_2d: onp.Array2D[np.int32]
i32_3d: onp.Array3D[np.int32]
i32_nd: onp.ArrayND[np.int32]

f16_1d: onp.Array1D[np.float16]
f16_2d: onp.Array2D[np.float16]
f16_3d: onp.Array3D[np.float16]
f16_nd: onp.ArrayND[np.float16]

f32_1d: onp.Array1D[np.float32]
f32_2d: onp.Array2D[np.float32]
f32_3d: onp.Array3D[np.float32]
f32_nd: onp.ArrayND[np.float32]

f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]
f64_nd: onp.ArrayND[np.float64]

f80_1d: onp.Array1D[np.longdouble]
f80_2d: onp.Array2D[np.longdouble]
f80_3d: onp.Array3D[np.longdouble]
f80_nd: onp.ArrayND[np.longdouble]

c64_1d: onp.Array1D[np.complex64]
c64_2d: onp.Array2D[np.complex64]
c64_3d: onp.Array3D[np.complex64]
c64_nd: onp.ArrayND[np.complex64]

c128_1d: onp.Array1D[np.complex128]
c128_2d: onp.Array2D[np.complex128]
c128_3d: onp.Array3D[np.complex128]
c128_nd: onp.ArrayND[np.complex128]

c160_1d: onp.Array1D[np.clongdouble]
c160_2d: onp.Array2D[np.clongdouble]
c160_3d: onp.Array3D[np.clongdouble]
c160_nd: onp.ArrayND[np.clongdouble]

py_b_2d: list[list[bool]]
py_b_3d: list[list[list[bool]]]

py_i_2d: list[list[int]]
py_i_3d: list[list[list[int]]]

py_f_1d: list[float]
py_f_2d: list[list[float]]
py_f_3d: list[list[list[float]]]

py_c_1d: list[complex]
py_c_2d: list[list[complex]]
py_c_3d: list[list[list[complex]]]

###
# solve

assert_type(solve(i8_2d, i8_1d), onp.Array2D[np.float64])
assert_type(solve(i8_2d, i8_2d), onp.Array2D[np.float64])
assert_type(solve(i8_2d, i8_3d), onp.ArrayND[np.float64])
assert_type(solve(i8_3d, i8_1d), onp.ArrayND[np.float64])

assert_type(solve(f16_2d, f16_1d), onp.Array2D[np.float32 | np.float64])
assert_type(solve(f16_2d, f16_2d), onp.Array2D[np.float32 | np.float64])
assert_type(solve(f16_2d, f16_3d), onp.ArrayND[np.float32 | np.float64])
assert_type(solve(f16_3d, f16_1d), onp.ArrayND[np.float32 | np.float64])

assert_type(solve(f32_2d, f32_1d), onp.Array2D[np.float32 | np.float64])
assert_type(solve(f32_2d, f32_2d), onp.Array2D[np.float32 | np.float64])
assert_type(solve(f32_2d, f32_3d), onp.ArrayND[np.float32 | np.float64])
assert_type(solve(f32_3d, f32_1d), onp.ArrayND[np.float32 | np.float64])

assert_type(solve(f64_2d, f64_1d), onp.Array2D[np.float64])
assert_type(solve(f64_2d, f64_2d), onp.Array2D[np.float64])
assert_type(solve(f64_2d, f64_3d), onp.ArrayND[np.float64])
assert_type(solve(f64_3d, f64_1d), onp.ArrayND[np.float64])

assert_type(solve(f80_2d, f80_1d), onp.Array2D[np.float64])
assert_type(solve(f80_2d, f80_2d), onp.Array2D[np.float64])
assert_type(solve(f80_2d, f80_3d), onp.ArrayND[np.float64])
assert_type(solve(f80_3d, f80_1d), onp.ArrayND[np.float64])

assert_type(solve(c64_2d, c64_1d), onp.Array2D[np.complex64 | np.complex128])
assert_type(solve(c64_2d, c64_2d), onp.Array2D[np.complex64 | np.complex128])
assert_type(solve(c64_2d, c64_3d), onp.ArrayND[np.complex64 | np.complex128])
assert_type(solve(c64_3d, c64_1d), onp.ArrayND[np.complex64 | np.complex128])

assert_type(solve(c128_2d, c128_1d), onp.Array2D[np.complex128])
assert_type(solve(c128_2d, c128_2d), onp.Array2D[np.complex128])
assert_type(solve(c128_2d, c128_3d), onp.ArrayND[np.complex128])
assert_type(solve(c128_3d, c128_1d), onp.ArrayND[np.complex128])

assert_type(solve(c160_2d, c160_1d), onp.Array2D[np.complex128])
assert_type(solve(c160_2d, c160_2d), onp.Array2D[np.complex128])
assert_type(solve(c160_2d, c160_3d), onp.ArrayND[np.complex128])
assert_type(solve(c160_3d, c160_1d), onp.ArrayND[np.complex128])

assert_type(solve(py_f_2d, py_f_1d), onp.Array2D[np.float64])
assert_type(solve(py_f_2d, py_f_2d), onp.Array2D[np.float64])
assert_type(solve(py_f_2d, py_f_3d), onp.ArrayND[np.float64])
assert_type(solve(py_f_3d, py_f_1d), onp.ArrayND[np.float64])

assert_type(solve(py_c_2d, py_c_1d), onp.Array2D[np.complex128])
assert_type(solve(py_c_2d, py_c_2d), onp.Array2D[np.complex128])
assert_type(solve(py_c_2d, py_c_3d), onp.ArrayND[np.complex128])
assert_type(solve(py_c_3d, py_c_1d), onp.ArrayND[np.complex128])

###
# solve_triangular

assert_type(solve_triangular(i8_2d, i8_1d), onp.Array1D[np.float64])
assert_type(solve_triangular(i8_2d, i8_2d), onp.Array2D[np.float64])
assert_type(solve_triangular(i8_2d, i8_3d), onp.ArrayND[np.float64])
assert_type(solve_triangular(i8_3d, i8_1d), onp.ArrayND[np.float64])

assert_type(solve_triangular(f16_2d, f16_1d), onp.Array1D[np.float32 | np.float64])
assert_type(solve_triangular(f16_2d, f16_2d), onp.Array2D[np.float32 | np.float64])
assert_type(solve_triangular(f16_2d, f16_3d), onp.ArrayND[np.float32 | np.float64])
assert_type(solve_triangular(f16_3d, f16_1d), onp.ArrayND[np.float32 | np.float64])

assert_type(solve_triangular(f32_2d, f32_1d), onp.Array1D[np.float32 | np.float64])
assert_type(solve_triangular(f32_2d, f32_2d), onp.Array2D[np.float32 | np.float64])
assert_type(solve_triangular(f32_2d, f32_3d), onp.ArrayND[np.float32 | np.float64])
assert_type(solve_triangular(f32_3d, f32_1d), onp.ArrayND[np.float32 | np.float64])

assert_type(solve_triangular(f64_2d, f64_1d), onp.Array1D[np.float64])
assert_type(solve_triangular(f64_2d, f64_2d), onp.Array2D[np.float64])
assert_type(solve_triangular(f64_2d, f64_3d), onp.ArrayND[np.float64])
assert_type(solve_triangular(f64_3d, f64_1d), onp.ArrayND[np.float64])

assert_type(solve_triangular(f80_2d, f80_1d), onp.Array1D[np.float64])
assert_type(solve_triangular(f80_2d, f80_2d), onp.Array2D[np.float64])
assert_type(solve_triangular(f80_2d, f80_3d), onp.ArrayND[np.float64])
assert_type(solve_triangular(f80_3d, f80_1d), onp.ArrayND[np.float64])

assert_type(solve_triangular(c64_2d, c64_1d), onp.Array1D[np.complex64 | np.complex128])
assert_type(solve_triangular(c64_2d, c64_2d), onp.Array2D[np.complex64 | np.complex128])
assert_type(solve_triangular(c64_2d, c64_3d), onp.ArrayND[np.complex64 | np.complex128])
assert_type(solve_triangular(c64_3d, c64_1d), onp.ArrayND[np.complex64 | np.complex128])

assert_type(solve_triangular(c128_2d, c128_1d), onp.Array1D[np.complex128])
assert_type(solve_triangular(c128_2d, c128_2d), onp.Array2D[np.complex128])
assert_type(solve_triangular(c128_2d, c128_3d), onp.ArrayND[np.complex128])
assert_type(solve_triangular(c128_3d, c128_1d), onp.ArrayND[np.complex128])

assert_type(solve_triangular(c160_2d, c160_1d), onp.Array1D[np.complex128])
assert_type(solve_triangular(c160_2d, c160_2d), onp.Array2D[np.complex128])
assert_type(solve_triangular(c160_2d, c160_3d), onp.ArrayND[np.complex128])
assert_type(solve_triangular(c160_3d, c160_1d), onp.ArrayND[np.complex128])

assert_type(solve_triangular(py_f_2d, py_f_1d), onp.Array1D[np.float64])
assert_type(solve_triangular(py_f_2d, py_f_2d), onp.Array2D[np.float64])
assert_type(solve_triangular(py_f_2d, py_f_3d), onp.ArrayND[np.float64])
assert_type(solve_triangular(py_f_3d, py_f_1d), onp.ArrayND[np.float64])

assert_type(solve_triangular(py_c_2d, py_c_1d), onp.Array1D[np.complex128])
assert_type(solve_triangular(py_c_2d, py_c_2d), onp.Array2D[np.complex128])
assert_type(solve_triangular(py_c_2d, py_c_3d), onp.ArrayND[np.complex128])
assert_type(solve_triangular(py_c_3d, py_c_1d), onp.ArrayND[np.complex128])

###
# solve_banded  (equivalent overload structure to `solveh_banded`)

assert_type(solve_banded((1, 2), i8_2d, i8_1d), onp.Array1D[np.float64])
assert_type(solve_banded((1, 2), i8_2d, i8_2d), onp.Array2D[np.float64])
assert_type(solve_banded((1, 2), i8_2d, i8_3d), onp.ArrayND[np.float64])
assert_type(solve_banded((1, 2), i8_3d, i8_1d), onp.ArrayND[np.float64])

assert_type(solve_banded((1, 2), f16_2d, f16_1d), onp.Array1D[np.float32 | np.float64])
assert_type(solve_banded((1, 2), f16_2d, f16_2d), onp.Array2D[np.float32 | np.float64])
assert_type(solve_banded((1, 2), f16_2d, f16_3d), onp.ArrayND[np.float32 | np.float64])
assert_type(solve_banded((1, 2), f16_3d, f16_1d), onp.ArrayND[np.float32 | np.float64])

assert_type(solve_banded((1, 2), f32_2d, f32_1d), onp.Array1D[np.float32 | np.float64])
assert_type(solve_banded((1, 2), f32_2d, f32_2d), onp.Array2D[np.float32 | np.float64])
assert_type(solve_banded((1, 2), f32_2d, f32_3d), onp.ArrayND[np.float32 | np.float64])
assert_type(solve_banded((1, 2), f32_3d, f32_1d), onp.ArrayND[np.float32 | np.float64])

assert_type(solve_banded((1, 2), f64_2d, f64_1d), onp.Array1D[np.float64])
assert_type(solve_banded((1, 2), f64_2d, f64_2d), onp.Array2D[np.float64])
assert_type(solve_banded((1, 2), f64_2d, f64_3d), onp.ArrayND[np.float64])
assert_type(solve_banded((1, 2), f64_3d, f64_1d), onp.ArrayND[np.float64])

assert_type(solve_banded((1, 2), f80_2d, f80_1d), onp.Array1D[np.float64])
assert_type(solve_banded((1, 2), f80_2d, f80_2d), onp.Array2D[np.float64])
assert_type(solve_banded((1, 2), f80_2d, f80_3d), onp.ArrayND[np.float64])
assert_type(solve_banded((1, 2), f80_3d, f80_1d), onp.ArrayND[np.float64])

assert_type(solve_banded((1, 2), c64_2d, c64_1d), onp.Array1D[np.complex64 | np.complex128])
assert_type(solve_banded((1, 2), c64_2d, c64_2d), onp.Array2D[np.complex64 | np.complex128])
assert_type(solve_banded((1, 2), c64_2d, c64_3d), onp.ArrayND[np.complex64 | np.complex128])
assert_type(solve_banded((1, 2), c64_3d, c64_1d), onp.ArrayND[np.complex64 | np.complex128])

assert_type(solve_banded((1, 2), c128_2d, c128_1d), onp.Array1D[np.complex128])
assert_type(solve_banded((1, 2), c128_2d, c128_2d), onp.Array2D[np.complex128])
assert_type(solve_banded((1, 2), c128_2d, c128_3d), onp.ArrayND[np.complex128])
assert_type(solve_banded((1, 2), c128_3d, c128_1d), onp.ArrayND[np.complex128])

assert_type(solve_banded((1, 2), c160_2d, c160_1d), onp.Array1D[np.complex128])
assert_type(solve_banded((1, 2), c160_2d, c160_2d), onp.Array2D[np.complex128])
assert_type(solve_banded((1, 2), c160_2d, c160_3d), onp.ArrayND[np.complex128])
assert_type(solve_banded((1, 2), c160_3d, c160_1d), onp.ArrayND[np.complex128])

assert_type(solve_banded((1, 2), py_f_2d, py_f_1d), onp.Array1D[np.float64])
assert_type(solve_banded((1, 2), py_f_2d, py_f_2d), onp.Array2D[np.float64])
assert_type(solve_banded((1, 2), py_f_2d, py_f_3d), onp.ArrayND[np.float64])
assert_type(solve_banded((1, 2), py_f_3d, py_f_1d), onp.ArrayND[np.float64])

assert_type(solve_banded((1, 2), py_c_2d, py_c_1d), onp.Array1D[np.complex128])
assert_type(solve_banded((1, 2), py_c_2d, py_c_2d), onp.Array2D[np.complex128])
assert_type(solve_banded((1, 2), py_c_2d, py_c_3d), onp.ArrayND[np.complex128])
assert_type(solve_banded((1, 2), py_c_3d, py_c_1d), onp.ArrayND[np.complex128])

###
# solve_toeplitz

assert_type(solve_toeplitz(i8_1d, i8_1d), onp.Array1D[np.float64])
assert_type(solve_toeplitz(i8_1d, i8_2d), onp.Array2D[np.float64])
assert_type(solve_toeplitz(i8_1d, i8_3d), onp.ArrayND[np.float64])
assert_type(solve_toeplitz(i8_2d, i8_1d), onp.ArrayND[np.float64])

assert_type(solve_toeplitz(f16_1d, f16_1d), onp.Array1D[np.float64])
assert_type(solve_toeplitz(f16_1d, f16_2d), onp.Array2D[np.float64])
assert_type(solve_toeplitz(f16_1d, f16_3d), onp.ArrayND[np.float64])
assert_type(solve_toeplitz(f16_2d, f16_1d), onp.ArrayND[np.float64])

assert_type(solve_toeplitz(f32_1d, f32_1d), onp.Array1D[np.float64])
assert_type(solve_toeplitz(f32_1d, f32_2d), onp.Array2D[np.float64])
assert_type(solve_toeplitz(f32_1d, f32_3d), onp.ArrayND[np.float64])
assert_type(solve_toeplitz(f32_2d, f32_1d), onp.ArrayND[np.float64])

assert_type(solve_toeplitz(f64_1d, f64_1d), onp.Array1D[np.float64])
assert_type(solve_toeplitz(f64_1d, f64_2d), onp.Array2D[np.float64])
assert_type(solve_toeplitz(f64_1d, f64_3d), onp.ArrayND[np.float64])
assert_type(solve_toeplitz(f64_2d, f64_1d), onp.ArrayND[np.float64])

assert_type(solve_toeplitz(f80_1d, f80_1d), onp.Array1D[np.float64])
assert_type(solve_toeplitz(f80_1d, f80_2d), onp.Array2D[np.float64])
assert_type(solve_toeplitz(f80_1d, f80_3d), onp.ArrayND[np.float64])
assert_type(solve_toeplitz(f80_2d, f80_1d), onp.ArrayND[np.float64])

assert_type(solve_toeplitz(c64_1d, c64_1d), onp.Array1D[np.complex128])
assert_type(solve_toeplitz(c64_1d, c64_2d), onp.Array2D[np.complex128])
assert_type(solve_toeplitz(c64_1d, c64_3d), onp.ArrayND[np.complex128])
assert_type(solve_toeplitz(c64_2d, c64_1d), onp.ArrayND[np.complex128])

assert_type(solve_toeplitz(c128_1d, c128_1d), onp.Array1D[np.complex128])
assert_type(solve_toeplitz(c128_1d, c128_2d), onp.Array2D[np.complex128])
assert_type(solve_toeplitz(c128_1d, c128_3d), onp.ArrayND[np.complex128])
assert_type(solve_toeplitz(c128_2d, c128_1d), onp.ArrayND[np.complex128])

assert_type(solve_toeplitz(c160_1d, c160_1d), onp.Array1D[np.complex128])
assert_type(solve_toeplitz(c160_1d, c160_2d), onp.Array2D[np.complex128])
assert_type(solve_toeplitz(c160_1d, c160_3d), onp.ArrayND[np.complex128])
assert_type(solve_toeplitz(c160_2d, c160_1d), onp.ArrayND[np.complex128])

assert_type(solve_toeplitz(py_f_1d, py_f_1d), onp.Array1D[np.float64])
assert_type(solve_toeplitz(py_f_1d, py_f_2d), onp.Array2D[np.float64])
assert_type(solve_toeplitz(py_f_1d, py_f_3d), onp.ArrayND[np.float64])
assert_type(solve_toeplitz(py_f_2d, py_f_1d), onp.ArrayND[np.float64])

assert_type(solve_toeplitz(py_c_1d, py_c_1d), onp.Array1D[np.complex128])
assert_type(solve_toeplitz(py_c_1d, py_c_2d), onp.Array2D[np.complex128])
assert_type(solve_toeplitz(py_c_1d, py_c_3d), onp.ArrayND[np.complex128])
assert_type(solve_toeplitz(py_c_2d, py_c_1d), onp.ArrayND[np.complex128])

###
# solve_circulant

assert_type(solve_circulant(i8_1d, i8_1d), onp.Array1D[np.float64])
assert_type(solve_circulant(i8_1d, i8_2d), onp.Array2D[np.float64])
assert_type(solve_circulant(i8_1d, i8_3d), onp.ArrayND[np.float64])
assert_type(solve_circulant(i8_2d, i8_1d), onp.ArrayND[np.float64])

assert_type(solve_circulant(f16_1d, f16_1d), onp.Array1D[npc.floating])
assert_type(solve_circulant(f16_1d, f16_2d), onp.Array2D[npc.floating])
assert_type(solve_circulant(f16_1d, f16_3d), onp.ArrayND[npc.floating])
assert_type(solve_circulant(f16_2d, f16_1d), onp.ArrayND[npc.floating])

assert_type(solve_circulant(f32_1d, f32_1d), onp.Array1D[npc.floating])
assert_type(solve_circulant(f32_1d, f32_2d), onp.Array2D[npc.floating])
assert_type(solve_circulant(f32_1d, f32_3d), onp.ArrayND[npc.floating])
assert_type(solve_circulant(f32_2d, f32_1d), onp.ArrayND[npc.floating])

assert_type(solve_circulant(f64_1d, f64_1d), onp.Array1D[np.float64])
assert_type(solve_circulant(f64_1d, f64_2d), onp.Array2D[np.float64])
assert_type(solve_circulant(f64_1d, f64_3d), onp.ArrayND[np.float64])
assert_type(solve_circulant(f64_2d, f64_1d), onp.ArrayND[np.float64])

assert_type(solve_circulant(f80_1d, f80_1d), onp.Array1D[npc.floating])
assert_type(solve_circulant(f80_1d, f80_2d), onp.Array2D[npc.floating])
assert_type(solve_circulant(f80_1d, f80_3d), onp.ArrayND[npc.floating])
assert_type(solve_circulant(f80_2d, f80_1d), onp.ArrayND[npc.floating])

assert_type(solve_circulant(c64_1d, c64_1d), onp.Array1D[npc.complexfloating])
assert_type(solve_circulant(c64_1d, c64_2d), onp.Array2D[npc.complexfloating])
assert_type(solve_circulant(c64_1d, c64_3d), onp.ArrayND[npc.complexfloating])
assert_type(solve_circulant(c64_2d, c64_1d), onp.ArrayND[npc.complexfloating])

assert_type(solve_circulant(c128_1d, c128_1d), onp.Array1D[np.complex128])
assert_type(solve_circulant(c128_1d, c128_2d), onp.Array2D[np.complex128])
assert_type(solve_circulant(c128_1d, c128_3d), onp.ArrayND[np.complex128])
assert_type(solve_circulant(c128_2d, c128_1d), onp.ArrayND[np.complex128])

assert_type(solve_circulant(c160_1d, c160_1d), onp.Array1D[npc.complexfloating])
assert_type(solve_circulant(c160_1d, c160_2d), onp.Array2D[npc.complexfloating])
assert_type(solve_circulant(c160_1d, c160_3d), onp.ArrayND[npc.complexfloating])
assert_type(solve_circulant(c160_2d, c160_1d), onp.ArrayND[npc.complexfloating])

assert_type(solve_circulant(py_f_1d, py_f_1d), onp.Array1D[np.float64])
assert_type(solve_circulant(py_f_1d, py_f_2d), onp.Array2D[np.float64])
assert_type(solve_circulant(py_f_1d, py_f_3d), onp.ArrayND[np.float64])
assert_type(solve_circulant(py_f_2d, py_f_1d), onp.ArrayND[np.float64])

assert_type(solve_circulant(py_c_1d, py_c_1d), onp.Array1D[np.complex128])
assert_type(solve_circulant(py_c_1d, py_c_2d), onp.Array2D[np.complex128])
assert_type(solve_circulant(py_c_1d, py_c_3d), onp.ArrayND[np.complex128])
assert_type(solve_circulant(py_c_2d, py_c_1d), onp.ArrayND[np.complex128])

###
# inv

assert_type(inv(f32_2d), onp.Array2D[np.float32])
assert_type(inv(f64_2d), onp.Array2D[np.float64])
assert_type(inv(c64_2d), onp.Array2D[np.complex64])
assert_type(inv(c128_2d), onp.Array2D[np.complex128])

assert_type(inv(py_b_2d), onp.Array2D[np.float32])
assert_type(inv(py_i_2d), onp.Array2D[np.float64])
assert_type(inv(py_f_2d), onp.Array2D[np.float64])
assert_type(inv(py_c_2d), onp.Array2D[np.complex128])

assert_type(inv(f32_3d), onp.Array3D[np.float32])
assert_type(inv(f64_3d), onp.Array3D[np.float64])
assert_type(inv(c64_3d), onp.Array3D[np.complex64])
assert_type(inv(c128_3d), onp.Array3D[np.complex128])

assert_type(inv(py_b_3d), onp.ArrayND[np.float32])
assert_type(inv(py_i_3d), onp.ArrayND[np.float64])
assert_type(inv(py_f_3d), onp.ArrayND[np.float64])
assert_type(inv(py_c_3d), onp.ArrayND[np.complex128])

assert_type(inv(b1_nd), onp.ArrayND[np.float32])
assert_type(inv(i8_nd), onp.ArrayND[np.float32])
assert_type(inv(f16_nd), onp.ArrayND[np.float32])
assert_type(inv(f32_nd), onp.ArrayND[np.float32])
assert_type(inv(i32_nd), onp.ArrayND[np.float64])
assert_type(inv(f64_nd), onp.ArrayND[np.float64])
assert_type(inv(f80_nd), onp.ArrayND[np.float64])
assert_type(inv(c64_nd), onp.ArrayND[np.complex64])
assert_type(inv(c128_nd), onp.ArrayND[np.complex128])
assert_type(inv(c160_nd), onp.ArrayND[np.complex128])

###
# TODO(jorenham): det

###
# TODO(jorenham): lstsq

###
# TODO(jorenham): pinv[h]

###
# TODO(jorenham): matrix_balance

###
# TODO(jorenham): matmul_toeplitz
