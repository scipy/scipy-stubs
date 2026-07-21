# type-tests for `linalg/_decomp_ldl.pyi`

from typing import assert_type

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import ldl

###
# Input arrays

py_int_2d: list[list[int]]
py_float_2d: list[list[float]]
py_complex_2d: list[list[op.JustComplex]]

i64_2d: onp.Array2D[np.int64]
f32_2d: onp.Array2D[np.float32]
f32_3d: onp.Array3D[np.float32]
f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]

c64_2d: onp.Array2D[np.complex64]
c64_3d: onp.Array3D[np.complex64]
c128_2d: onp.Array2D[np.complex128]
c128_3d: onp.Array3D[np.complex128]

inexact_3d: onp.Array3D[npc.inexact]

###
# ldl

# -> float32
assert_type(ldl(f32_2d), tuple[onp.Array2D[np.float32], onp.Array2D[np.float32], onp.Array1D[np.intp]])
assert_type(ldl(f32_3d), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32], onp.ArrayND[np.intp]])

# -> float64
assert_type(ldl(py_int_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array1D[np.intp]])
assert_type(ldl(py_float_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array1D[np.intp]])
assert_type(ldl(i64_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array1D[np.intp]])
assert_type(ldl(f64_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array1D[np.intp]])
assert_type(ldl(f64_3d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.intp]])

# -> complex64
assert_type(ldl(c64_2d), tuple[onp.Array2D[np.complex64], onp.Array2D[np.complex64], onp.Array1D[np.intp]])
assert_type(ldl(c64_3d), tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.complex64], onp.ArrayND[np.intp]])

# -> complex128
assert_type(ldl(py_complex_2d), tuple[onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array1D[np.intp]])
assert_type(ldl(c128_2d), tuple[onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array1D[np.intp]])
assert_type(ldl(c128_3d), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128], onp.ArrayND[np.intp]])

# -> f32 | f64 | c64 | c128
assert_type(
    ldl(inexact_3d),
    tuple[
        onp.ArrayND[np.float32 | np.float64 | np.complex64 | np.complex128],
        onp.ArrayND[np.float32 | np.float64 | np.complex64 | np.complex128],
        onp.ArrayND[np.intp],
    ],
)
