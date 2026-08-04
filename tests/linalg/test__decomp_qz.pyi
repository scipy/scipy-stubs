# type-tests for `linalg/_decomp_qz.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import ordqz, qz

###

type _Tuple4[T] = tuple[T, T, T, T]

###

bool_2d: onp.Array2D[np.bool]
i8_2d: onp.Array2D[np.int8]
i32_2d: onp.Array2D[np.int32]
f16_2d: onp.Array2D[np.float16]
f32_2d: onp.Array2D[np.float32]
f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]
f80_2d: onp.Array2D[np.float128]
c64_2d: onp.Array2D[np.complex64]
c128_2d: onp.Array2D[np.complex128]
c160_2d: onp.Array2D[np.complex256]
py_f_2d: list[list[float]]
py_c_2d: list[list[complex]]

###
# qz

assert_type(qz(i8_2d, i8_2d), _Tuple4[onp.ArrayND[np.float32]])
assert_type(qz(f32_2d, f32_2d), _Tuple4[onp.ArrayND[np.float32]])
assert_type(qz(i32_2d, f32_2d), _Tuple4[onp.ArrayND[np.float64]])
assert_type(qz(f32_2d, f64_2d), _Tuple4[onp.ArrayND[np.float64]])
assert_type(qz(f64_2d, f32_2d), _Tuple4[onp.ArrayND[np.float64]])
assert_type(qz(f64_2d, f64_2d), _Tuple4[onp.ArrayND[np.float64]])
assert_type(qz(f64_3d, f64_3d), _Tuple4[onp.ArrayND[np.float64]])
assert_type(qz(py_f_2d, py_f_2d), _Tuple4[onp.ArrayND[np.float64]])
assert_type(qz(c64_2d, c64_2d), _Tuple4[onp.ArrayND[np.complex64]])
assert_type(qz(c64_2d, f32_2d), _Tuple4[onp.ArrayND[np.complex64]])
assert_type(qz(f32_2d, c64_2d), _Tuple4[onp.ArrayND[np.complex64]])
assert_type(qz(c64_2d, f64_2d), _Tuple4[onp.ArrayND[np.complex128]])
assert_type(qz(f64_2d, c64_2d), _Tuple4[onp.ArrayND[np.complex128]])
assert_type(qz(c128_2d, f32_2d), _Tuple4[onp.ArrayND[np.complex128]])
assert_type(qz(f32_2d, c128_2d), _Tuple4[onp.ArrayND[np.complex128]])
assert_type(qz(c128_2d, c128_2d), _Tuple4[onp.ArrayND[np.complex128]])
assert_type(qz(py_c_2d, py_c_2d), _Tuple4[onp.ArrayND[np.complex128]])

assert_type(qz(i8_2d, i8_2d, "complex"), _Tuple4[onp.ArrayND[np.complex64]])
assert_type(qz(f32_2d, f32_2d, "complex"), _Tuple4[onp.ArrayND[np.complex64]])
assert_type(qz(c64_2d, f32_2d, "complex"), _Tuple4[onp.ArrayND[np.complex64]])
assert_type(qz(f32_2d, f64_2d, "complex"), _Tuple4[onp.ArrayND[np.complex128]])
assert_type(qz(f64_2d, f64_2d, output="complex"), _Tuple4[onp.ArrayND[np.complex128]])
assert_type(qz(c128_2d, c64_2d, "complex"), _Tuple4[onp.ArrayND[np.complex128]])
assert_type(qz(py_c_2d, py_c_2d, "complex"), _Tuple4[onp.ArrayND[np.complex128]])

assert_type(qz(bool_2d, f64_2d), _Tuple4[onp.ArrayND[np.float64 | Any]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(qz(f16_2d, f16_2d), _Tuple4[onp.ArrayND[np.float64 | Any]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(qz(f64_2d, f80_2d), _Tuple4[onp.ArrayND[np.float64 | Any]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(qz(c160_2d, c160_2d), _Tuple4[onp.ArrayND[np.float64 | Any]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(qz(f80_2d, f80_2d, "complex"), _Tuple4[onp.ArrayND[np.float64 | Any]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

###
# ordqz

assert_type(
    ordqz(f64_2d, f64_2d),
    tuple[
        onp.ArrayND[npc.floating],
        onp.ArrayND[npc.floating],
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.floating],
        onp.ArrayND[npc.floating],
        onp.ArrayND[npc.floating],
    ],
)
assert_type(
    ordqz(c128_2d, c128_2d),
    tuple[
        onp.ArrayND[npc.inexact],
        onp.ArrayND[npc.inexact],
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.inexact],
        onp.ArrayND[npc.inexact],
        onp.ArrayND[npc.inexact],
    ],
)
assert_type(
    ordqz(c128_2d, c128_2d, "lhp", "complex"),
    tuple[
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.complexfloating],
    ],
)
assert_type(
    ordqz(c128_2d, c128_2d, output="complex"),
    tuple[
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.complexfloating],
    ],
)
