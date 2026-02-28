# type-tests for `linalg/_decomp_polar.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import polar

###
# Input arrays

py_i_2d: list[list[int]]
f32_2d: onp.Array2D[np.float32]
f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]
c64_2d: onp.Array2D[np.complex64]
c128_2d: onp.Array2D[np.complex128]
c128_3d: onp.Array3D[np.complex128]

###
# polar

assert_type(polar(py_i_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(polar(f64_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(polar(f64_3d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(polar(f32_2d), tuple[onp.Array2D[npc.floating], onp.Array2D[npc.floating]])
assert_type(polar(c128_2d), tuple[onp.Array2D[npc.complexfloating], onp.Array2D[npc.complexfloating]])
assert_type(polar(c128_3d), tuple[onp.ArrayND[npc.complexfloating], onp.ArrayND[npc.complexfloating]])
assert_type(polar(c64_2d), tuple[onp.Array2D[npc.complexfloating], onp.Array2D[npc.complexfloating]])
