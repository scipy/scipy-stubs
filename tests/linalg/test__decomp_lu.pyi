# type-tests for `linalg/_decomp_lu.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import lu, lu_factor, lu_solve

###

f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]
c128_1d: onp.Array1D[np.complex128]
c128_2d: onp.Array2D[np.complex128]
c128_3d: onp.Array3D[np.complex128]

###
# lu_factor

assert_type(lu_factor(f64_2d), tuple[onp.Array2D[npc.floating], onp.Array1D[npc.floating]])
assert_type(lu_factor(f64_3d), tuple[onp.ArrayND[npc.floating], onp.ArrayND[npc.floating]])
assert_type(lu_factor(c128_2d), tuple[onp.Array2D[npc.complexfloating], onp.Array1D[npc.complexfloating]])
assert_type(lu_factor(c128_3d), tuple[onp.ArrayND[npc.complexfloating], onp.ArrayND[npc.complexfloating]])

###
# lu_solve

assert_type(lu_solve((f64_2d, f64_1d), f64_1d), onp.Array2D[npc.floating])
assert_type(lu_solve((f64_3d, f64_3d), f64_3d), onp.ArrayND[npc.floating])
assert_type(lu_solve((c128_2d, c128_1d), c128_1d), onp.Array2D[npc.complexfloating])
assert_type(lu_solve((c128_3d, c128_3d), c128_3d), onp.ArrayND[npc.complexfloating])

###
# lu

assert_type(lu(f64_2d), tuple[onp.ArrayND[npc.floating], onp.ArrayND[npc.floating], onp.ArrayND[npc.floating]])
assert_type(
    lu(f64_2d, False, False, True, True), tuple[onp.ArrayND[np.intp], onp.ArrayND[npc.floating], onp.ArrayND[npc.floating]]
)
assert_type(lu(f64_2d, p_indices=True), tuple[onp.ArrayND[np.intp], onp.ArrayND[npc.floating], onp.ArrayND[npc.floating]])
assert_type(lu(f64_2d, True), tuple[onp.ArrayND[npc.floating], onp.ArrayND[npc.floating]])

assert_type(
    lu(c128_2d), tuple[onp.ArrayND[npc.complexfloating], onp.ArrayND[npc.complexfloating], onp.ArrayND[npc.complexfloating]]
)
assert_type(
    lu(c128_2d, False, False, True, True),
    tuple[onp.ArrayND[np.intp], onp.ArrayND[npc.complexfloating], onp.ArrayND[npc.complexfloating]],
)
assert_type(
    lu(c128_2d, p_indices=True), tuple[onp.ArrayND[np.intp], onp.ArrayND[npc.complexfloating], onp.ArrayND[npc.complexfloating]]
)
assert_type(lu(c128_2d, True), tuple[onp.ArrayND[npc.complexfloating], onp.ArrayND[npc.complexfloating]])
