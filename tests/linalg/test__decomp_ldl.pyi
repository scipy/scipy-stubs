# type-tests for `linalg/_decomp_ldl.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import ldl

###
# Input arrays

f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]
c128_2d: onp.Array2D[np.complex128]
c128_3d: onp.Array3D[np.complex128]

###
# ldl

assert_type(ldl(f64_2d), tuple[onp.Array2D[npc.floating], onp.Array2D[npc.floating], onp.Array1D[np.intp]])
assert_type(ldl(f64_3d), tuple[onp.ArrayND[npc.floating], onp.ArrayND[npc.floating], onp.ArrayND[np.intp]])
assert_type(ldl(c128_2d), tuple[onp.Array2D[npc.complexfloating], onp.Array2D[npc.complexfloating], onp.Array1D[np.intp]])
assert_type(ldl(c128_3d), tuple[onp.ArrayND[npc.complexfloating], onp.ArrayND[npc.complexfloating], onp.ArrayND[np.intp]])
