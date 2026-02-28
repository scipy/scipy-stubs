# type-tests for `linalg/_decomp_schur.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import rsf2csf, schur

###

f64_2d: onp.Array2D[np.float64]
c128_2d: onp.Array2D[np.complex128]

###
# schur

assert_type(schur(f64_2d), tuple[onp.ArrayND[npc.floating], onp.ArrayND[npc.floating]])
assert_type(schur(f64_2d, sort="lhp"), tuple[onp.ArrayND[npc.inexact], onp.ArrayND[npc.inexact], int])

assert_type(schur(c128_2d), tuple[onp.ArrayND[npc.inexact], onp.ArrayND[npc.inexact]])
assert_type(schur(c128_2d, sort="lhp"), tuple[onp.ArrayND[npc.inexact], onp.ArrayND[npc.inexact], int])

assert_type(schur(c128_2d, "complex"), tuple[onp.ArrayND[npc.complexfloating], onp.ArrayND[npc.complexfloating]])
assert_type(schur(c128_2d, "complex", sort="lhp"), tuple[onp.ArrayND[npc.complexfloating], onp.ArrayND[npc.complexfloating], int])

###
# rsf2csf

assert_type(rsf2csf(f64_2d, c128_2d), tuple[onp.ArrayND[npc.complexfloating], onp.ArrayND[npc.complexfloating]])
