# type-tests for `linalg/_decomp_qz.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import ordqz, qz

###

f64_2d: onp.Array2D[np.float64]
c128_2d: onp.Array2D[np.complex128]

###
# qz

assert_type(
    qz(f64_2d, f64_2d),
    tuple[onp.ArrayND[npc.floating], onp.ArrayND[npc.floating], onp.ArrayND[npc.floating], onp.ArrayND[npc.floating]],
)
assert_type(
    qz(c128_2d, c128_2d),
    tuple[onp.ArrayND[npc.inexact], onp.ArrayND[npc.inexact], onp.ArrayND[npc.inexact], onp.ArrayND[npc.inexact]],
)
assert_type(
    qz(c128_2d, c128_2d, "complex"),
    tuple[
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.complexfloating],
        onp.ArrayND[npc.complexfloating],
    ],
)

###
# ordqz

assert_type(
    ordqz(f64_2d, f64_2d),
    tuple[
        onp.ArrayND[npc.floating],
        onp.ArrayND[npc.floating],
        onp.ArrayND[npc.floating],
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
        onp.ArrayND[npc.inexact],
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
