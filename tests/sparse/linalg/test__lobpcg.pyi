from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse.linalg import lobpcg

a: onp.Array2D[np.float64]
x: onp.Array2D[np.float64]

# lobpcg
assert_type(lobpcg(a, x), tuple[onp.Array1D[np.float64], onp.Array2D[np.float64 | np.complex64 | np.complex128]])
assert_type(
    lobpcg(a, x, None, None, None, None, None, True, 0, False, True),
    tuple[onp.Array1D[np.float64], onp.Array2D[np.float64 | np.complex64 | np.complex128], list[onp.Array0D[np.float64]]],
)
assert_type(
    lobpcg(a, x, retResidualNormsHistory=True),
    tuple[onp.Array1D[np.float64], onp.Array2D[np.float64 | np.complex64 | np.complex128], list[onp.Array0D[np.float64]]],
)
assert_type(
    lobpcg(a, x, None, None, None, None, None, True, 0, True, False),
    tuple[onp.Array1D[np.float64], onp.Array2D[np.float64 | np.complex64 | np.complex128], list[onp.Array0D[np.float64]]],
)
assert_type(
    lobpcg(a, x, retLambdaHistory=True),
    tuple[onp.Array1D[np.float64], onp.Array2D[np.float64 | np.complex64 | np.complex128], list[onp.Array0D[np.float64]]],
)
assert_type(
    lobpcg(a, x, None, None, None, None, None, True, 0, True, True),
    tuple[
        onp.Array1D[np.float64],
        onp.Array2D[np.float64 | np.complex64 | np.complex128],
        list[onp.Array0D[np.float64]],
        list[onp.Array0D[np.float64]],
    ],
)
assert_type(
    lobpcg(a, x, retLambdaHistory=True, retResidualNormsHistory=True),
    tuple[
        onp.Array1D[np.float64],
        onp.Array2D[np.float64 | np.complex64 | np.complex128],
        list[onp.Array0D[np.float64]],
        list[onp.Array0D[np.float64]],
    ],
)
