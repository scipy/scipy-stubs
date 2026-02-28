# type-tests for `linalg/_decomp_cholesky.pyi`

from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import cho_factor, cho_solve, cho_solve_banded, cholesky, cholesky_banded

_Float2D: TypeAlias = onp.Array2D[npc.floating]
_FloatND: TypeAlias = onp.ArrayND[npc.floating]
_Complex2D: TypeAlias = onp.Array2D[npc.inexact]
_ComplexND: TypeAlias = onp.ArrayND[npc.inexact]

###
# Input arrays

f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]
c128_1d: onp.Array1D[np.complex128]
c128_2d: onp.Array2D[np.complex128]
c128_3d: onp.Array3D[np.complex128]

###
# cholesky

assert_type(cholesky(f64_2d), _Float2D)
assert_type(cholesky(f64_3d), _FloatND)
assert_type(cholesky(c128_2d), _Complex2D)
assert_type(cholesky(c128_3d), _ComplexND)

###
# cho_factor

assert_type(cho_factor(f64_2d), tuple[_FloatND, bool])
assert_type(cho_factor(c128_2d), tuple[_ComplexND, bool])

###
# cho_solve

assert_type(cho_solve((f64_2d, False), f64_1d), _Float2D)
assert_type(cho_solve((f64_3d, False), f64_3d), _FloatND)
assert_type(cho_solve((c128_2d, False), c128_1d), _Complex2D)
assert_type(cho_solve((c128_3d, False), c128_3d), _ComplexND)

###
# cholesky_banded

assert_type(cholesky_banded(f64_2d), _Float2D)
assert_type(cholesky_banded(f64_3d), _FloatND)
assert_type(cholesky_banded(c128_2d), _Complex2D)
assert_type(cholesky_banded(c128_3d), _ComplexND)

###
# cho_solve_banded

assert_type(cho_solve_banded((f64_2d, False), c128_1d), _Complex2D)
assert_type(cho_solve_banded((f64_3d, False), c128_3d), _ComplexND)
assert_type(cho_solve_banded((c128_2d, False), c128_1d), _Complex2D)
assert_type(cho_solve_banded((c128_3d, False), c128_3d), _ComplexND)
