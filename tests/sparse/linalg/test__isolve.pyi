from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import csr_array
from scipy.sparse.linalg import bicg, bicgstab, cg, cgs, gmres, qmr

a_f64: csr_array[np.float64]
a_c128: csr_array[np.complex128]
b_f: onp.Array1D[np.float64]
b_c: onp.Array1D[np.complex128]

# bicg
assert_type(bicg(a_f64, b_f), tuple[onp.Array1D[np.float64], int])
assert_type(bicg(a_c128, b_c), tuple[onp.Array1D[np.complex128], int])

# bicgstab
assert_type(bicgstab(a_f64, b_f), tuple[onp.Array1D[np.float64], int])
assert_type(bicgstab(a_c128, b_c), tuple[onp.Array1D[np.complex128], int])

# cg
assert_type(cg(a_f64, b_f), tuple[onp.Array1D[np.float64], int])
assert_type(cg(a_c128, b_c), tuple[onp.Array1D[np.complex128], int])

# cgs
assert_type(cgs(a_f64, b_f), tuple[onp.Array1D[np.float64], int])

# gmres
assert_type(gmres(a_f64, b_f), tuple[onp.Array1D[np.float64], int])
assert_type(gmres(a_f64, b_f, callback_type="x"), tuple[onp.Array1D[np.float64], int])
assert_type(gmres(a_c128, b_c), tuple[onp.Array1D[np.complex128], int])
assert_type(gmres(a_c128, b_c, callback_type="x"), tuple[onp.Array1D[np.complex128], int])

# qmr
assert_type(qmr(a_f64, b_f), tuple[onp.Array1D[np.float64], int])
assert_type(qmr(a_c128, b_c), tuple[onp.Array1D[np.complex128], int])
