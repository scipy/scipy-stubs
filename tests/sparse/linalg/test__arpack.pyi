from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import csr_array
from scipy.sparse.linalg import eigs, eigsh

a_f: csr_array[np.float64]
a_c: csr_array[np.complex128]
a_fc: csr_array[np.float64] | csr_array[np.complex128]

# eigs
assert_type(eigs(a_f), tuple[onp.Array1D[np.complex128], onp.Array2D[np.complex128]])
assert_type(eigs(a_c), tuple[onp.Array1D[np.complex128], onp.Array2D[np.complex128]])
assert_type(eigs(a_f, 6, None, None, "LM", None, None, None, 0, False), onp.Array1D[np.complex128])
assert_type(eigs(a_c, 6, None, None, "LM", None, None, None, 0, False), onp.Array1D[np.complex128])
assert_type(eigs(a_f, return_eigenvectors=False), onp.Array1D[np.complex128])
assert_type(eigs(a_c, return_eigenvectors=False), onp.Array1D[np.complex128])

# eigsh
assert_type(eigsh(a_f), tuple[onp.Array1D[np.float64], onp.Array2D[np.float64]])
assert_type(eigsh(a_c), tuple[onp.Array1D[np.float64], onp.Array2D[np.complex128]])
assert_type(eigsh(a_fc), tuple[onp.Array1D[np.float64], onp.Array2D[np.float64 | np.complex128]])
assert_type(eigsh(a_f, 6, None, None, "LM", None, None, None, 0, False), onp.Array1D[np.float64])
assert_type(eigsh(a_c, 6, None, None, "LM", None, None, None, 0, False), onp.Array1D[np.float64])
assert_type(eigsh(a_f, return_eigenvectors=False), onp.Array1D[np.float64])
assert_type(eigsh(a_c, return_eigenvectors=False), onp.Array1D[np.float64])
