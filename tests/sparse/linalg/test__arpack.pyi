from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import csr_array
from scipy.sparse.linalg import ArpackNoConvergence, eigs, eigsh

###

_a_i: csr_array[np.int64]
_a_f: csr_array[np.float64]
_a_c: csr_array[np.complex128]
_a_fc: csr_array[np.float64] | csr_array[np.complex128]

_a_f32: csr_array[np.float32]
_a_c128: csr_array[np.complex64]
_a_f32c64: csr_array[np.float32] | csr_array[np.complex64]

###

# ArpackNoConvergence
_exc: ArpackNoConvergence
assert_type(_exc.eigenvalues, onp.Array1D[np.float64 | np.complex128 | Any])
assert_type(_exc.eigenvectors, onp.Array2D[np.float64 | np.complex128 | Any])

# eigs
assert_type(eigs(_a_i), tuple[onp.Array1D[np.complex128], onp.Array2D[np.complex128]])
assert_type(eigs(_a_f), tuple[onp.Array1D[np.complex128], onp.Array2D[np.complex128]])
assert_type(eigs(_a_c), tuple[onp.Array1D[np.complex128], onp.Array2D[np.complex128]])
assert_type(eigs(_a_fc), tuple[onp.Array1D[np.complex128], onp.Array2D[np.complex128]])
assert_type(eigs(_a_f32), tuple[onp.Array1D[np.complex64], onp.Array2D[np.complex64]])
assert_type(eigs(_a_c128), tuple[onp.Array1D[np.complex64], onp.Array2D[np.complex64]])
assert_type(eigs(_a_f32c64), tuple[onp.Array1D[np.complex64], onp.Array2D[np.complex64]])
assert_type(eigs(_a_f, return_eigenvectors=False), onp.Array1D[np.complex128])
assert_type(eigs(_a_c, return_eigenvectors=False), onp.Array1D[np.complex128])
assert_type(eigs(_a_f32, return_eigenvectors=False), onp.Array1D[np.complex64])
assert_type(eigs(_a_c128, return_eigenvectors=False), onp.Array1D[np.complex64])

# eigsh
assert_type(eigsh(_a_i), tuple[onp.Array1D[np.float64], onp.Array2D[np.float64]])
assert_type(eigsh(_a_f), tuple[onp.Array1D[np.float64], onp.Array2D[np.float64]])
assert_type(eigsh(_a_c), tuple[onp.Array1D[np.float64], onp.Array2D[np.complex128]])
assert_type(eigsh(_a_fc), tuple[onp.Array1D[np.float64], onp.Array2D[np.float64 | np.complex128]])
assert_type(eigsh(_a_f32), tuple[onp.Array1D[np.float32], onp.Array2D[np.float32]])
assert_type(eigsh(_a_c128), tuple[onp.Array1D[np.float32], onp.Array2D[np.complex64]])
assert_type(eigsh(_a_f32c64), tuple[onp.Array1D[np.float32], onp.Array2D[np.float32 | np.complex64]])
assert_type(eigsh(_a_f, return_eigenvectors=False), onp.Array1D[np.float64])
assert_type(eigsh(_a_c, return_eigenvectors=False), onp.Array1D[np.float64])
assert_type(eigsh(_a_f32, return_eigenvectors=False), onp.Array1D[np.float32])
assert_type(eigsh(_a_c128, return_eigenvectors=False), onp.Array1D[np.float32])
