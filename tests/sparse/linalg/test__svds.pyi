from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse.linalg import svds

###

_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]

###
# svds

assert_type(svds(_f32_2d), tuple[onp.Array2D[np.float32], onp.Array1D[np.float32], onp.Array2D[np.float32]])
assert_type(svds(_f64_2d), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64], onp.Array2D[np.float64]])
assert_type(svds(_c64_2d), tuple[onp.Array2D[np.complex64], onp.Array1D[np.float32], onp.Array2D[np.complex64]])
assert_type(svds(_c128_2d), tuple[onp.Array2D[np.complex128], onp.Array1D[np.float64], onp.Array2D[np.complex128]])

assert_type(svds(_f32_2d, return_singular_vectors=False), onp.Array1D[np.float32])
assert_type(svds(_f64_2d, return_singular_vectors=False), onp.Array1D[np.float64])
assert_type(svds(_c64_2d, return_singular_vectors=False), onp.Array1D[np.float32])
assert_type(svds(_c128_2d, return_singular_vectors=False), onp.Array1D[np.float64])

assert_type(svds(_f32_2d, return_singular_vectors="u"), tuple[onp.Array2D[np.float32], onp.Array1D[np.float32], None])
assert_type(svds(_f64_2d, return_singular_vectors="u"), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64], None])
assert_type(svds(_c64_2d, return_singular_vectors="u"), tuple[onp.Array2D[np.complex64], onp.Array1D[np.float32], None])
assert_type(svds(_c128_2d, return_singular_vectors="u"), tuple[onp.Array2D[np.complex128], onp.Array1D[np.float64], None])

assert_type(svds(_f32_2d, return_singular_vectors="vh"), tuple[None, onp.Array1D[np.float32], onp.Array2D[np.float32]])
assert_type(svds(_f64_2d, return_singular_vectors="vh"), tuple[None, onp.Array1D[np.float64], onp.Array2D[np.float64]])
assert_type(svds(_c64_2d, return_singular_vectors="vh"), tuple[None, onp.Array1D[np.float32], onp.Array2D[np.complex64]])
assert_type(svds(_c128_2d, return_singular_vectors="vh"), tuple[None, onp.Array1D[np.float64], onp.Array2D[np.complex128]])
