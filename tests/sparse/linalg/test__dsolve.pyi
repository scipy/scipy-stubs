from typing import assert_type

import numpy as np

from scipy.sparse._csc import csc_array
from scipy.sparse.linalg import SuperLU, spilu, splu, use_solver

i16_mat: csc_array[np.int16] | np.ndarray[tuple[int, int], np.dtype[np.int16]]
f32_mat: csc_array[np.float32] | np.ndarray[tuple[int, int], np.dtype[np.float32]]
f64_mat: csc_array[np.float64] | np.ndarray[tuple[int, int], np.dtype[np.float64]]
c64_mat: csc_array[np.complex64] | np.ndarray[tuple[int, int], np.dtype[np.complex64]]
c128_mat: csc_array[np.complex128] | np.ndarray[tuple[int, int], np.dtype[np.complex128]]

# use_solver
assert_type(use_solver(useUmfpack=False), None)
assert_type(use_solver(useUmfpack=True), None)
assert_type(use_solver(), None)

# factorized
# TODO(jorenham): type-tests, see https://github.com/scipy/scipy-stubs/issues/677

# spsolve
# TODO(jorenham): type-tests

# spsolve_triangular
# TODO(jorenham): type-tests

# splu
assert_type(splu(i16_mat), SuperLU[np.float64])
assert_type(splu(f32_mat), SuperLU[np.float32])
assert_type(splu(f64_mat), SuperLU[np.float64])
assert_type(splu(c64_mat), SuperLU[np.complex64])
assert_type(splu(c128_mat), SuperLU[np.complex128])

# spilu
assert_type(spilu(i16_mat), SuperLU[np.float64])
assert_type(spilu(f32_mat), SuperLU[np.float32])
assert_type(spilu(f64_mat), SuperLU[np.float64])
assert_type(spilu(c64_mat), SuperLU[np.complex64])
assert_type(spilu(c128_mat), SuperLU[np.complex128])

# is_sptriangular
# TODO(jorenham): type-tests

# spbandwidth
# TODO(jorenham): type-tests
#
