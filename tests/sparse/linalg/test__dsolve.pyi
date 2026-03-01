from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import bsr_array, csc_array, csr_array, lil_array
from scipy.sparse.linalg import (
    SuperLU,
    factorized,
    is_sptriangular,
    spbandwidth,
    spilu,
    splu,
    spsolve,
    spsolve_triangular,
    use_solver,
)

i16_mat: csc_array[np.int16] | np.ndarray[tuple[int, int], np.dtype[np.int16]]
f32_mat: csc_array[np.float32] | np.ndarray[tuple[int, int], np.dtype[np.float32]]
f64_mat: csc_array[np.float64] | np.ndarray[tuple[int, int], np.dtype[np.float64]]
c64_mat: csc_array[np.complex64] | np.ndarray[tuple[int, int], np.dtype[np.complex64]]
c128_mat: csc_array[np.complex128] | np.ndarray[tuple[int, int], np.dtype[np.complex128]]
fc_float_mat: csc_array[np.float32] | csc_array[np.float64]
fc_complex_mat: csc_array[np.complex64] | csc_array[np.complex128]

b_f: onp.Array1D[np.float64]
b_f2d: onp.Array2D[np.float64]
b_f32: onp.Array1D[np.float32]
b_f32_2d: onp.Array2D[np.float32]
b_c: onp.Array1D[np.complex128]
b_c2d: onp.Array2D[np.complex128]
b_c64: onp.Array1D[np.complex64]
b_c64_2d: onp.Array2D[np.complex64]
b_sparse: csc_array[np.float64]
bsr_: bsr_array
csr_: csr_array
lil_: lil_array

# use_solver
assert_type(use_solver(useUmfpack=False), None)
assert_type(use_solver(useUmfpack=True), None)
assert_type(use_solver(), None)

# factorized
solve_f64 = factorized(f64_mat)
assert_type(solve_f64(b_f), onp.Array1D[np.float64])
assert_type(solve_f64(b_f2d), onp.Array2D[np.float64])

solve_c128 = factorized(c128_mat)
assert_type(solve_c128(b_c), onp.Array1D[np.complex128])
assert_type(solve_c128(b_c2d), onp.Array2D[np.complex128])

solve_f32 = factorized(f32_mat)
assert_type(solve_f32(b_f32), onp.Array1D[np.float32])
assert_type(solve_f32(b_f32_2d), onp.Array2D[np.float32])

solve_c64 = factorized(c64_mat)
assert_type(solve_c64(b_c64), onp.Array1D[np.complex64])
assert_type(solve_c64(b_c64_2d), onp.Array2D[np.complex64])

# spsolve
assert_type(spsolve(f64_mat, b_sparse), csc_array[np.float64])
assert_type(spsolve(f64_mat, b_f), onp.Array1D[np.float64])
assert_type(spsolve(f64_mat, b_f2d), onp.Array2D[np.float64])
assert_type(spsolve(c128_mat, b_c), onp.Array1D[np.complex128])
assert_type(spsolve(c128_mat, b_c2d), onp.Array2D[np.complex128])

# spsolve_triangular
assert_type(spsolve_triangular(f64_mat, b_f), onp.Array1D[np.float64])
assert_type(spsolve_triangular(f64_mat, b_f2d), onp.Array2D[np.float64])
assert_type(spsolve_triangular(c128_mat, b_c), onp.Array1D[np.complex128])
assert_type(spsolve_triangular(c128_mat, b_c2d), onp.Array2D[np.complex128])

# splu
assert_type(splu(i16_mat), SuperLU[np.float64])
assert_type(splu(f32_mat), SuperLU[np.float32])
assert_type(splu(f64_mat), SuperLU[np.float64])
assert_type(splu(c64_mat), SuperLU[np.complex64])
assert_type(splu(c128_mat), SuperLU[np.complex128])
assert_type(splu(fc_float_mat), SuperLU[np.float32 | np.float64])
assert_type(splu(fc_complex_mat), SuperLU[np.complex64 | np.complex128])

# spilu
assert_type(spilu(i16_mat), SuperLU[np.float64])
assert_type(spilu(f32_mat), SuperLU[np.float32])
assert_type(spilu(f64_mat), SuperLU[np.float64])
assert_type(spilu(c64_mat), SuperLU[np.complex64])
assert_type(spilu(c128_mat), SuperLU[np.complex128])
assert_type(spilu(fc_float_mat), SuperLU[np.float32 | np.float64])
assert_type(spilu(fc_complex_mat), SuperLU[np.complex64 | np.complex128])

# is_sptriangular
assert_type(is_sptriangular(csr_), tuple[bool, bool])
assert_type(is_sptriangular(bsr_), tuple[bool, bool])  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]

# spbandwidth
assert_type(spbandwidth(csr_), tuple[int, int])
assert_type(spbandwidth(bsr_), tuple[int, int])  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
assert_type(spbandwidth(lil_), tuple[int, int])  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
