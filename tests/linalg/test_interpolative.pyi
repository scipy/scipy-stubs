# type-tests for `linalg/interpolative.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg.interpolative import (
    estimate_rank,
    estimate_spectral_norm,
    estimate_spectral_norm_diff,
    id_to_svd,
    interp_decomp,
    reconstruct_interp_matrix,
    reconstruct_matrix_from_id,
    reconstruct_skel_matrix,
    svd,
)
from scipy.sparse.linalg import LinearOperator

###

_rng: np.random.Generator

_lo_f64: LinearOperator[np.float64]
_lo_c128: LinearOperator[np.complex128]

_i64_1d: onp.Array1D[np.int64]

_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_c128_2d: onp.Array2D[np.complex128]
_c128_nd: onp.ArrayND[np.complex128]

_py_i_1d: list[int]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]

###

# interp_decomp
assert_type(interp_decomp(_f64_2d, 0), tuple[int, onp.Array1D[np.intp], onp.Array2D[np.float64]])
assert_type(interp_decomp(_f64_2d, 1), tuple[onp.Array1D[np.intp], onp.Array2D[np.float64]])
assert_type(
    interp_decomp(_f64_2d, 0.5),
    tuple[int, onp.Array1D[np.intp], onp.Array2D[np.float64]] | tuple[onp.Array1D[np.intp], onp.Array2D[np.float64]],
)
assert_type(interp_decomp(_c128_2d, 0), tuple[int, onp.Array1D[np.intp], onp.Array2D[np.complex128]])
assert_type(interp_decomp(_c128_2d, 1), tuple[onp.Array1D[np.intp], onp.Array2D[np.complex128]])
assert_type(
    interp_decomp(_c128_2d, 0.5),
    tuple[int, onp.Array1D[np.intp], onp.Array2D[np.complex128]] | tuple[onp.Array1D[np.intp], onp.Array2D[np.complex128]],
)
assert_type(interp_decomp(_lo_f64, 0), tuple[int, onp.Array1D[np.intp], onp.Array2D[np.float64]])
assert_type(interp_decomp(_lo_f64, 1), tuple[onp.Array1D[np.intp], onp.Array2D[np.float64]])
assert_type(
    interp_decomp(_lo_f64, 0.5),
    tuple[int, onp.Array1D[np.intp], onp.Array2D[np.float64]] | tuple[onp.Array1D[np.intp], onp.Array2D[np.float64]],
)
assert_type(interp_decomp(_lo_c128, 0), tuple[int, onp.Array1D[np.intp], onp.Array2D[np.complex128]])
assert_type(interp_decomp(_lo_c128, 1), tuple[onp.Array1D[np.intp], onp.Array2D[np.complex128]])
assert_type(
    interp_decomp(_lo_c128, 0.5),
    tuple[int, onp.Array1D[np.intp], onp.Array2D[np.complex128]] | tuple[onp.Array1D[np.intp], onp.Array2D[np.complex128]],
)

# reconstruct_matrix_from_id
assert_type(reconstruct_matrix_from_id(_f64_2d, _i64_1d, _f64_2d), onp.Array2D[np.float64])
assert_type(reconstruct_matrix_from_id(_f64_2d, _i64_1d, _py_f_2d), onp.Array2D[np.float64])
assert_type(reconstruct_matrix_from_id(_c128_2d, _i64_1d, _c128_2d), onp.Array2D[np.complex128])
assert_type(reconstruct_matrix_from_id(_c128_2d, _i64_1d, _py_c_2d), onp.Array2D[np.complex128])

# reconstruct_interp_matrix
assert_type(reconstruct_interp_matrix(_i64_1d, _f64_2d), onp.Array2D[np.float64])
assert_type(reconstruct_interp_matrix(_py_i_1d, _f64_2d), onp.Array2D[np.float64])
assert_type(reconstruct_interp_matrix(_i64_1d, _c128_2d), onp.Array2D[np.complex128])
assert_type(reconstruct_interp_matrix(_py_i_1d, _c128_2d), onp.Array2D[np.complex128])

# reconstruct_skel_matrix
assert_type(reconstruct_skel_matrix(_f64_2d, 3, _i64_1d), onp.Array2D[np.float64])
assert_type(reconstruct_skel_matrix(_c128_2d, 3, _i64_1d), onp.Array2D[np.complex128])

# id_to_svd
assert_type(
    id_to_svd(_f64_2d, _i64_1d, _f64_2d), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64], onp.Array2D[np.float64]]
)
assert_type(
    id_to_svd(_c128_2d, _i64_1d, _c128_2d), tuple[onp.Array2D[np.complex128], onp.Array1D[np.float64], onp.Array2D[np.complex128]]
)

# svd
assert_type(svd(_f64_2d, 1), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64], onp.Array2D[np.float64]])
assert_type(svd(_lo_f64, 1), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64], onp.Array2D[np.float64]])
assert_type(svd(_c128_2d, 1), tuple[onp.Array2D[np.complex128], onp.Array1D[np.float64], onp.Array2D[np.complex128]])
assert_type(svd(_lo_c128, 1), tuple[onp.Array2D[np.complex128], onp.Array1D[np.float64], onp.Array2D[np.complex128]])

# estimate_spectral_norm
assert_type(estimate_spectral_norm(_f64_2d), float)
assert_type(estimate_spectral_norm(_lo_f64), float)
assert_type(estimate_spectral_norm(_c128_2d), float)
assert_type(estimate_spectral_norm(_lo_c128), float)

# estimate_spectral_norm_diff
assert_type(estimate_spectral_norm_diff(_f64_2d, _f64_2d), float)
assert_type(estimate_spectral_norm_diff(_f64_2d, _lo_f64), float)
assert_type(estimate_spectral_norm_diff(_lo_f64, _f64_2d), float)
assert_type(estimate_spectral_norm_diff(_lo_f64, _lo_f64), float)
assert_type(estimate_spectral_norm_diff(_c128_2d, _c128_2d), float)
assert_type(estimate_spectral_norm_diff(_c128_2d, _lo_c128), float)
assert_type(estimate_spectral_norm_diff(_lo_c128, _c128_2d), float)
assert_type(estimate_spectral_norm_diff(_lo_c128, _lo_c128), float)

# estimate_rank
assert_type(estimate_rank(_f64_2d, 1), int)
assert_type(estimate_rank(_lo_f64, 1), int)
assert_type(estimate_rank(_c128_2d, 1), int)
assert_type(estimate_rank(_lo_c128, 1), int)
