# type-tests for `linalg/interpolative.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

import scipy.linalg.interpolative as interp
from scipy.sparse.linalg import LinearOperator

###

_rng: np.random.Generator

_lo_f64: LinearOperator[np.float64]
_lo_c128: LinearOperator[np.complex128]

_intp_1d: onp.Array1D[np.intp]

_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_c128_2d: onp.Array2D[np.complex128]
_c128_nd: onp.ArrayND[np.complex128]

###

# interp_decomp
assert_type(interp.interp_decomp(_f64_2d, 0), tuple[int, onp.Array1D[np.intp], onp.Array2D[np.float64]])
assert_type(interp.interp_decomp(_f64_2d, 1), tuple[onp.Array1D[np.intp], onp.Array2D[np.float64]])
assert_type(
    interp.interp_decomp(_f64_2d, 0.5),
    tuple[int, onp.Array1D[np.intp], onp.Array2D[np.float64]] | tuple[onp.Array1D[np.intp], onp.Array2D[np.float64]],
)
assert_type(interp.interp_decomp(_c128_2d, 0), tuple[int, onp.Array1D[np.intp], onp.Array2D[np.complex128]])
assert_type(interp.interp_decomp(_c128_2d, 1), tuple[onp.Array1D[np.intp], onp.Array2D[np.complex128]])
assert_type(
    interp.interp_decomp(_c128_2d, 0.5),
    tuple[int, onp.Array1D[np.intp], onp.Array2D[np.complex128]] | tuple[onp.Array1D[np.intp], onp.Array2D[np.complex128]],
)
assert_type(interp.interp_decomp(_lo_f64, 0), tuple[int, onp.Array1D[np.intp], onp.Array2D[np.float64]])
assert_type(interp.interp_decomp(_lo_f64, 1), tuple[onp.Array1D[np.intp], onp.Array2D[np.float64]])
assert_type(
    interp.interp_decomp(_lo_f64, 0.5),
    tuple[int, onp.Array1D[np.intp], onp.Array2D[np.float64]] | tuple[onp.Array1D[np.intp], onp.Array2D[np.float64]],
)
assert_type(interp.interp_decomp(_lo_c128, 0), tuple[int, onp.Array1D[np.intp], onp.Array2D[np.complex128]])
assert_type(interp.interp_decomp(_lo_c128, 1), tuple[onp.Array1D[np.intp], onp.Array2D[np.complex128]])
assert_type(
    interp.interp_decomp(_lo_c128, 0.5),
    tuple[int, onp.Array1D[np.intp], onp.Array2D[np.complex128]] | tuple[onp.Array1D[np.intp], onp.Array2D[np.complex128]],
)

# reconstruct_matrix_from_id
assert_type(interp.reconstruct_matrix_from_id(_f64_nd, _intp_1d, _f64_nd), onp.ArrayND[npc.number])
assert_type(interp.reconstruct_matrix_from_id(_c128_2d, _intp_1d, _c128_nd), onp.ArrayND[npc.number])

# reconstruct_interp_matrix
assert_type(interp.reconstruct_interp_matrix(_intp_1d, _f64_nd), onp.ArrayND[np.float64 | np.complex128])

# reconstruct_skel_matrix
assert_type(interp.reconstruct_skel_matrix(_f64_2d, 3, _intp_1d), onp.ArrayND[np.float64])

# id_to_svd
assert_type(
    interp.id_to_svd(_f64_nd, _intp_1d, _f64_nd),
    tuple[onp.Array2D[npc.inexact], onp.Array1D[npc.inexact], onp.Array2D[npc.inexact]],
)
assert_type(
    interp.id_to_svd(_c128_2d, _intp_1d, _c128_nd),
    tuple[onp.Array2D[npc.inexact], onp.Array1D[npc.inexact], onp.Array2D[npc.inexact]],
)

# svd
assert_type(interp.svd(_f64_2d, 1e-6), tuple[onp.Array2D[npc.inexact], onp.Array1D[npc.inexact], onp.Array2D[npc.inexact]])
assert_type(interp.svd(_lo_f64, 3), tuple[onp.Array2D[npc.inexact], onp.Array1D[npc.inexact], onp.Array2D[npc.inexact]])

# estimate_*
assert_type(interp.estimate_spectral_norm(_lo_f64), float | np.float64)
assert_type(interp.estimate_spectral_norm_diff(_lo_f64, _lo_f64), float | np.float64)
assert_type(interp.estimate_rank(_f64_2d, 1e-6), int)
assert_type(interp.estimate_rank(_lo_f64, 1e-6), int)
