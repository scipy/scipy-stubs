# type-tests for `linalg/interpolative.pyi`

from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

import scipy.linalg.interpolative as interp
from scipy.sparse.linalg import LinearOperator

###

rng: np.random.Generator
lo: LinearOperator
f64_2d: onp.Array2D[np.float64]
c128_2d: onp.Array2D[np.complex128]
f64_nd: onp.ArrayND[np.float64]
intp_1d: onp.Array1D[np.intp]
f64_proj: onp.ArrayND[np.float64]
c128_proj: onp.ArrayND[np.complex128]

_Inexact1D: TypeAlias = onp.Array1D[npc.inexact]
_Inexact2D: TypeAlias = onp.Array2D[npc.inexact]

###
# interp_decomp

assert_type(interp.interp_decomp(f64_2d, 1e-6), tuple[int, onp.ArrayND[np.intp], onp.ArrayND[np.float64]])
assert_type(interp.interp_decomp(lo, 3), tuple[int, onp.ArrayND[np.intp], onp.ArrayND[np.float64]])

###
# reconstruct_matrix_from_id

assert_type(interp.reconstruct_matrix_from_id(f64_nd, intp_1d, f64_proj), onp.ArrayND[npc.number])
assert_type(interp.reconstruct_matrix_from_id(c128_2d, intp_1d, c128_proj), onp.ArrayND[npc.number])

###
# reconstruct_interp_matrix

assert_type(interp.reconstruct_interp_matrix(intp_1d, f64_proj), onp.ArrayND[np.float64 | np.complex128])

###
# reconstruct_skel_matrix

assert_type(interp.reconstruct_skel_matrix(f64_2d, 3, intp_1d), onp.ArrayND[np.float64])

###
# id_to_svd

assert_type(interp.id_to_svd(f64_nd, intp_1d, f64_proj), tuple[_Inexact2D, _Inexact1D, _Inexact2D])
assert_type(interp.id_to_svd(c128_2d, intp_1d, c128_proj), tuple[_Inexact2D, _Inexact1D, _Inexact2D])

###
# svd

assert_type(interp.svd(f64_2d, 1e-6), tuple[_Inexact2D, _Inexact1D, _Inexact2D])
assert_type(interp.svd(lo, 3), tuple[_Inexact2D, _Inexact1D, _Inexact2D])

###
# estimate_*

assert_type(interp.estimate_spectral_norm(lo), float | np.float64)
assert_type(interp.estimate_spectral_norm_diff(lo, lo), float | np.float64)
assert_type(interp.estimate_rank(f64_2d, 1e-6), int)
assert_type(interp.estimate_rank(lo, 1e-6), int)
