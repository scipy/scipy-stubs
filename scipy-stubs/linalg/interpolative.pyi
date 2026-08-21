from collections.abc import Sequence
from typing import Any, Final, SupportsIndex, overload
from typing_extensions import TypeIs

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.sparse.linalg import LinearOperator

__all__ = [
    "estimate_rank",
    "estimate_spectral_norm",
    "estimate_spectral_norm_diff",
    "id_to_svd",
    "interp_decomp",
    "reconstruct_interp_matrix",
    "reconstruct_matrix_from_id",
    "reconstruct_skel_matrix",
    "svd",
]

###

type _IndexArray = onp.Array1D[npc.integer] | Sequence[int]
type _ToLinOp[NumberT: npc.number] = onp.Array2D[NumberT] | LinearOperator[NumberT]

###

_DTYPE_ERROR: Final[ValueError] = ...  # undocumented
_TYPE_ERROR: Final[TypeError] = ...  # undocumented

# undocumented
@overload
def _C_contiguous_copy[ArrayT: np.ndarray[Any, Any]](A: ArrayT) -> ArrayT: ...
@overload
def _C_contiguous_copy(A: onp.ToJustFloat64_ND) -> onp.ArrayND[np.float64]: ...
@overload
def _C_contiguous_copy(A: onp.ToJustComplex128_ND) -> onp.ArrayND[np.complex128]: ...

# undocumented
def _is_real[ShapeT: tuple[int, ...]](A: onp.ArrayND[npc.inexact64, ShapeT]) -> TypeIs[onp.ArrayND[np.float64, ShapeT]]: ...

#
@overload  # f64, ~int
def interp_decomp(
    A: _ToLinOp[npc.floating64], eps_or_k: op.JustInt | npc.integer, rand: bool = True, rng: onp.random.ToRNG | None = None
) -> tuple[onp.Array1D[np.intp], onp.Array2D[np.float64]]: ...
@overload  # f64, ~float
def interp_decomp(
    A: _ToLinOp[npc.floating64], eps_or_k: op.JustFloat | npc.floating, rand: bool = True, rng: onp.random.ToRNG | None = None
) -> tuple[int, onp.Array1D[np.intp], onp.Array2D[np.float64]]: ...
@overload  # c128, ~int
def interp_decomp(
    A: _ToLinOp[npc.complexfloating128],
    eps_or_k: op.JustInt | npc.integer,
    rand: bool = True,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.intp], onp.Array2D[np.complex128]]: ...
@overload  # c128, ~float
def interp_decomp(
    A: _ToLinOp[npc.complexfloating128],
    eps_or_k: op.JustFloat | npc.floating,
    rand: bool = True,
    rng: onp.random.ToRNG | None = None,
) -> tuple[int, onp.Array1D[np.intp], onp.Array2D[np.complex128]]: ...

#
@overload
def reconstruct_matrix_from_id(
    B: onp.Array2D[npc.floating64], idx: _IndexArray, proj: onp.ToFloat2D
) -> onp.Array2D[np.float64]: ...
@overload
def reconstruct_matrix_from_id(
    B: onp.Array2D[npc.complexfloating128], idx: _IndexArray, proj: onp.ToComplex2D
) -> onp.Array2D[np.complex128]: ...

#
@overload
def reconstruct_interp_matrix(idx: _IndexArray, proj: onp.Array2D[npc.floating64]) -> onp.Array2D[np.float64]: ...
@overload
def reconstruct_interp_matrix(idx: _IndexArray, proj: onp.Array2D[npc.complexfloating128]) -> onp.Array2D[np.complex128]: ...

#
def reconstruct_skel_matrix[NumberT: npc.number](
    A: onp.Array2D[NumberT], k: SupportsIndex, idx: _IndexArray
) -> onp.Array2D[NumberT]: ...

#
@overload
def id_to_svd(
    B: onp.Array2D[npc.floating64], idx: onp.Array1D[np.int64], proj: onp.Array2D[npc.floating64]
) -> tuple[onp.Array2D[np.float64], onp.Array1D[np.float64], onp.Array2D[np.float64]]: ...
@overload
def id_to_svd(
    B: onp.Array2D[npc.complexfloating128], idx: onp.Array1D[np.int64], proj: onp.Array2D[npc.complexfloating128]
) -> tuple[onp.Array2D[np.complex128], onp.Array1D[np.float64], onp.Array2D[np.complex128]]: ...

#
@overload
def svd(
    A: _ToLinOp[npc.floating64], eps_or_k: float, rand: bool = True, rng: onp.random.ToRNG | None = None
) -> tuple[onp.Array2D[np.float64], onp.Array1D[np.float64], onp.Array2D[np.float64]]: ...
@overload
def svd(
    A: _ToLinOp[npc.complexfloating128], eps_or_k: float, rand: bool = True, rng: onp.random.ToRNG | None = None
) -> tuple[onp.Array2D[np.complex128], onp.Array1D[np.float64], onp.Array2D[np.complex128]]: ...

#
def estimate_spectral_norm(A: _ToLinOp[npc.inexact64], its: int = 20, rng: onp.random.ToRNG | None = None) -> float: ...

#
@overload
def estimate_spectral_norm_diff(
    A: _ToLinOp[npc.floating64], B: _ToLinOp[npc.floating64], its: int = 20, rng: onp.random.ToRNG | None = None
) -> float: ...
@overload
def estimate_spectral_norm_diff(
    A: _ToLinOp[npc.complexfloating128], B: _ToLinOp[npc.complexfloating128], its: int = 20, rng: onp.random.ToRNG | None = None
) -> float: ...

#
def estimate_rank(A: _ToLinOp[npc.inexact64], eps: float, rng: onp.random.ToRNG | None = None) -> int: ...
