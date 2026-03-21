from collections.abc import Sequence
from typing import Any, Final, Literal, SupportsIndex, TypeAlias, TypeVar, overload
from typing_extensions import TypeIs

import numpy as np
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

_NumberT = TypeVar("_NumberT", bound=npc.number)
_ArrayT = TypeVar("_ArrayT", bound=np.ndarray[Any, Any])
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

_IndexArray: TypeAlias = onp.Array1D[npc.integer] | Sequence[int]

###

_DTYPE_ERROR: Final[ValueError] = ...  # undocumented
_TYPE_ERROR: Final[TypeError] = ...  # undocumented

# undocumented
@overload
def _C_contiguous_copy(A: _ArrayT) -> _ArrayT: ...
@overload
def _C_contiguous_copy(A: onp.ToJustFloat64_ND) -> onp.ArrayND[np.float64]: ...
@overload
def _C_contiguous_copy(A: onp.ToJustComplex128_ND) -> onp.ArrayND[np.complex128]: ...

# undocumented
def _is_real(A: onp.ArrayND[np.float64 | np.complex128, _ShapeT]) -> TypeIs[onp.ArrayND[np.float64, _ShapeT]]: ...

#
@overload  # f64, eps_or_k<1
def interp_decomp(
    A: onp.Array2D[np.float64] | LinearOperator[np.float64],
    eps_or_k: Literal[0, -1, -2, -3, -4],
    rand: bool = True,
    rng: onp.random.ToRNG | None = None,
) -> tuple[int, onp.Array1D[np.intp], onp.Array2D[np.float64]]: ...
@overload  # f64, eps_or_k>=1
def interp_decomp(
    A: onp.Array2D[np.float64] | LinearOperator[np.float64],
    eps_or_k: Literal[1, 2, 3, 4, 5],
    rand: bool = True,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.intp], onp.Array2D[np.float64]]: ...
@overload  # f64, eps_or_k unknown
def interp_decomp(
    A: onp.Array2D[np.float64] | LinearOperator[np.float64],
    eps_or_k: float,
    rand: bool = True,
    rng: onp.random.ToRNG | None = None,
) -> tuple[int, onp.Array1D[np.intp], onp.Array2D[np.float64]] | tuple[onp.Array1D[np.intp], onp.Array2D[np.float64]]: ...
@overload  # c128, eps_or_k<1
def interp_decomp(
    A: onp.Array2D[np.complex128] | LinearOperator[np.complex128],
    eps_or_k: Literal[0, -1, -2, -3, -4],
    rand: bool = True,
    rng: onp.random.ToRNG | None = None,
) -> tuple[int, onp.Array1D[np.intp], onp.Array2D[np.complex128]]: ...
@overload  # c128, eps_or_k>=1
def interp_decomp(
    A: onp.Array2D[np.complex128] | LinearOperator[np.complex128],
    eps_or_k: Literal[1, 2, 3, 4, 5],
    rand: bool = True,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.intp], onp.Array2D[np.complex128]]: ...
@overload  # c128, eps_or_k unknown
def interp_decomp(
    A: onp.Array2D[np.complex128] | LinearOperator[np.complex128],
    eps_or_k: float,
    rand: bool = True,
    rng: onp.random.ToRNG | None = None,
) -> tuple[int, onp.Array1D[np.intp], onp.Array2D[np.complex128]] | tuple[onp.Array1D[np.intp], onp.Array2D[np.complex128]]: ...

#
@overload
def reconstruct_matrix_from_id(B: onp.Array2D[np.float64], idx: _IndexArray, proj: onp.ToFloat2D) -> onp.Array2D[np.float64]: ...
@overload
def reconstruct_matrix_from_id(
    B: onp.Array2D[np.complex128], idx: _IndexArray, proj: onp.ToComplex2D
) -> onp.Array2D[np.complex128]: ...

#
@overload
def reconstruct_interp_matrix(idx: _IndexArray, proj: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]: ...
@overload
def reconstruct_interp_matrix(idx: _IndexArray, proj: onp.Array2D[np.complex128]) -> onp.Array2D[np.complex128]: ...

#
def reconstruct_skel_matrix(A: onp.Array2D[_NumberT], k: SupportsIndex, idx: _IndexArray) -> onp.Array2D[_NumberT]: ...

#
def id_to_svd(
    B: onp.ArrayND, idx: onp.ArrayND[npc.integer], proj: onp.ArrayND[npc.number]
) -> tuple[onp.Array2D[npc.inexact], onp.Array1D[npc.inexact], onp.Array2D[npc.inexact]]: ...

#
def svd(
    A: onp.ArrayND[npc.number] | LinearOperator, eps_or_k: onp.ToFloat, rand: bool = True, rng: onp.random.ToRNG | None = None
) -> tuple[onp.Array2D[npc.inexact], onp.Array1D[npc.inexact], onp.Array2D[npc.inexact]]: ...

#
def estimate_spectral_norm(A: LinearOperator, its: int = 20, rng: onp.random.ToRNG | None = None) -> float | np.float64: ...
def estimate_spectral_norm_diff(
    A: LinearOperator, B: LinearOperator, its: int = 20, rng: onp.random.ToRNG | None = None
) -> float | np.float64: ...
def estimate_rank(A: onp.ArrayND[npc.number] | LinearOperator, eps: onp.ToFloat, rng: onp.random.ToRNG | None = None) -> int: ...
