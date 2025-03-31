from typing import Any, SupportsIndex, TypeAlias, TypeVar
from typing_extensions import deprecated

import numpy as np
import optype.numpy as onp
from scipy._typing import ToRNG
from scipy.sparse.linalg import LinearOperator

__all__ = [
    "estimate_rank",
    "estimate_spectral_norm",
    "estimate_spectral_norm_diff",
    "id_to_svd",
    "interp_decomp",
    "rand",
    "reconstruct_interp_matrix",
    "reconstruct_matrix_from_id",
    "reconstruct_skel_matrix",
    "seed",
    "svd",
]

_DT = TypeVar("_DT", bound=np.dtype[np.generic])
_Inexact1D: TypeAlias = onp.Array1D[np.inexact[Any]]
_Inexact2D: TypeAlias = onp.Array2D[np.inexact[Any]]

_AnyNumber: TypeAlias = np.number[Any]

@deprecated("will be removed in SciPy 1.17.0.")
def seed(seed: None = None) -> None: ...
@deprecated("will be removed in SciPy 1.17.0.")
def rand(*shape: int | bool) -> onp.ArrayND[np.float64]: ...

#
def interp_decomp(
    A: onp.ArrayND[_AnyNumber] | LinearOperator,
    eps_or_k: onp.ToFloat,
    rand: bool = True,
    rng: ToRNG = None,
) -> tuple[int | bool, onp.ArrayND[np.intp], onp.ArrayND[np.float64]]: ...

#
def reconstruct_matrix_from_id(
    B: onp.ArrayND,
    idx: onp.ArrayND[np.integer[Any]],
    proj: onp.ArrayND[_AnyNumber],
) -> onp.ArrayND[_AnyNumber]: ...

#
def reconstruct_interp_matrix(
    idx: onp.ArrayND[np.integer[Any]],
    proj: onp.ArrayND[_AnyNumber],
) -> onp.ArrayND[np.float64 | np.complex128]: ...

#
def reconstruct_skel_matrix(
    A: np.ndarray[tuple[int | bool, ...], _DT],
    k: SupportsIndex,
    idx: onp.ArrayND[np.integer[Any]],
) -> np.ndarray[tuple[int | bool, ...], _DT]: ...

#
def id_to_svd(
    B: onp.ArrayND,
    idx: onp.ArrayND[np.integer[Any]],
    proj: onp.ArrayND[_AnyNumber],
) -> tuple[_Inexact2D, _Inexact1D, _Inexact2D]: ...

#
def svd(
    A: onp.ArrayND[_AnyNumber] | LinearOperator,
    eps_or_k: onp.ToFloat,
    rand: bool = True,
    rng: ToRNG = None,
) -> tuple[_Inexact2D, _Inexact1D, _Inexact2D]: ...

#
def estimate_spectral_norm(A: LinearOperator, its: int | bool = 20, rng: ToRNG = None) -> float | int | bool | np.float64: ...
def estimate_spectral_norm_diff(
    A: LinearOperator, B: LinearOperator, its: int | bool = 20, rng: ToRNG = None
) -> float | int | bool | np.float64: ...
def estimate_rank(A: onp.ArrayND[_AnyNumber] | LinearOperator, eps: onp.ToFloat, rng: ToRNG = None) -> int | bool: ...
