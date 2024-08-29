from typing import SupportsIndex, TypeAlias, TypeVar
from typing_extensions import Never, deprecated

import numpy as np

import numpy.typing as npt
import scipy._typing as spt

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
_Array_fc_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.inexact[npt.NBitBase]]]
_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]

@deprecated("will be removed in SciPy 1.17.0.")
def seed(seed: Never = ...) -> None: ...
@deprecated("will be removed in SciPy 1.17.0.")
def rand(*shape: int) -> npt.NDArray[np.float64]: ...
def interp_decomp(
    A: npt.NDArray[np.number[npt.NBitBase]] | LinearOperator,
    eps_or_k: spt.AnyReal,
    rand: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[int, npt.NDArray[np.intp], npt.NDArray[np.float64]]: ...
def reconstruct_matrix_from_id(
    B: npt.NDArray[np.generic],
    idx: npt.NDArray[np.integer[npt.NBitBase]],
    proj: npt.NDArray[np.number[npt.NBitBase]],
) -> npt.NDArray[np.number[npt.NBitBase]]: ...
def reconstruct_interp_matrix(
    idx: npt.NDArray[np.integer[npt.NBitBase]],
    proj: npt.NDArray[np.number[npt.NBitBase]],
) -> npt.NDArray[np.float64 | np.complex128]: ...
def reconstruct_skel_matrix(
    A: np.ndarray[tuple[int, ...], _DT],
    k: SupportsIndex,
    idx: npt.NDArray[np.integer[npt.NBitBase]],
) -> np.ndarray[tuple[int, ...], _DT]: ...
def id_to_svd(
    B: npt.NDArray[np.generic],
    idx: npt.NDArray[np.integer[npt.NBitBase]],
    proj: npt.NDArray[np.number[npt.NBitBase]],
) -> tuple[_Array_fc_2d, _Array_fc_1d, _Array_fc_2d]: ...
def estimate_spectral_norm(A: LinearOperator, its: int = 20, rng: np.random.Generator | None = None) -> float | np.float64: ...
def estimate_spectral_norm_diff(
    A: LinearOperator,
    B: LinearOperator,
    its: int = 20,
    rng: np.random.Generator | None = None,
) -> float | np.float64: ...
def svd(
    A: npt.NDArray[np.number[npt.NBitBase]] | LinearOperator,
    eps_or_k: spt.AnyReal,
    rand: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[_Array_fc_2d, _Array_fc_1d, _Array_fc_2d]: ...
def estimate_rank(
    A: npt.NDArray[np.number[npt.NBitBase]] | LinearOperator,
    eps: spt.AnyReal,
    rng: np.random.Generator | None = None,
) -> int: ...
