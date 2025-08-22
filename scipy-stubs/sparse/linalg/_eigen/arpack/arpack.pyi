from collections.abc import Mapping
from typing import Final, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp

from scipy.sparse._base import _spbase
from scipy.sparse.linalg import LinearOperator

__all__ = ["ArpackError", "ArpackNoConvergence", "eigs", "eigsh"]

_KT = TypeVar("_KT")

_ToComplexMatrix: TypeAlias = onp.ToComplex2D | LinearOperator | _spbase

_Which_eigs: TypeAlias = Literal["LM", "SM", "LR", "SR", "LI", "SI"]
_Which_eigsh: TypeAlias = Literal["LM", "SM", "LA", "SA", "BE"]
_OPpart: TypeAlias = Literal["r", "i"]
_Mode: TypeAlias = Literal["normal", "buckling", "cayley"]

###

class ArpackError(RuntimeError):
    def __init__(self, /, info: _KT, infodict: Mapping[_KT, str] | None = None) -> None: ...

class ArpackNoConvergence(ArpackError):
    eigenvalues: Final[onp.Array1D[np.float64 | np.complex128]]
    eigenvectors: Final[onp.Array2D[np.float64 | np.complex128]]
    def __init__(
        self,
        /,
        msg: str,
        eigenvalues: onp.Array1D[np.float64 | np.complex128],
        eigenvectors: onp.Array2D[np.float64 | np.complex128],
    ) -> None: ...

#
@overload  # returns_eigenvectors: truthy (default)
def eigs(
    A: _ToComplexMatrix,
    k: int = 6,
    M: _ToComplexMatrix | None = None,
    sigma: onp.ToComplex | None = None,
    which: _Which_eigs = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: onp.ToTrue = True,
    Minv: _ToComplexMatrix | None = None,
    OPinv: _ToComplexMatrix | None = None,
    OPpart: _OPpart | None = None,
) -> tuple[onp.Array1D[np.complex128], onp.Array2D[np.complex128]]: ...
@overload  # returns_eigenvectors: falsy (positional)
def eigs(
    A: _ToComplexMatrix,
    k: int,
    M: _ToComplexMatrix | None,
    sigma: onp.ToComplex | None,
    which: _Which_eigs,
    v0: onp.ToComplex1D | None,
    ncv: int | None,
    maxiter: int | None,
    tol: float,
    return_eigenvectors: onp.ToFalse,
    Minv: _ToComplexMatrix | None = None,
    OPinv: _ToComplexMatrix | None = None,
    OPpart: _OPpart | None = None,
) -> onp.Array1D[np.complex128]: ...
@overload  # returns_eigenvectors: falsy (keyword)
def eigs(
    A: _ToComplexMatrix,
    k: int = 6,
    M: _ToComplexMatrix | None = None,
    sigma: onp.ToComplex | None = None,
    which: _Which_eigs = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    *,
    return_eigenvectors: onp.ToFalse,
    Minv: _ToComplexMatrix | None = None,
    OPinv: _ToComplexMatrix | None = None,
    OPpart: _OPpart | None = None,
) -> onp.Array1D[np.complex128]: ...

#
@overload  # returns_eigenvectors: truthy (default)
def eigsh(
    A: _ToComplexMatrix,
    k: int = 6,
    M: _ToComplexMatrix | None = None,
    sigma: onp.ToFloat | None = None,
    which: _Which_eigsh = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: onp.ToTrue = True,
    Minv: _ToComplexMatrix | None = None,
    OPinv: _ToComplexMatrix | None = None,
    mode: _Mode = "normal",
) -> tuple[onp.Array1D[np.float64], onp.Array2D[np.float64 | np.complex128]]: ...
@overload  # returns_eigenvectors: falsy (positional)
def eigsh(
    A: _ToComplexMatrix,
    k: int,
    M: _ToComplexMatrix | None,
    sigma: onp.ToFloat | None,
    which: _Which_eigsh,
    v0: onp.ToComplex1D | None,
    ncv: int | None,
    maxiter: int | None,
    tol: float,
    return_eigenvectors: onp.ToFalse,
    Minv: _ToComplexMatrix | None = None,
    OPinv: _ToComplexMatrix | None = None,
    mode: _Mode = "normal",
) -> onp.Array1D[np.float64]: ...
@overload  # returns_eigenvectors: falsy (keyword)
def eigsh(
    A: _ToComplexMatrix,
    k: int = 6,
    M: _ToComplexMatrix | None = None,
    sigma: onp.ToFloat | None = None,
    which: _Which_eigsh = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    *,
    return_eigenvectors: onp.ToFalse,
    Minv: _ToComplexMatrix | None = None,
    OPinv: _ToComplexMatrix | None = None,
    mode: _Mode = "normal",
) -> onp.Array1D[np.float64]: ...
