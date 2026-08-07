from collections.abc import Mapping
from typing import Any, Final, Literal, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator

__all__ = ["ArpackError", "ArpackNoConvergence", "eigs", "eigsh"]

###

type _Numeric = npc.number | np.bool
type _ToFloat = npc.floating | npc.integer | np.bool
type _ToF64 = npc.floating64 | npc.floating16 | npc.integer | np.bool
type _ToC64 = _ToF64 | npc.complexfloating128

type _MatOp[SCT: _Numeric] = spmatrix[SCT] | sparray[SCT, tuple[int, int]] | LinearOperator[SCT]

type _ToMatFloat = onp.ToFloat2D | _MatOp[_ToFloat]
type _ToMatF64 = onp.ToArray2D[float, _ToF64] | _MatOp[_ToF64]
type _ToMatComplex = onp.ToComplex2D | _MatOp[_Numeric]
type _ToMatC128 = onp.ToArray2D[complex, _ToC64] | _MatOp[_ToC64]

type _AsMatF32 = onp.ToJustFloat32_2D | _MatOp[npc.floating32]
type _AsMatC64 = onp.ToJustComplex64_2D | _MatOp[npc.complexfloating64]
type _AsMatC128 = onp.ToJustComplex128_2D | _MatOp[npc.complexfloating128]
type _AsMatF32C64 = onp.ToJustFloat32_2D | onp.ToJustComplex64_2D | _MatOp[npc.inexact32]

type _Which_eigs = Literal["LM", "SM", "LR", "SR", "LI", "SI"]
type _Which_eigsh = Literal["LM", "SM", "LA", "SA", "BE"]
type _OPpart = Literal["r", "i"]
type _Mode = Literal["normal", "buckling", "cayley"]

###

# NOTE: mypy incorrectly sees disjoint dtypes like `npc.floating32` and `npc.floating64` as overlapping
# mypy: disable-error-code=overload-overlap

class ArpackError(RuntimeError):
    def __init__[T](self, /, info: T, infodict: Mapping[T, str] | None = None) -> None: ...

class ArpackNoConvergence(ArpackError):
    eigenvalues: Final[onp.Array1D[np.float64 | np.complex128 | Any]]
    eigenvectors: Final[onp.Array2D[np.float64 | np.complex128 | Any]]

    def __init__(
        self,
        /,
        msg: str,
        eigenvalues: onp.Array1D[np.float64 | np.complex128 | Any],
        eigenvectors: onp.Array2D[np.float64 | np.complex128 | Any],
    ) -> None: ...

#
@overload  # ~f32 | ~c64, returns_eigenvectors: truthy (default)
def eigs(
    A: _AsMatF32C64,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToComplex | None = None,
    which: _Which_eigs = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: onp.ToTrue = True,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    OPpart: _OPpart | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.complex64], onp.Array2D[np.complex64]]: ...
@overload  # +f64 | ~c128, returns_eigenvectors: truthy (default)
def eigs(
    A: _ToMatC128,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToComplex | None = None,
    which: _Which_eigs = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: onp.ToTrue = True,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    OPpart: _OPpart | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.complex128], onp.Array2D[np.complex128]]: ...
@overload  # +complex (fallback), returns_eigenvectors: truthy (default)
def eigs(
    A: _ToMatComplex,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToComplex | None = None,
    which: _Which_eigs = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: onp.ToTrue = True,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    OPpart: _OPpart | None = None,
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.complex128 | Any], onp.Array2D[np.complex128 | Any]]: ...
@overload  # ~f32 | ~c64, returns_eigenvectors: falsy (keyword)
def eigs(
    A: _AsMatF32C64,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToComplex | None = None,
    which: _Which_eigs = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    *,
    return_eigenvectors: onp.ToFalse,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    OPpart: _OPpart | None = None,
    rng: onp.random.ToRNG | None = None,
) -> onp.Array1D[np.complex64]: ...
@overload  # +f64 | ~c128, returns_eigenvectors: falsy (keyword)
def eigs(
    A: _ToMatC128,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToComplex | None = None,
    which: _Which_eigs = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    *,
    return_eigenvectors: onp.ToFalse,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    OPpart: _OPpart | None = None,
    rng: onp.random.ToRNG | None = None,
) -> onp.Array1D[np.complex128]: ...
@overload  # +complex (fallback), returns_eigenvectors: falsy (keyword)
def eigs(
    A: _ToMatComplex,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToComplex | None = None,
    which: _Which_eigs = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    *,
    return_eigenvectors: onp.ToFalse,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    OPpart: _OPpart | None = None,
    rng: onp.random.ToRNG | None = None,
) -> onp.Array1D[np.complex128 | Any]: ...

#
@overload  # ~f32, returns_eigenvectors: truthy (default)
def eigsh(
    A: _AsMatF32,
    k: int = 6,
    M: _ToMatFloat | None = None,
    sigma: onp.ToFloat | None = None,
    which: _Which_eigsh = "LM",
    v0: onp.ToFloat1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: onp.ToTrue = True,
    Minv: _ToMatFloat | None = None,
    OPinv: _ToMatFloat | None = None,
    mode: _Mode = "normal",
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.float32], onp.Array2D[np.float32]]: ...
@overload  # ~c64, returns_eigenvectors: truthy (default)
def eigsh(
    A: _AsMatC64,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToFloat | None = None,
    which: _Which_eigsh = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: onp.ToTrue = True,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    mode: _Mode = "normal",
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.float32], onp.Array2D[np.complex64]]: ...
@overload  # ~f32 | ~c64, returns_eigenvectors: truthy (default)
def eigsh(
    A: _AsMatF32C64,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToFloat | None = None,
    which: _Which_eigsh = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: onp.ToTrue = True,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    mode: _Mode = "normal",
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.float32], onp.Array2D[np.float32 | np.complex64]]: ...
@overload  # +f64, returns_eigenvectors: truthy (default)
def eigsh(
    A: _ToMatF64,
    k: int = 6,
    M: _ToMatFloat | None = None,
    sigma: onp.ToFloat | None = None,
    which: _Which_eigsh = "LM",
    v0: onp.ToFloat1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: onp.ToTrue = True,
    Minv: _ToMatFloat | None = None,
    OPinv: _ToMatFloat | None = None,
    mode: _Mode = "normal",
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.float64], onp.Array2D[np.float64]]: ...
@overload  # ~c128, returns_eigenvectors: truthy (default)
def eigsh(
    A: _AsMatC128,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToFloat | None = None,
    which: _Which_eigsh = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: onp.ToTrue = True,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    mode: _Mode = "normal",
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.float64], onp.Array2D[np.complex128]]: ...
@overload  # +f64 | ~c128, returns_eigenvectors: truthy (default)
def eigsh(
    A: _ToMatC128,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToFloat | None = None,
    which: _Which_eigsh = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: onp.ToTrue = True,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    mode: _Mode = "normal",
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.float64], onp.Array2D[np.float64 | np.complex128]]: ...
@overload  # +complex (fallback), returns_eigenvectors: truthy (default)
def eigsh(
    A: _ToMatComplex,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToFloat | None = None,
    which: _Which_eigsh = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: onp.ToTrue = True,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    mode: _Mode = "normal",
    rng: onp.random.ToRNG | None = None,
) -> tuple[onp.Array1D[np.float64 | Any], onp.Array2D[np.float64 | np.complex128 | Any]]: ...
@overload  # ~f32 | ~c64, returns_eigenvectors: falsy (keyword)
def eigsh(
    A: _AsMatF32C64,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToFloat | None = None,
    which: _Which_eigsh = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    *,
    return_eigenvectors: onp.ToFalse,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    mode: _Mode = "normal",
    rng: onp.random.ToRNG | None = None,
) -> onp.Array1D[np.float32]: ...
@overload  # +f64 | ~c128, returns_eigenvectors: falsy (keyword)
def eigsh(
    A: _ToMatC128,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToFloat | None = None,
    which: _Which_eigsh = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    *,
    return_eigenvectors: onp.ToFalse,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    mode: _Mode = "normal",
    rng: onp.random.ToRNG | None = None,
) -> onp.Array1D[np.float64]: ...
@overload  # +complex (fallback), returns_eigenvectors: falsy (keyword)
def eigsh(
    A: _ToMatComplex,
    k: int = 6,
    M: _ToMatComplex | None = None,
    sigma: onp.ToFloat | None = None,
    which: _Which_eigsh = "LM",
    v0: onp.ToComplex1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    *,
    return_eigenvectors: onp.ToFalse,
    Minv: _ToMatComplex | None = None,
    OPinv: _ToMatComplex | None = None,
    mode: _Mode = "normal",
    rng: onp.random.ToRNG | None = None,
) -> onp.Array1D[np.float64 | Any]: ...
