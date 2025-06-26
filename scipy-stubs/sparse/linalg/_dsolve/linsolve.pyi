from collections.abc import Mapping
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, overload, type_check_only
from typing_extensions import deprecated

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._superlu import SuperLU
from scipy.sparse._base import SparseEfficiencyWarning, _spbase
from scipy.sparse._bsr import _bsr_base
from scipy.sparse._lil import _lil_base

__all__ = [
    "MatrixRankWarning",
    "factorized",
    "is_sptriangular",
    "spbandwidth",
    "spilu",
    "splu",
    "spsolve",
    "spsolve_triangular",
    "use_solver",
]

_SparseT = TypeVar("_SparseT", bound=_spbase)

_PermcSpec: TypeAlias = Literal["COLAMD", "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A"]
_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]
_Complex1D: TypeAlias = onp.Array1D[np.complex128]
_Complex2D: TypeAlias = onp.Array2D[np.complex128]

_ToFloatMat: TypeAlias = _spbase[npc.floating | npc.integer | np.bool_, tuple[int, int]] | onp.ToFloat2D
_ToFloatMatStrict: TypeAlias = _spbase[npc.floating | npc.integer | np.bool_, tuple[int, int]] | onp.ToFloatStrict2D
_ToComplexMat: TypeAlias = _spbase[npc.complexfloating, tuple[int, int]] | onp.ToJustComplex2D
_ToInexactMat: TypeAlias = _spbase[Any, tuple[int, int]] | onp.ToComplex2D
_ToInexactMatStrict: TypeAlias = _spbase[Any, tuple[int, int]] | onp.ToComplexStrict2D

@type_check_only
class _Solve(Protocol):
    @overload
    def __call__(self, b: onp.Array1D[npc.integer | npc.floating], /) -> _Float1D: ...
    @overload
    def __call__(self, b: onp.Array1D[npc.complexfloating], /) -> _Complex1D: ...
    @overload
    def __call__(self, b: onp.Array2D[npc.integer | npc.floating], /) -> _Float2D: ...
    @overload
    def __call__(self, b: onp.Array2D[npc.complexfloating], /) -> _Complex2D: ...
    @overload
    def __call__(self, b: onp.ArrayND[npc.integer | npc.floating], /) -> _Float1D | _Float2D: ...
    @overload
    def __call__(self, b: onp.ArrayND[npc.complexfloating], /) -> _Complex1D | _Complex2D: ...
    @overload
    def __call__(self, b: onp.ArrayND[npc.number], /) -> _Float1D | _Complex1D | _Float2D | _Complex2D: ...

###

class MatrixRankWarning(UserWarning): ...

def use_solver(*, useUmfpack: bool = ..., assumeSortedIndices: bool = ...) -> None: ...
def factorized(A: _ToInexactMat) -> _Solve: ...

#
@overload  # 2d float, sparse 2d
def spsolve(A: _ToInexactMat, b: _SparseT, permc_spec: _PermcSpec | None = None, use_umfpack: bool = True) -> _SparseT: ...
@overload  # 2d float, 1d float
def spsolve(
    A: _ToFloatMat, b: onp.ToFloatStrict1D, permc_spec: _PermcSpec | None = None, use_umfpack: bool = True
) -> onp.Array1D[np.float64]: ...
@overload  # 2d float, 2d float
def spsolve(
    A: _ToFloatMat, b: onp.ToFloatStrict2D, permc_spec: _PermcSpec | None = None, use_umfpack: bool = True
) -> onp.Array2D[np.float64]: ...
@overload  # 2d float, 1d or 2d float
def spsolve(
    A: _ToFloatMat, b: onp.ToFloat2D | onp.ToFloat1D, permc_spec: _PermcSpec | None = None, use_umfpack: bool = True
) -> onp.ArrayND[np.float64]: ...
@overload  # 2d complex, 1d complex
def spsolve(
    A: _ToComplexMat, b: onp.ToComplexStrict1D, permc_spec: _PermcSpec | None = None, use_umfpack: bool = True
) -> onp.Array1D[np.complex128]: ...
@overload  # 2d complex, 2d complex
def spsolve(
    A: _ToComplexMat, b: onp.ToComplexStrict2D, permc_spec: _PermcSpec | None = None, use_umfpack: bool = True
) -> onp.Array2D[np.complex128]: ...
@overload  # 2d complex, 1d or 2d complex
def spsolve(
    A: _ToComplexMat, b: onp.ToComplex2D | onp.ToComplex1D, permc_spec: _PermcSpec | None = None, use_umfpack: bool = True
) -> onp.ArrayND[np.complex128]: ...
@overload  # 2d inexact, 1d or 2d inexact
def spsolve(
    A: _ToInexactMat, b: onp.ToComplex2D | onp.ToComplex1D, permc_spec: _PermcSpec | None = None, use_umfpack: bool = True
) -> onp.ArrayND[np.float64 | np.complex128]: ...

#
@overload  # 2d float, 1d float
def spsolve_triangular(
    A: _ToFloatMat,
    b: onp.ToFloatStrict1D,
    lower: bool = True,
    overwrite_A: bool = False,
    overwrite_b: bool = False,
    unit_diagonal: bool = False,
) -> onp.Array1D[np.float64]: ...
@overload  # 2d float, 2d float
def spsolve_triangular(
    A: _ToFloatMat,
    b: _ToFloatMatStrict,
    lower: bool = True,
    overwrite_A: bool = False,
    overwrite_b: bool = False,
    unit_diagonal: bool = False,
) -> onp.Array2D[np.float64]: ...
@overload  # 2d float, 1d or 2d float
def spsolve_triangular(
    A: _ToFloatMat,
    b: _ToFloatMat | onp.ToFloat1D,
    lower: bool = True,
    overwrite_A: bool = False,
    overwrite_b: bool = False,
    unit_diagonal: bool = False,
) -> onp.ArrayND[np.float64]: ...
@overload  # 2d complex, 1d complex
def spsolve_triangular(
    A: _ToComplexMat,
    b: onp.ToComplexStrict1D,
    lower: bool = True,
    overwrite_A: bool = False,
    overwrite_b: bool = False,
    unit_diagonal: bool = False,
) -> onp.Array1D[np.complex128]: ...
@overload  # 2d complex, 2d complex
def spsolve_triangular(
    A: _ToComplexMat,
    b: _ToInexactMatStrict,
    lower: bool = True,
    overwrite_A: bool = False,
    overwrite_b: bool = False,
    unit_diagonal: bool = False,
) -> onp.Array2D[np.complex128]: ...
@overload  # 2d complex, 1d or 2d complex
def spsolve_triangular(
    A: _ToComplexMat,
    b: _ToInexactMat | onp.ToComplex1D,
    lower: bool = True,
    overwrite_A: bool = False,
    overwrite_b: bool = False,
    unit_diagonal: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload  # 2d inexact, 1d or 2d inexact
def spsolve_triangular(
    A: _ToInexactMat,
    b: _ToInexactMat | onp.ToComplex1D,
    lower: bool = True,
    overwrite_A: bool = False,
    overwrite_b: bool = False,
    unit_diagonal: bool = False,
) -> onp.ArrayND[np.float64 | np.complex128]: ...

#
@overload
def splu(
    A: _ToFloatMat,
    permc_spec: _PermcSpec | None = None,
    diag_pivot_thresh: onp.ToFloat | None = None,
    relax: int | None = None,
    panel_size: int | None = None,
    options: Mapping[str, object] | None = None,
) -> SuperLU[np.float64]: ...
@overload
def splu(
    A: _ToComplexMat,
    permc_spec: _PermcSpec | None = None,
    diag_pivot_thresh: onp.ToFloat | None = None,
    relax: int | None = None,
    panel_size: int | None = None,
    options: Mapping[str, object] | None = None,
) -> SuperLU[np.complex128]: ...
@overload
def splu(
    A: _ToInexactMat,
    permc_spec: _PermcSpec | None = None,
    diag_pivot_thresh: onp.ToFloat | None = None,
    relax: int | None = None,
    panel_size: int | None = None,
    options: Mapping[str, object] | None = None,
) -> SuperLU[np.float64 | np.complex128]: ...

#
@overload
def spilu(
    A: _ToFloatMat,
    drop_tol: onp.ToFloat | None = None,
    fill_factor: onp.ToFloat | None = None,
    drop_rule: str | None = None,
    permc_spec: _PermcSpec | None = None,
    diag_pivot_thresh: onp.ToFloat | None = None,
    relax: int | None = None,
    panel_size: int | None = None,
    options: Mapping[str, object] | None = None,
) -> SuperLU[np.float64]: ...
@overload
def spilu(
    A: _ToComplexMat,
    drop_tol: onp.ToFloat | None = None,
    fill_factor: onp.ToFloat | None = None,
    drop_rule: str | None = None,
    permc_spec: _PermcSpec | None = None,
    diag_pivot_thresh: onp.ToFloat | None = None,
    relax: int | None = None,
    panel_size: int | None = None,
    options: Mapping[str, object] | None = None,
) -> SuperLU[np.complex128]: ...
@overload
def spilu(
    A: _ToInexactMat,
    drop_tol: onp.ToFloat | None = None,
    fill_factor: onp.ToFloat | None = None,
    drop_rule: str | None = None,
    permc_spec: _PermcSpec | None = None,
    diag_pivot_thresh: onp.ToFloat | None = None,
    relax: int | None = None,
    panel_size: int | None = None,
    options: Mapping[str, object] | None = None,
) -> SuperLU[np.float64 | np.complex128]: ...

#
@overload
@deprecated("is_sptriangular needs sparse and not BSR format. Converting to CSR.", category=SparseEfficiencyWarning)
def is_sptriangular(A: _bsr_base) -> tuple[bool, bool]: ...
@overload
def is_sptriangular(A: _spbase) -> tuple[bool, bool]: ...

#
@overload
@deprecated("spbandwidth needs sparse format not LIL and BSR. Converting to CSR.", category=SparseEfficiencyWarning)
def spbandwidth(A: _bsr_base | _lil_base) -> tuple[int, int]: ...
@overload
def spbandwidth(A: _spbase) -> tuple[int, int]: ...
