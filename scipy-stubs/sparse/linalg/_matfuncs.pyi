from typing import Any, Final, Generic, Literal, Self, SupportsIndex, overload, override
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._interface import LinearOperator
from scipy.sparse import (
    bsr_array,
    bsr_matrix,
    coo_array,
    coo_matrix,
    csc_array,
    csc_matrix,
    csr_array,
    csr_matrix,
    dia_array,
    dia_matrix,
    dok_array,
    dok_matrix,
    lil_array,
    lil_matrix,
)
from scipy.sparse._base import _spbase

__all__ = ["expm", "inv", "matrix_power"]

###

type _Structure = Literal["upper_triangular"]

type _CS[ScalarT: npc.number | np.bool] = csc_array[ScalarT] | csc_matrix[ScalarT] | csr_array[ScalarT] | csr_matrix[ScalarT]
type _NonCS[ScalarT: npc.number | np.bool] = (
    bsr_array[ScalarT]
    | bsr_matrix[ScalarT]
    | coo_array[ScalarT]
    | coo_matrix[ScalarT]
    | dok_array[ScalarT]
    | dok_matrix[ScalarT]
    | dia_array[ScalarT]
    | dia_matrix[ScalarT]
    | lil_array[ScalarT]
    | lil_matrix[ScalarT]
)

type _SubF64 = npc.integer64 | npc.integer32
type _SubF32 = npc.integer16 | npc.integer8 | np.bool

_SCT_co = TypeVar("_SCT_co", bound=npc.number | np.bool, default=Any, covariant=True)

###

UPPER_TRIANGULAR: Final[_Structure] = "upper_triangular"

class MatrixPowerOperator(LinearOperator[_SCT_co], Generic[_SCT_co]):
    @property
    @override
    # pyrefly: ignore [bad-override]
    def T(self, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def __init__(self, /, A: onp.Array2D[_SCT_co] | _spbase, p: int, structure: _Structure | None = None) -> None: ...

class ProductOperator(LinearOperator[_SCT_co], Generic[_SCT_co]):
    @property
    @override
    # pyrefly: ignore [bad-override]
    def T(self, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def __init__(self, /, *args: onp.Array2D[_SCT_co] | _spbase, structure: _Structure | None = None) -> None: ...

#
@overload
def inv[ScalarT: npc.inexact](A: _NonCS[ScalarT]) -> csc_array[ScalarT]: ...
@overload
def inv(A: _NonCS[_SubF64] | csc_array[_SubF64]) -> csc_array[np.float64]: ...  # type: ignore[overload-overlap]
@overload
def inv(A: _NonCS[_SubF32] | csc_array[_SubF32]) -> csc_array[np.float32]: ...
@overload
def inv[SparseT: _CS[npc.inexact]](A: SparseT) -> SparseT: ...
@overload
def inv(A: csr_array[_SubF64]) -> csr_array[np.float64]: ...  # type: ignore[overload-overlap]
@overload
def inv(A: csr_array[_SubF32]) -> csr_array[np.float32]: ...
@overload
def inv(A: csr_matrix[_SubF64]) -> csr_matrix[np.float64]: ...  # type: ignore[overload-overlap]
@overload
def inv(A: csr_matrix[_SubF32]) -> csr_matrix[np.float32]: ...
@overload
def inv(A: csc_matrix[_SubF64]) -> csc_matrix[np.float64]: ...  # type: ignore[overload-overlap]
@overload
def inv(A: csc_matrix[_SubF32]) -> csc_matrix[np.float32]: ...

#
def expm[SparseT: _spbase](A: SparseT) -> SparseT: ...

#
def matrix_power[SparseT: _spbase](A: SparseT, power: SupportsIndex) -> SparseT: ...
