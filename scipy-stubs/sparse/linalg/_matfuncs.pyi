from typing import Any, Final, Generic, Literal, Self, SupportsIndex, override
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._interface import LinearOperator
from scipy.sparse._base import _spbase

__all__ = ["expm", "inv", "matrix_power"]

###

type _Structure = Literal["upper_triangular"]

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

def inv[SparseT: _spbase](A: SparseT) -> SparseT: ...
def expm[SparseT: _spbase](A: SparseT) -> SparseT: ...
def matrix_power[SparseT: _spbase](A: SparseT, power: SupportsIndex) -> SparseT: ...
