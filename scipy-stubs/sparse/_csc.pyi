from typing import Any, ClassVar, Generic, Literal, overload, type_check_only
from typing_extensions import TypeIs, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp

from ._base import sparray
from ._compressed import _cs_matrix
from ._matrix import spmatrix
from ._typing import Index1D, Numeric

__all__ = ["csc_array", "csc_matrix", "isspmatrix_csc"]

_SCT = TypeVar("_SCT", bound=Numeric, default=Any)
_AsSCT = TypeVar("_AsSCT", bound=Numeric)

###

class _csc_base(_cs_matrix[_SCT, tuple[int, int]], Generic[_SCT]):
    _format: ClassVar = "csc"

    @property
    @override
    def format(self, /) -> Literal["csc"]: ...
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...
    @property
    @override
    def shape(self, /) -> tuple[int, int]: ...

    #
    @overload
    def count_nonzero(self, /, axis: None = None) -> np.intp: ...
    @overload
    def count_nonzero(self, /, axis: op.CanIndex) -> onp.Array1D[np.intp]: ...

class csc_array(_csc_base[_SCT], sparray[_SCT, tuple[int, int]], Generic[_SCT]):
    # NOTE: These two methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> csc_array[_SCT]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__(self, sctype: _AsSCT, /) -> csc_array[_AsSCT]: ...

class csc_matrix(_csc_base[_SCT], spmatrix[_SCT], Generic[_SCT]):
    # NOTE: These two methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> csc_matrix[_SCT]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__(self, sctype: _AsSCT, /) -> csc_matrix[_AsSCT]: ...

    #
    @overload
    def getnnz(self, /, axis: None = None) -> int: ...
    @overload
    def getnnz(self, /, axis: op.CanIndex) -> Index1D: ...

def isspmatrix_csc(x: object) -> TypeIs[csc_matrix]: ...
