from typing import Any, ClassVar, Generic, Literal, Never, TypeAlias, overload, type_check_only
from typing_extensions import TypeIs, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp

from ._base import sparray
from ._compressed import _cs_matrix
from ._matrix import spmatrix
from ._typing import Index1D, Numeric

__all__ = ["csr_array", "csr_matrix", "isspmatrix_csr"]

_SCT = TypeVar("_SCT", bound=Numeric, default=Any)
_AsSCT = TypeVar("_AsSCT", bound=Numeric)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int] | tuple[int, int], default=tuple[int, int], covariant=True)

# workaround for the typing-spec non-conformance regarding overload behavior of mypy and pyright
_NeitherD: TypeAlias = tuple[Never] | tuple[Never, Never]

###

class _csr_base(_cs_matrix[_SCT, _ShapeT_co], Generic[_SCT, _ShapeT_co]):
    _format: ClassVar = "csr"
    _allow_nd: ClassVar = 1, 2

    @property
    @override
    def ndim(self, /) -> Literal[1, 2]: ...
    @property
    @override
    def format(self, /) -> Literal["csr"]: ...

    #
    @overload
    def count_nonzero(self, /, axis: None = None) -> np.intp: ...
    @overload
    def count_nonzero(self: _csr_base[Any, _NeitherD], /, axis: op.CanIndex) -> onp.Array1D[np.intp] | Any: ...  # noqa: ANN401
    @overload
    def count_nonzero(self: csr_array[Any, tuple[int]], /, axis: op.CanIndex) -> np.intp: ...  # type: ignore[misc]
    @overload
    def count_nonzero(self: _csr_base[Any], /, axis: op.CanIndex) -> onp.Array1D[np.intp]: ...
    @overload
    def count_nonzero(self: csr_array[Any, Any], /, axis: op.CanIndex) -> onp.Array1D[np.intp] | Any: ...  # type: ignore[misc]  # noqa: ANN401

class csr_array(_csr_base[_SCT, _ShapeT_co], sparray[_SCT, _ShapeT_co], Generic[_SCT, _ShapeT_co]):
    # NOTE: These two methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> csr_array[_SCT, tuple[int, int]]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__(self, sctype: _AsSCT, /) -> csr_array[_AsSCT, tuple[int, int]]: ...

class csr_matrix(_csr_base[_SCT], spmatrix[_SCT], Generic[_SCT]):
    # NOTE: These two methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> csr_matrix[_SCT]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__(self, sctype: _AsSCT, /) -> csr_matrix[_AsSCT]: ...

    #
    @overload
    def getnnz(self, /, axis: None = None) -> int: ...
    @overload
    def getnnz(self, /, axis: op.CanIndex) -> Index1D: ...

def isspmatrix_csr(x: object) -> TypeIs[csr_matrix]: ...
