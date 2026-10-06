# needed (once) for `numpy>=2.2.0`
# mypy: disable-error-code="overload-overlap"

from types import GenericAlias
from typing import Any, Generic, Self, SupportsIndex, overload, override, type_check_only
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._base import _spbase
from ._bsr import bsr_matrix
from ._coo import coo_matrix
from ._csc import csc_matrix
from ._csr import csr_matrix
from ._dia import dia_matrix
from ._dok import dok_matrix
from ._lil import lil_matrix
from ._typing import _Format

###

type _Scalar = npc.number | np.bool

type _StackedSparseMatrix[ST: _Scalar] = coo_matrix[ST] | csc_matrix[ST] | csr_matrix[ST]

_ScalarT_co = TypeVar("_ScalarT_co", bound=_Scalar, default=Any, covariant=True)

###

class spmatrix(_spbase[_ScalarT_co, tuple[int, int]], Generic[_ScalarT_co]):
    # NOTE: These two methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> _StackedSparseMatrix[_ScalarT_co]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__[ST: _Scalar](self, sctype: ST, /) -> _StackedSparseMatrix[ST]: ...

    #
    @override
    @type_check_only
    def __assoc_as_any__(self, /) -> spmatrix[Any]: ...

    #
    @property
    def _bsr_container(self, /) -> bsr_matrix[_ScalarT_co]: ...
    @property
    def _coo_container(self, /) -> coo_matrix[_ScalarT_co]: ...
    @property
    def _csc_container(self, /) -> csc_matrix[_ScalarT_co]: ...
    @property
    def _csr_container(self, /) -> csr_matrix[_ScalarT_co]: ...
    @property
    def _dia_container(self, /) -> dia_matrix[_ScalarT_co]: ...
    @property
    def _dok_container(self, /) -> dok_matrix[_ScalarT_co]: ...
    @property
    def _lil_container(self, /) -> lil_matrix[_ScalarT_co]: ...

    #
    @property
    @type_check_only
    def dtype(self, /) -> np.dtype[_ScalarT_co]: ...
    @property
    @override
    def shape(self, /) -> tuple[int, int]: ...
    def get_shape(self, /) -> tuple[int, int]: ...
    def set_shape(self, /, shape: tuple[SupportsIndex, SupportsIndex]) -> None: ...

    #
    def getmaxprint(self, /) -> int: ...
    def getformat(self, /) -> _Format: ...
    # NOTE: `axis` is only supported by `{coo,csc,csr,lil}_matrix`
    def getnnz(self, /, axis: None = None) -> int: ...
    def getH(self, /) -> Self: ...

    #
    @overload
    def getcol[SelfT: bsr_matrix | csc_matrix | csr_matrix](self: SelfT, /, j: onp.ToJustInt) -> SelfT: ...  # type: ignore[misc]
    @overload
    def getcol[ST: _Scalar](  # type: ignore[misc]
        self: coo_matrix[ST] | dia_matrix[ST] | dok_matrix[ST] | lil_matrix[ST], /, j: onp.ToJustInt
    ) -> csr_matrix[ST]: ...

    #
    def getrow(self, /, i: onp.ToJustInt) -> csr_matrix[_ScalarT_co]: ...

    # NOTE: mypy reports a false positive for overlapping overloads
    @overload
    def asfptype(self: spmatrix[np.bool | npc.integer8 | npc.integer16], /) -> spmatrix[np.float32]: ...
    @overload
    def asfptype(self: spmatrix[npc.integer32 | npc.integer64], /) -> spmatrix[np.float64]: ...
    @overload
    def asfptype(self, /) -> Self: ...

    #
    @classmethod
    def __class_getitem__(cls, arg: type | object, /) -> GenericAlias: ...
