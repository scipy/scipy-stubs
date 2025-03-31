# needed (once) for `numpy>=2.2.0`
# mypy: disable-error-code="overload-overlap"

from collections.abc import Sequence
from typing import Any, Generic, TypeAlias, overload
from typing_extensions import Self, TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import OrderCF
from ._base import _spbase
from ._bsr import bsr_matrix
from ._coo import coo_matrix
from ._csc import csc_matrix
from ._csr import csr_matrix
from ._dia import dia_matrix
from ._dok import dok_matrix
from ._lil import lil_matrix
from ._typing import CFloating, Floating, Integer, Numeric, SPFormat, ToShape2D

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Numeric)
_SCT_co = TypeVar("_SCT_co", bound=Numeric, default=Numeric, covariant=True)

_SpMatrixT = TypeVar("_SpMatrixT", bound=spmatrix)

_SpFromInT = TypeVar("_SpFromInT", bound=spmatrix[_FromInt])
_SpFromFloatT = TypeVar("_SpFromFloatT", bound=spmatrix[_FromFloat])
_SpFromComplexT = TypeVar("_SpFromComplexT", bound=spmatrix[_FromComplex])

_FromInt: TypeAlias = Integer | Floating | CFloating
_FromFloat: TypeAlias = Floating | CFloating
_FromComplex: TypeAlias = CFloating

_ToBool: TypeAlias = np.bool_
_ToInt8: TypeAlias = np.bool_ | np.int8
_ToInt: TypeAlias = Integer | _ToBool
_ToFloat32: TypeAlias = np.bool_ | Integer | np.float32
_ToFloat: TypeAlias = np.bool_ | Integer | Floating
_ToComplex64: TypeAlias = np.bool_ | Integer | Floating | np.complex64

_DualMatrixLike: TypeAlias = _T | _SCT | _spbase[_SCT]
_DualArrayLike: TypeAlias = Sequence[Sequence[_T | _SCT] | onp.CanArrayND[_SCT]] | onp.CanArrayND[_SCT]

_SpMatrixOut: TypeAlias = bsr_matrix[_SCT] | csc_matrix[_SCT] | csr_matrix[_SCT]

###

class spmatrix(Generic[_SCT_co]):
    @property
    def _bsr_container(self, /) -> bsr_matrix[_SCT_co]: ...
    @property
    def _coo_container(self, /) -> coo_matrix[_SCT_co]: ...
    @property
    def _csc_container(self, /) -> csc_matrix[_SCT_co]: ...
    @property
    def _csr_container(self, /) -> csr_matrix[_SCT_co]: ...
    @property
    def _dia_container(self, /) -> dia_matrix[_SCT_co]: ...
    @property
    def _dok_container(self, /) -> dok_matrix[_SCT_co]: ...
    @property
    def _lil_container(self, /) -> lil_matrix[_SCT_co]: ...

    #
    @property
    def shape(self, /) -> tuple[int | bool, int | bool]: ...
    def get_shape(self, /) -> tuple[int | bool, int | bool]: ...
    def set_shape(self, /, shape: ToShape2D) -> None: ...

    #
    @overload  # Self[-Bool], other: scalar-like +Bool
    def __mul__(self, other: bool | _ToBool, /) -> Self: ...
    @overload  # Self[-Int], other: scalar-like +Int
    def __mul__(self: _SpFromInT, other: onp.ToInt, /) -> _SpFromInT: ...
    @overload  # Self[-Float], other: scalar-like +Float
    def __mul__(self: _SpFromFloatT, other: onp.ToFloat, /) -> _SpFromFloatT: ...
    @overload  # Self[-Complex], other: scalar-like +Complex
    def __mul__(self: _SpFromComplexT, other: onp.ToComplex, /) -> _SpFromComplexT: ...
    @overload  # spmatrix, other: spmatrix
    def __mul__(self: _SpMatrixT, other: _SpMatrixT, /) -> _SpMatrixT: ...
    @overload  # spmatrix[-Bool], other: sparse +Bool
    def __mul__(self: spmatrix, other: _spbase[_ToBool], /) -> _SpMatrixOut[_SCT_co]: ...
    @overload  # spmatrix[-Bool], other: array-like +Bool
    def __mul__(self: spmatrix, other: _DualArrayLike[bool, _ToBool], /) -> onp.Array2D[_SCT_co]: ...
    @overload  # spmatrix[-Int], other: sparse +Int
    def __mul__(self: spmatrix[_FromInt], other: _spbase[_ToInt8], /) -> _SpMatrixOut[_SCT_co]: ...
    @overload  # spmatrix[-Int], other: array-like +Int
    def __mul__(self: spmatrix[_FromInt], other: _DualArrayLike[bool, _ToInt8], /) -> onp.Array2D[_SCT_co]: ...
    @overload  # spmatrix[-Float], other: sparse +Float
    def __mul__(self: spmatrix[_FromFloat], other: _spbase[_ToFloat32 | _SCT_co], /) -> _SpMatrixOut[_SCT_co]: ...
    @overload  # spmatrix[-Float], other: array-like +Float
    def __mul__(self: spmatrix[_FromFloat], other: _DualArrayLike[int | bool, _ToFloat32], /) -> onp.Array2D[_SCT_co]: ...
    @overload  # spmatrix[-Complex], other: sparse +Complex
    def __mul__(self: spmatrix[_FromComplex], other: _spbase[_ToComplex64 | _SCT_co], /) -> _SpMatrixOut[_SCT_co]: ...
    @overload  # spmatrix[-Complex], other: array-like +Complex
    def __mul__(
        self: spmatrix[_FromComplex], other: _DualArrayLike[float | int | bool, _ToComplex64], /
    ) -> onp.Array2D[_SCT_co]: ...
    @overload  # spmatrix[+Bool], other: scalar- or matrix-like ~Int
    def __mul__(self: spmatrix[_ToBool], other: _DualMatrixLike[op.JustInt, Integer], /) -> spmatrix[Integer]: ...
    @overload  # spmatrix[+Bool], other: array-like ~Int
    def __mul__(self: spmatrix[_ToBool], other: _DualArrayLike[op.JustInt, Integer], /) -> onp.Array2D[Integer]: ...
    @overload  # spmatrix[+Int], other: scalar- or matrix-like ~Float
    def __mul__(self: spmatrix[_ToInt], other: _DualMatrixLike[op.JustFloat, Floating], /) -> spmatrix[Floating]: ...
    @overload  # spmatrix[+Int], other: array-like ~Float
    def __mul__(self: spmatrix[_ToInt], other: _DualArrayLike[op.JustFloat, Floating], /) -> onp.Array2D[Floating]: ...
    @overload  # spmatrix[+Float], other: scalar- or matrix-like ~Complex
    def __mul__(self: spmatrix[_ToFloat], other: _DualMatrixLike[op.JustComplex, CFloating], /) -> spmatrix[CFloating]: ...
    @overload  # spmatrix[+Float], other: array-like ~Complex
    def __mul__(self: spmatrix[_ToFloat], other: _DualArrayLike[op.JustComplex, CFloating], /) -> onp.Array2D[CFloating]: ...
    @overload  # catch-all
    def __mul__(
        self, other: _DualArrayLike[complex | float | int | bool, Numeric] | _spbase, /
    ) -> _spbase[Any, Any] | onp.Array[Any, Any]: ...
    __rmul__ = __mul__

    #
    def __pow__(self, rhs: op.CanIndex, /) -> Self: ...

    #
    def getmaxprint(self, /) -> int | bool: ...
    def getformat(self, /) -> SPFormat: ...
    # NOTE: `axis` is only supported by `{coo,csc,csr,lil}_matrix`
    def getnnz(self, /, axis: None = None) -> int | bool: ...
    def getH(self, /) -> Self: ...
    def getcol(self, /, j: onp.ToJustInt) -> csc_matrix[_SCT_co]: ...
    def getrow(self, /, i: onp.ToJustInt) -> csr_matrix[_SCT_co]: ...

    # NOTE: mypy reports a false positive for overlapping overloads
    @overload
    def asfptype(self: spmatrix[np.bool_ | np.int8 | np.int16 | np.uint8 | np.uint16], /) -> spmatrix[np.float32]: ...
    @overload
    def asfptype(self: spmatrix[np.int32 | np.int64 | np.uint32 | np.uint64], /) -> spmatrix[np.float64]: ...
    @overload
    def asfptype(self, /) -> Self: ...

    #
    @overload
    def todense(self, /, order: OrderCF | None = None, out: None = None) -> onp.Matrix[_SCT_co]: ...
    @overload
    def todense(self, /, order: OrderCF | None, out: onp.ArrayND[_SCT]) -> onp.Matrix[_SCT]: ...
    @overload
    def todense(self, /, order: OrderCF | None = None, *, out: onp.ArrayND[_SCT]) -> onp.Matrix[_SCT]: ...
