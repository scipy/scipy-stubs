import abc
from typing import Any, Generic, Literal, Self
from typing_extensions import TypeVar, override

import optype.numpy as onp

from ._data import _data_matrix, _minmax_mixin
from ._index import IndexMixin
from ._typing import Index1D, Numeric

__all__: list[str] = []

_ScalarT = TypeVar("_ScalarT", bound=Numeric, default=Any)
_ShapeT_co = TypeVar("_ShapeT_co", bound=onp.AtLeast1D, default=onp.AtLeast0D[Any], covariant=True)

###

class _cs_matrix(
    _data_matrix[_ScalarT, _ShapeT_co],
    _minmax_mixin[_ScalarT, _ShapeT_co],
    IndexMixin[_ScalarT, _ShapeT_co],
    Generic[_ScalarT, _ShapeT_co],
):
    data: onp.ArrayND[_ScalarT]
    indices: Index1D
    indptr: Index1D

    @property
    @override
    @abc.abstractmethod
    def format(self, /) -> Literal["bsr", "csc", "csr"]: ...

    #
    @property
    def has_canonical_format(self, /) -> bool: ...
    @has_canonical_format.setter
    def has_canonical_format(self, val: bool, /) -> None: ...

    #
    @property
    def has_sorted_indices(self, /) -> bool: ...
    @has_sorted_indices.setter
    def has_sorted_indices(self, val: bool, /) -> None: ...

    #
    def sorted_indices(self, /) -> Self: ...
    def sort_indices(self, /) -> None: ...

    #
    def check_format(self, /, full_check: bool = True) -> None: ...
    def eliminate_zeros(self, /) -> None: ...
    def sum_duplicates(self, /) -> None: ...
    def prune(self, /) -> None: ...
