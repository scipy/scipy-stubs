from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from numpy._typing import _ArrayLike, _DTypeLike
from ._typing import _ScalarArrayOut, _ScalarValueOut

__all__ = [
    "center_of_mass",
    "extrema",
    "find_objects",
    "histogram",
    "label",
    "labeled_comprehension",
    "maximum",
    "maximum_position",
    "mean",
    "median",
    "minimum",
    "minimum_position",
    "standard_deviation",
    "sum",
    "sum_labels",
    "value_indices",
    "variance",
    "watershed_ift",
]

_SCT = TypeVar("_SCT", bound=_ScalarValueOut, default=_ScalarValueOut)
_ISCT = TypeVar("_ISCT", bound=np.inexact[Any], default=np.inexact[Any])

__Func1: TypeAlias = Callable[[onp.ToComplex | onp.ToComplexND], onp.ToComplex]
__Func2: TypeAlias = Callable[[onp.ToComplex | onp.ToComplexND, onp.ToComplex | onp.ToComplexND], onp.ToComplex]
_ComprehensionFunc: TypeAlias = __Func1 | __Func2

_Idx0D: TypeAlias = tuple[np.intp, ...]
_IdxND: TypeAlias = list[_Idx0D]

_Extrema0D: TypeAlias = tuple[_SCT, _SCT, _Idx0D, _Idx0D]
_ExtremaND: TypeAlias = tuple[onp.ArrayND[_SCT], onp.ArrayND[_SCT], _IdxND, _IdxND]

_Coord0D: TypeAlias = tuple[np.float64, ...]
_Coord1D: TypeAlias = list[_Coord0D]
_CoordND: TypeAlias = list[tuple[onp.ArrayND[np.float64], ...]]

#
def label(
    input: onp.ToComplex | onp.ToComplexND,
    structure: onp.ToComplex | onp.ToComplexND | None = None,
    output: onp.ArrayND[np.int32 | np.intp] | None = None,
) -> int | bool | tuple[onp.ArrayND[np.int32 | np.intp], int | bool]: ...

#
def find_objects(input: onp.ToInt | onp.ToIntND, max_label: int | bool = 0) -> list[tuple[slice, ...]]: ...

#
def value_indices(
    arr: onp.ToInt | onp.ToIntND,
    *,
    ignore_value: int | bool | None = None,
) -> dict[np.intp, tuple[onp.ArrayND[np.intp], ...]]: ...

#
@overload
def labeled_comprehension(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToComplex | onp.ToComplexND | None,
    index: onp.ToInt | onp.ToIntND | None,
    func: _ComprehensionFunc,
    out_dtype: _DTypeLike[_SCT],
    default: onp.ToFloat,
    pass_positions: bool = False,
) -> onp.ArrayND[_SCT]: ...
@overload
def labeled_comprehension(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToComplex | onp.ToComplexND | None,
    index: onp.ToInt | onp.ToIntND | None,
    func: _ComprehensionFunc,
    out_dtype: type[int | bool],
    default: onp.ToInt,
    pass_positions: bool = False,
) -> onp.ArrayND[np.intp]: ...
@overload
def labeled_comprehension(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToComplex | onp.ToComplexND | None,
    index: onp.ToInt | onp.ToIntND | None,
    func: _ComprehensionFunc,
    out_dtype: type[float | int | bool],
    default: onp.ToFloat,
    pass_positions: bool = False,
) -> onp.ArrayND[np.float64 | np.intp]: ...
@overload
def labeled_comprehension(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToComplex | onp.ToComplexND | None,
    index: onp.ToInt | onp.ToIntND | None,
    func: _ComprehensionFunc,
    out_dtype: type[complex | float | int | bool],
    default: onp.ToComplex,
    pass_positions: bool = False,
) -> onp.ArrayND[np.complex128 | np.float64 | np.intp]: ...

#
@type_check_only
class _DefStatistic(Protocol):
    @overload
    def __call__(
        self,
        /,
        input: onp.CanArrayND[_ISCT],
        labels: onp.ToInt | onp.ToIntND | None = None,
        index: None = None,
    ) -> _ISCT: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.CanArrayND[_ISCT],
        labels: onp.ToInt | onp.ToIntND,
        index: onp.ArrayND[np.integer[Any]],
    ) -> onp.ArrayND[_ISCT]: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.CanArrayND[_ISCT],
        labels: onp.ToInt | onp.ToIntND,
        index: onp.ToInt | onp.ToIntND,
    ) -> onp.ArrayND[_ISCT]: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToInt | onp.ToIntND,
        labels: onp.ToInt | onp.ToIntND | None = None,
        index: None = None,
    ) -> np.float64: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToInt | onp.ToIntND,
        labels: onp.ToInt | onp.ToIntND,
        index: onp.ArrayND[np.integer[Any]],
    ) -> onp.ArrayND[np.float64]: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToComplex | onp.ToComplexND,
        labels: onp.ToInt | onp.ToIntND | None = None,
        index: None = None,
    ) -> np.inexact[Any]: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToComplex | onp.ToComplexND,
        labels: onp.ToInt | onp.ToIntND,
        index: onp.ArrayND[np.integer[Any]],
    ) -> onp.ArrayND[np.inexact[Any]]: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToComplex | onp.ToComplexND,
        labels: onp.ToInt | onp.ToIntND,
        index: onp.ToInt | onp.ToIntND,
    ) -> np.inexact[Any] | onp.ArrayND[np.inexact[Any]]: ...

sum: _DefStatistic
sum_labels: _DefStatistic
mean: _DefStatistic
variance: _DefStatistic
standard_deviation: _DefStatistic
median: _DefStatistic

#
@type_check_only
class _DefExtreme(Protocol):
    @overload
    def __call__(
        self,
        /,
        input: onp.CanArrayND[_SCT],
        labels: onp.ToInt | onp.ToIntND | None = None,
        index: None = None,
    ) -> _SCT: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.CanArrayND[_SCT],
        labels: onp.ToInt | onp.ToIntND,
        index: onp.ToIntND,
    ) -> onp.ArrayND[_SCT]: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToComplex | onp.ToComplexND,
        labels: onp.ToInt | onp.ToIntND | None = None,
        index: None = None,
    ) -> _ScalarValueOut: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToComplex | onp.ToComplexND,
        labels: onp.ToInt | onp.ToIntND,
        index: onp.ArrayND[np.integer[Any]],
    ) -> _ScalarArrayOut: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.CanArrayND[_SCT],
        labels: onp.ToInt | onp.ToIntND,
        index: onp.ToInt | onp.ToIntND,
    ) -> _SCT | onp.ArrayND[_SCT]: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToComplex | onp.ToComplexND,
        labels: onp.ToInt | onp.ToIntND,
        index: onp.ToInt | onp.ToIntND,
    ) -> _ScalarValueOut | _ScalarArrayOut: ...

minimum: _DefExtreme
maximum: _DefExtreme

#
@overload
def extrema(
    input: _ArrayLike[_SCT],
    labels: onp.ToInt | onp.ToIntND | None = None,
    index: onp.ToInt | None = None,
) -> _Extrema0D[_SCT]: ...
@overload
def extrema(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToInt | onp.ToIntND | None = None,
    index: onp.ToInt | None = None,
) -> _Extrema0D: ...
@overload
def extrema(
    input: _ArrayLike[_SCT],
    labels: onp.ToInt | onp.ToIntND,
    index: onp.ArrayND[np.integer[Any]],
) -> _ExtremaND[_SCT]: ...
@overload
def extrema(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToInt | onp.ToIntND,
    index: onp.ArrayND[np.integer[Any]],
) -> _ExtremaND: ...
@overload
def extrema(
    input: _ArrayLike[_SCT],
    labels: onp.ToInt | onp.ToIntND,
    index: onp.ToInt | onp.ToIntND,
) -> _Extrema0D[_SCT] | _ExtremaND[_SCT]: ...
@overload
def extrema(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToInt | onp.ToIntND,
    index: onp.ToInt | onp.ToIntND,
) -> _Extrema0D | _ExtremaND: ...

#
@type_check_only
class _DefArgExtreme(Protocol):
    @overload
    def __call__(
        self,
        /,
        input: onp.ToComplex | onp.ToComplexND,
        labels: onp.ToInt | onp.ToIntND | None = None,
        index: None = None,
    ) -> _Idx0D: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToComplex | onp.ToComplexND,
        labels: onp.ToInt | onp.ToIntND,
        index: onp.ArrayND[np.integer[Any]],
    ) -> _IdxND: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToComplex | onp.ToComplexND,
        labels: onp.ToInt | onp.ToIntND,
        index: onp.ToInt | onp.ToIntND,
    ) -> _Idx0D | _IdxND: ...

minimum_position: _DefArgExtreme
maximum_position: _DefArgExtreme

#
@overload
def center_of_mass(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToInt | onp.ToIntND | None = None,
    index: onp.ToInt | None = None,
) -> _Coord0D: ...
@overload
def center_of_mass(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToInt | onp.ToIntND,
    index: Sequence[onp.ToInt],
) -> _Coord1D: ...
@overload
def center_of_mass(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToInt | onp.ToIntND,
    index: Sequence[Sequence[onp.ToInt | onp.ToIntND]],
) -> _CoordND: ...
@overload
def center_of_mass(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToInt | onp.ToIntND,
    index: onp.ToInt | onp.ToIntND,
) -> _Coord0D | _Coord1D | _CoordND: ...

#
@overload
def histogram(
    input: onp.ToComplex | onp.ToComplexND,
    min: onp.ToInt,
    max: onp.ToInt,
    bins: onp.ToInt,
    labels: onp.ToInt | onp.ToIntND | None = None,
    index: onp.ToInt | None = None,
) -> onp.ArrayND[np.intp]: ...
@overload
def histogram(
    input: onp.ToComplex | onp.ToComplexND,
    min: onp.ToInt,
    max: onp.ToInt,
    bins: onp.ToInt,
    labels: onp.ToInt | onp.ToIntND,
    index: onp.ArrayND[np.integer[Any]],
) -> onp.ArrayND[np.object_]: ...
@overload
def histogram(
    input: onp.ToComplex | onp.ToComplexND,
    min: onp.ToInt,
    max: onp.ToInt,
    bins: onp.ToInt,
    labels: onp.ToInt | onp.ToIntND,
    index: onp.ToInt | onp.ToIntND,
) -> onp.ArrayND[np.intp | np.object_]: ...

#
def watershed_ift(
    input: _ArrayLike[np.uint8 | np.uint16],
    markers: _ArrayLike[np.signedinteger[Any]] | onp.SequenceND[int | bool],
    structure: onp.ToInt | onp.ToIntND | None = None,
    output: onp.ArrayND[np.signedinteger[Any]] | None = None,
) -> onp.ArrayND[np.signedinteger[Any]]: ...
