from _typeshed import Incomplete
from collections.abc import Callable, Sequence
from typing import Any, Protocol, overload, type_check_only

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc
from numpy._typing import _ArrayLike, _DTypeLike

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

###
type __Func1[T] = Callable[[onp.ToComplex | onp.ToComplexND], T]
type __Func2[T] = Callable[[onp.ToComplex | onp.ToComplexND, onp.ToComplex | onp.ToComplexND], T]
type _ComprehensionFunc[T] = __Func1[T] | __Func2[T]

type _Idx0D = tuple[np.intp, ...]
type _IdxND = list[_Idx0D]

type _Extrema0D[ScalarT: npc.number | np.bool] = tuple[ScalarT, ScalarT, _Idx0D, _Idx0D]
type _ExtremaND[ScalarT: npc.number | np.bool] = tuple[onp.ArrayND[ScalarT], onp.ArrayND[ScalarT], _IdxND, _IdxND]

type _Coord0D = tuple[np.float64, ...]
type _Coord1D = list[_Coord0D]
type _CoordND = list[tuple[onp.ArrayND[np.float64], ...]]

type _AsComplex128OrND = onp.ToJustComplex128 | onp.ToJustComplex128_ND | onp.ToJustComplex64 | onp.ToJustComplex64_ND
type _AsUInt64OrND = npc.unsignedinteger | onp.ToArrayND[npc.unsignedinteger, npc.unsignedinteger]
type _AsInt64OrND = int | npc.signedinteger | np.bool | onp.ToArrayND[int, npc.signedinteger | np.bool]

###

#
@overload
def label(
    input: onp.ToComplex | onp.ToComplexND,
    structure: onp.ToComplex | onp.ToComplexND | None = None,
    *,
    output: onp.ArrayND[np.int32 | np.intp],
) -> int: ...
@overload
def label(
    input: onp.ToComplex | onp.ToComplexND, structure: onp.ToComplex | onp.ToComplexND | None = None, output: None = None
) -> tuple[onp.ArrayND[np.int32 | np.intp], int]: ...

#
@overload
def find_objects(input: onp.ToInt, max_label: int = 0) -> list[tuple[()] | None]: ...
@overload
def find_objects(input: onp.ToIntND, max_label: int = 0) -> list[tuple[slice[int, int, None], ...] | None]: ...

#
def value_indices(
    arr: onp.ToInt | onp.ToIntND, *, ignore_value: int | None = None
) -> dict[np.intp, tuple[onp.ArrayND[np.intp], ...]]: ...

#
@overload
def labeled_comprehension[T](
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToComplex | onp.ToComplexND | None,
    index: None,
    func: _ComprehensionFunc[T],
    out_dtype: object,
    default: object,
    pass_positions: bool = False,
) -> T: ...
@overload
def labeled_comprehension[ScalarT: npc.number | np.bool](
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToComplex | onp.ToComplexND,
    index: onp.ToInt,
    func: _ComprehensionFunc[onp.ToComplex],
    out_dtype: _DTypeLike[ScalarT],
    default: onp.ToFloat,
    pass_positions: bool = False,
) -> ScalarT: ...
@overload
def labeled_comprehension[ScalarT: npc.number | np.bool](
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToComplex | onp.ToComplexND,
    index: onp.ToIntND,
    func: _ComprehensionFunc[onp.ToComplex],
    out_dtype: _DTypeLike[ScalarT],
    default: onp.ToFloat,
    pass_positions: bool = False,
) -> onp.ArrayND[ScalarT]: ...
@overload
def labeled_comprehension(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToComplex | onp.ToComplexND,
    index: onp.ToInt,
    func: _ComprehensionFunc[onp.ToComplex],
    out_dtype: onp.AnyIntPDType,
    default: onp.ToInt,
    pass_positions: bool = False,
) -> np.intp: ...
@overload
def labeled_comprehension(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToComplex | onp.ToComplexND,
    index: onp.ToIntND,
    func: _ComprehensionFunc[onp.ToComplex],
    out_dtype: onp.AnyIntPDType,
    default: onp.ToInt,
    pass_positions: bool = False,
) -> onp.ArrayND[np.intp]: ...
@overload
def labeled_comprehension(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToComplex | onp.ToComplexND,
    index: onp.ToInt,
    func: _ComprehensionFunc[onp.ToComplex],
    out_dtype: onp.AnyFloat64DType | None,
    default: onp.ToFloat,
    pass_positions: bool = False,
) -> np.float64: ...
@overload
def labeled_comprehension(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToComplex | onp.ToComplexND,
    index: onp.ToIntND,
    func: _ComprehensionFunc[onp.ToComplex],
    out_dtype: onp.AnyFloat64DType | None,
    default: onp.ToFloat,
    pass_positions: bool = False,
) -> onp.ArrayND[np.float64]: ...
@overload
def labeled_comprehension(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToComplex | onp.ToComplexND,
    index: onp.ToInt,
    func: _ComprehensionFunc[onp.ToComplex],
    out_dtype: onp.AnyComplex128DType,
    default: onp.ToComplex,
    pass_positions: bool = False,
) -> np.complex128: ...
@overload
def labeled_comprehension(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToComplex | onp.ToComplexND,
    index: onp.ToIntND,
    func: _ComprehensionFunc[onp.ToComplex],
    out_dtype: onp.AnyComplex128DType,
    default: onp.ToComplex,
    pass_positions: bool = False,
) -> onp.ArrayND[np.complex128]: ...

#
@overload
def sum_labels[ScalarT: npc.inexact](
    input: onp.ToArrayND[ScalarT, ScalarT], labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> ScalarT: ...
@overload
def sum_labels[ScalarT: npc.inexact](
    input: onp.ToArrayND[ScalarT, ScalarT], labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> ScalarT: ...
@overload
def sum_labels(input: _AsUInt64OrND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None) -> np.uint64: ...
@overload
def sum_labels(input: _AsUInt64OrND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None) -> np.uint64: ...
@overload
def sum_labels(input: _AsInt64OrND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None) -> np.int64: ...
@overload
def sum_labels(input: _AsInt64OrND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None) -> np.int64: ...
@overload
def sum_labels(
    input: onp.ToFloat64 | onp.ToFloat64_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND
) -> onp.ArrayND[np.float64]: ...
@overload
def sum_labels(
    input: onp.ToJustFloat64 | onp.ToJustFloat64_ND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> np.float64: ...
@overload
def sum_labels(
    input: onp.ToJustFloat64 | onp.ToJustFloat64_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> np.float64: ...
@overload
def sum_labels(
    input: onp.ToJustComplex128 | onp.ToJustComplex128_ND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> np.complex128: ...
@overload
def sum_labels(
    input: onp.ToJustComplex128 | onp.ToJustComplex128_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> np.complex128: ...
@overload
def sum_labels(
    input: onp.ToComplex | onp.ToComplexND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> Any: ...
@overload
def sum_labels(
    input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> Any: ...
@overload
def sum_labels(
    input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND
) -> onp.ArrayND[Any]: ...

sum = sum_labels

#
@overload
def median[ScalarT: npc.inexact](
    input: onp.ToArrayND[ScalarT, ScalarT], labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> ScalarT: ...
@overload
def median[ScalarT: npc.inexact](
    input: onp.ToArrayND[ScalarT, ScalarT], labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> ScalarT: ...
@overload
def median[ScalarT: npc.inexact](
    input: onp.ToArrayND[ScalarT, ScalarT], labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND
) -> onp.ArrayND[ScalarT]: ...
@overload
def median(input: onp.ToInt | onp.ToIntND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None) -> np.float64: ...
@overload
def median(input: onp.ToInt | onp.ToIntND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None) -> np.float64: ...
@overload
def median(input: onp.ToInt | onp.ToIntND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND) -> onp.ArrayND[np.float64]: ...
@overload
def median(
    input: onp.ToJustFloat64 | onp.ToJustFloat64_ND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> np.float64: ...
@overload
def median(
    input: onp.ToJustFloat64 | onp.ToJustFloat64_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> np.float64: ...
@overload
def median(
    input: onp.ToJustFloat64 | onp.ToJustFloat64_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND
) -> onp.ArrayND[np.float64]: ...
@overload
def median(
    input: onp.ToJustComplex128 | onp.ToJustComplex128_ND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> np.complex128: ...
@overload
def median(
    input: onp.ToJustComplex128 | onp.ToJustComplex128_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> np.complex128: ...
@overload
def median(
    input: onp.ToJustComplex128 | onp.ToJustComplex128_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND
) -> onp.ArrayND[np.complex128]: ...
@overload
def median(input: onp.ToComplex | onp.ToComplexND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None) -> Any: ...
@overload
def median(input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None) -> Any: ...
@overload
def median(input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND) -> onp.ArrayND[Any]: ...

# keep in sync with `variance` and `standard_deviation`
@overload
def mean[ScalarT: npc.inexact80](
    input: onp.ToArrayND[ScalarT, ScalarT], labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> ScalarT: ...
@overload
def mean[ScalarT: npc.inexact80](
    input: onp.ToArrayND[ScalarT, ScalarT], labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> ScalarT: ...
@overload
def mean(
    input: onp.ToFloat64 | onp.ToFloat64_ND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> np.float64: ...
@overload
def mean(
    input: onp.ToFloat64 | onp.ToFloat64_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> np.float64: ...
@overload
def mean(
    input: onp.ToFloat64 | onp.ToFloat64_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND
) -> onp.ArrayND[np.float64]: ...
@overload
def mean(input: _AsComplex128OrND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None) -> np.complex128: ...
@overload
def mean(input: _AsComplex128OrND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None) -> np.complex128: ...
@overload
def mean(input: onp.ToComplex | onp.ToComplexND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None) -> Any: ...
@overload
def mean(input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None) -> Any: ...
@overload
def mean(input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND) -> onp.ArrayND[Any]: ...

# keep in sync with `mean` and `standard_deviation`
@overload
def variance[ScalarT: npc.inexact80](
    input: onp.ToArrayND[ScalarT, ScalarT], labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> ScalarT: ...
@overload
def variance[ScalarT: npc.inexact80](
    input: onp.ToArrayND[ScalarT, ScalarT], labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> ScalarT: ...
@overload
def variance(
    input: onp.ToFloat64 | onp.ToFloat64_ND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> np.float64: ...
@overload
def variance(
    input: onp.ToFloat64 | onp.ToFloat64_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> np.float64: ...
@overload
def variance(
    input: onp.ToFloat64 | onp.ToFloat64_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND
) -> onp.ArrayND[np.float64]: ...
@overload
def variance(input: _AsComplex128OrND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None) -> np.complex128: ...
@overload
def variance(input: _AsComplex128OrND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None) -> np.complex128: ...
@overload
def variance(
    input: onp.ToComplex | onp.ToComplexND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> Any: ...
@overload
def variance(input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None) -> Any: ...
@overload
def variance(input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND) -> onp.ArrayND[Any]: ...

# keep in sync with `mean` and `variance`
@overload
def standard_deviation[ScalarT: npc.inexact80](
    input: onp.ToArrayND[ScalarT, ScalarT], labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> ScalarT: ...
@overload
def standard_deviation[ScalarT: npc.inexact80](
    input: onp.ToArrayND[ScalarT, ScalarT], labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> ScalarT: ...
@overload
def standard_deviation(
    input: onp.ToFloat64 | onp.ToFloat64_ND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> np.float64: ...
@overload
def standard_deviation(
    input: onp.ToFloat64 | onp.ToFloat64_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> np.float64: ...
@overload
def standard_deviation(
    input: onp.ToFloat64 | onp.ToFloat64_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND
) -> onp.ArrayND[np.float64]: ...
@overload
def standard_deviation(
    input: _AsComplex128OrND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> np.complex128: ...
@overload
def standard_deviation(
    input: _AsComplex128OrND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> np.complex128: ...
@overload
def standard_deviation(
    input: onp.ToComplex | onp.ToComplexND, labels: None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> Any: ...
@overload
def standard_deviation(
    input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: onp.ToInt | None = None
) -> Any: ...
@overload
def standard_deviation(
    input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND
) -> onp.ArrayND[Any]: ...

#
@type_check_only
class _DefExtreme(Protocol):
    @overload
    def __call__[ScalarT: npc.number | np.bool](
        self, /, input: onp.CanArrayND[ScalarT], labels: onp.ToInt | onp.ToIntND | None = None, index: None = None
    ) -> ScalarT: ...
    @overload
    def __call__[ScalarT: npc.number | np.bool](
        self, /, input: onp.CanArrayND[ScalarT], labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND
    ) -> onp.ArrayND[ScalarT]: ...
    @overload
    def __call__[ScalarT: npc.number | np.bool](
        self, /, input: onp.CanArrayND[ScalarT], labels: onp.ToInt | onp.ToIntND | None = None, *, index: onp.ToIntND
    ) -> onp.ArrayND[ScalarT]: ...
    @overload
    def __call__(
        self, /, input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: onp.ArrayND[npc.integer]
    ) -> onp.ArrayND[Any]: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToComplex | onp.ToComplexND,
        labels: onp.ToInt | onp.ToIntND | None = None,
        *,
        index: onp.ArrayND[npc.integer],
    ) -> onp.ArrayND[Any]: ...
    @overload
    def __call__[ScalarT: npc.number | np.bool](
        self,
        /,
        input: onp.CanArrayND[ScalarT],
        labels: onp.ToInt | onp.ToIntND | None = None,
        index: onp.ToInt | onp.ToIntND | None = None,
    ) -> ScalarT | onp.ArrayND[ScalarT]: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToComplex | onp.ToComplexND,
        labels: onp.ToInt | onp.ToIntND | None = None,
        index: onp.ToInt | onp.ToIntND | None = None,
    ) -> Incomplete: ...

minimum: _DefExtreme
maximum: _DefExtreme

#
@overload
def extrema(
    input: onp.ToJustFloat64 | onp.ToJustFloat64_ND, labels: onp.ToInt | onp.ToIntND | None = None, index: onp.ToInt | None = None
) -> _Extrema0D[np.float64]: ...
@overload
def extrema(
    input: onp.ToJustComplex128 | onp.ToJustComplex128_ND,
    labels: onp.ToInt | onp.ToIntND | None = None,
    index: onp.ToInt | None = None,
) -> _Extrema0D[np.complex128]: ...
@overload
def extrema[ScalarT: npc.number | np.bool](
    input: _ArrayLike[ScalarT], labels: onp.ToInt | onp.ToIntND | None = None, index: onp.ToInt | None = None
) -> _Extrema0D[ScalarT]: ...
@overload
def extrema(
    input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND | None = None, index: onp.ToInt | None = None
) -> _Extrema0D[Any]: ...
@overload
def extrema(
    input: onp.ToJustFloat64 | onp.ToJustFloat64_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND
) -> _ExtremaND[np.float64]: ...
@overload
def extrema(
    input: onp.ToJustFloat64 | onp.ToJustFloat64_ND, labels: onp.ToInt | onp.ToIntND | None = None, *, index: onp.ToIntND
) -> _ExtremaND[np.float64]: ...
@overload
def extrema(
    input: onp.ToJustComplex128 | onp.ToJustComplex128_ND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND
) -> _ExtremaND[np.complex128]: ...
@overload
def extrema(
    input: onp.ToJustComplex128 | onp.ToJustComplex128_ND, labels: onp.ToInt | onp.ToIntND | None = None, *, index: onp.ToIntND
) -> _ExtremaND[np.complex128]: ...
@overload
def extrema[ScalarT: npc.number | np.bool](
    input: _ArrayLike[ScalarT], labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND
) -> _ExtremaND[ScalarT]: ...
@overload
def extrema[ScalarT: npc.number | np.bool](
    input: _ArrayLike[ScalarT], labels: onp.ToInt | onp.ToIntND | None = None, *, index: onp.ToIntND
) -> _ExtremaND[ScalarT]: ...
@overload
def extrema(input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: onp.ToIntND) -> _ExtremaND[Any]: ...
@overload
def extrema(
    input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND | None = None, *, index: onp.ToIntND
) -> _ExtremaND[Any]: ...
@overload
def extrema[ScalarT: npc.number | np.bool](
    input: _ArrayLike[ScalarT], labels: onp.ToInt | onp.ToIntND | None = None, index: onp.ToInt | onp.ToIntND | None = None
) -> _Extrema0D[ScalarT] | _ExtremaND[ScalarT]: ...
@overload
def extrema(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToInt | onp.ToIntND | None = None,
    index: onp.ToInt | onp.ToIntND | None = None,
) -> _Extrema0D[Any] | _ExtremaND[Any]: ...

#
@type_check_only
class _DefArgExtreme(Protocol):
    @overload
    def __call__(
        self, /, input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND | None = None, index: None = None
    ) -> _Idx0D: ...
    @overload
    def __call__(
        self, /, input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: onp.ArrayND[npc.integer]
    ) -> _IdxND: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToComplex | onp.ToComplexND,
        labels: onp.ToInt | onp.ToIntND | None = None,
        *,
        index: onp.ArrayND[npc.integer],
    ) -> _IdxND: ...
    @overload
    def __call__(
        self,
        /,
        input: onp.ToComplex | onp.ToComplexND,
        labels: onp.ToInt | onp.ToIntND | None = None,
        index: onp.ToInt | onp.ToIntND | None = None,
    ) -> _Idx0D | _IdxND: ...

minimum_position: _DefArgExtreme
maximum_position: _DefArgExtreme

#
@overload
def center_of_mass(
    input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND | None = None, index: onp.ToInt | None = None
) -> _Coord0D: ...
@overload
def center_of_mass(
    input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: Sequence[onp.ToInt]
) -> _Coord1D: ...
@overload
def center_of_mass(
    input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND | None = None, *, index: Sequence[onp.ToInt]
) -> _Coord1D: ...
@overload
def center_of_mass(
    input: onp.ToComplex | onp.ToComplexND, labels: onp.ToInt | onp.ToIntND, index: Sequence[Sequence[onp.ToInt | onp.ToIntND]]
) -> _CoordND: ...
@overload
def center_of_mass(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToInt | onp.ToIntND | None = None,
    *,
    index: Sequence[Sequence[onp.ToInt | onp.ToIntND]],
) -> _CoordND: ...
@overload
def center_of_mass(
    input: onp.ToComplex | onp.ToComplexND,
    labels: onp.ToInt | onp.ToIntND | None = None,
    index: onp.ToInt | onp.ToIntND | None = None,
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
    index: onp.ToIntND,
) -> onp.ArrayND[np.object_]: ...
@overload
def histogram(
    input: onp.ToComplex | onp.ToComplexND,
    min: onp.ToInt,
    max: onp.ToInt,
    bins: onp.ToInt,
    labels: onp.ToInt | onp.ToIntND | None = None,
    *,
    index: onp.ToIntND,
) -> onp.ArrayND[np.object_]: ...
@overload
def histogram(
    input: onp.ToComplex | onp.ToComplexND,
    min: onp.ToInt,
    max: onp.ToInt,
    bins: onp.ToInt,
    labels: onp.ToInt | onp.ToIntND | None,
    index: onp.ToInt | onp.ToIntND | None,
) -> onp.ArrayND[Any]: ...

#
def watershed_ift(
    input: _ArrayLike[np.uint8 | np.uint16],
    markers: _ArrayLike[npc.signedinteger] | onp.SequenceND[int],
    structure: onp.ToInt | onp.ToIntND | None = None,
    output: onp.ArrayND[npc.signedinteger] | None = None,
) -> onp.ArrayND[npc.signedinteger]: ...
