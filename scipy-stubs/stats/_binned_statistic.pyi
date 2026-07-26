from collections.abc import Callable, Sequence
from typing import Any, Generic, Literal, NamedTuple, Never, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["binned_statistic", "binned_statistic_2d", "binned_statistic_dd"]

###

type _StatisticF64 = Literal["count", "std"]  # always `float64`, even for complex `values`
type _StatisticPromoted = Literal["mean", "median", "sum", "min", "max"]  # `result_type(values, float64)`
type _Statistic = Literal[_StatisticF64, _StatisticPromoted]

type _StatFunc[ScalarT: npc.inexact, ToT] = Callable[[onp.Array1D[ScalarT]], ToT]

type _JustAnyShape = tuple[Never, Never, Never, Never]  # workaround for https://github.com/microsoft/pyright/issues/10232

type _AsF64 = npc.floating64 | npc.floating32 | npc.floating16 | npc.integer | np.bool
type _AsC128 = npc.complexfloating128 | npc.complexfloating64

type _ToBins = onp.ToInt | onp.ToFloat1D
type _ToRange = tuple[float, float] | Sequence[tuple[float, float]] | None

type _ToSample = onp.ToFloat1D | onp.ToFloat2D
type _ToBinsND = onp.ToInt | onp.ToFloat1D | Sequence[onp.ToFloat1D]
type _ToRangeND = Sequence[tuple[float, float]] | None

type _Shape1Or2 = tuple[int] | tuple[int, int]
type _Shape2Or3 = tuple[int, int] | tuple[int, int, int]

_InexactT_co = TypeVar("_InexactT_co", bound=npc.inexact, covariant=True, default=np.float64 | np.complex128)
_Shape1Or2T_co = TypeVar("_Shape1Or2T_co", bound=_Shape1Or2, covariant=True, default=_Shape1Or2)
_Shape2Or3T_co = TypeVar("_Shape2Or3T_co", bound=_Shape2Or3, covariant=True, default=_Shape2Or3)

###

class BinnedStatisticResult(NamedTuple, Generic[_InexactT_co, _Shape1Or2T_co]):
    statistic: onp.Array[_Shape1Or2T_co, _InexactT_co]
    bin_edges: onp.Array1D[np.float64]
    binnumber: onp.Array1D[np.intp]

class BinnedStatistic2dResult(NamedTuple, Generic[_InexactT_co, _Shape2Or3T_co, _Shape1Or2T_co]):
    statistic: onp.Array[_Shape2Or3T_co, _InexactT_co]
    x_edge: onp.Array1D[np.float64]
    y_edge: onp.Array1D[np.float64]
    binnumber: onp.Array[_Shape1Or2T_co, np.intp]

class BinnedStatisticddResult(NamedTuple, Generic[_InexactT_co, _Shape1Or2T_co]):
    statistic: onp.ArrayND[_InexactT_co]  # `(nx1, ..., nxD)`, or `(len(values), nx1, ..., nxD)` for 2-d `values`
    bin_edges: list[onp.Array1D[np.float64]]
    binnumber: onp.Array[_Shape1Or2T_co, np.intp]

#
@overload  # ?d, count|std
def binned_statistic(
    x: onp.ToFloat1D,
    values: onp.ArrayND[_AsF64 | _AsC128, _JustAnyShape],
    statistic: _StatisticF64,
    bins: _ToBins = 10,
    range: _ToRange = None,
) -> BinnedStatisticResult[np.float64]: ...
@overload  # ?d real
def binned_statistic(
    x: onp.ToFloat1D,
    values: onp.ArrayND[_AsF64, _JustAnyShape],
    statistic: _StatisticPromoted | _StatFunc[np.float64, onp.ToFloat] = "mean",
    bins: _ToBins = 10,
    range: _ToRange = None,
) -> BinnedStatisticResult[np.float64]: ...
@overload  # ?d ~complex
def binned_statistic(
    x: onp.ToFloat1D,
    values: onp.ArrayND[_AsC128, _JustAnyShape],
    statistic: _StatisticPromoted | _StatFunc[np.complex128, onp.ToComplex] = "mean",
    bins: _ToBins = 10,
    range: _ToRange = None,
) -> BinnedStatisticResult[np.complex128]: ...
@overload  # ?d real, ~complex statistic
def binned_statistic(
    x: onp.ToFloat1D,
    values: onp.ArrayND[_AsF64, _JustAnyShape],
    statistic: _StatFunc[np.float64, onp.ToJustComplex],
    bins: _ToBins = 10,
    range: _ToRange = None,
) -> BinnedStatisticResult[np.complex128]: ...
@overload  # 1d, count|std
def binned_statistic(
    x: onp.ToFloat1D,
    values: onp.ToArrayStrict1D[complex, _AsF64 | _AsC128],
    statistic: _StatisticF64,
    bins: _ToBins = 10,
    range: _ToRange = None,
) -> BinnedStatisticResult[np.float64, tuple[int]]: ...
@overload  # 1d real
def binned_statistic(
    x: onp.ToFloat1D,
    values: onp.ToArrayStrict1D[float, _AsF64],
    statistic: _StatisticPromoted | _StatFunc[np.float64, onp.ToFloat] = "mean",
    bins: _ToBins = 10,
    range: _ToRange = None,
) -> BinnedStatisticResult[np.float64, tuple[int]]: ...
@overload  # 1d ~complex
def binned_statistic(
    x: onp.ToFloat1D,
    values: onp.ToArrayStrict1D[op.JustComplex, _AsC128],
    statistic: _StatisticPromoted | _StatFunc[np.complex128, onp.ToComplex] = "mean",
    bins: _ToBins = 10,
    range: _ToRange = None,
) -> BinnedStatisticResult[np.complex128, tuple[int]]: ...
@overload  # 1d real, ~complex statistic
def binned_statistic(
    x: onp.ToFloat1D,
    values: onp.ToArrayStrict1D[float, _AsF64],
    statistic: _StatFunc[np.float64, onp.ToJustComplex],
    bins: _ToBins = 10,
    range: _ToRange = None,
) -> BinnedStatisticResult[np.complex128, tuple[int]]: ...
@overload  # 2d, count|std
def binned_statistic(
    x: onp.ToFloat1D,
    values: onp.ToArrayStrict2D[complex, _AsF64 | _AsC128],
    statistic: _StatisticF64,
    bins: _ToBins = 10,
    range: _ToRange = None,
) -> BinnedStatisticResult[np.float64, tuple[int, int]]: ...
@overload  # 2d real
def binned_statistic(
    x: onp.ToFloat1D,
    values: onp.ToArrayStrict2D[float, _AsF64],
    statistic: _StatisticPromoted | _StatFunc[np.float64, onp.ToFloat] = "mean",
    bins: _ToBins = 10,
    range: _ToRange = None,
) -> BinnedStatisticResult[np.float64, tuple[int, int]]: ...
@overload  # 2d ~complex
def binned_statistic(
    x: onp.ToFloat1D,
    values: onp.ToArrayStrict2D[op.JustComplex, _AsC128],
    statistic: _StatisticPromoted | _StatFunc[np.complex128, onp.ToComplex] = "mean",
    bins: _ToBins = 10,
    range: _ToRange = None,
) -> BinnedStatisticResult[np.complex128, tuple[int, int]]: ...
@overload  # 2d real, ~complex statistic
def binned_statistic(
    x: onp.ToFloat1D,
    values: onp.ToArrayStrict2D[float, _AsF64],
    statistic: _StatFunc[np.float64, onp.ToJustComplex],
    bins: _ToBins = 10,
    range: _ToRange = None,
) -> BinnedStatisticResult[np.complex128, tuple[int, int]]: ...
@overload  # values=None, count
def binned_statistic(
    x: onp.ToFloat1D, values: None, statistic: Literal["count"], bins: _ToBins = 10, range: _ToRange = None
) -> BinnedStatisticResult[np.float64, tuple[int]]: ...
@overload  # fallback
def binned_statistic(
    x: onp.ToFloat1D,
    values: onp.ToComplexND | None,
    statistic: _Statistic | _StatFunc[Any, onp.ToComplex] = "mean",
    bins: _ToBins = 10,
    range: _ToRange = None,
) -> BinnedStatisticResult: ...

#
@overload  # ?d, count|std
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ArrayND[_AsF64 | _AsC128, _JustAnyShape],
    statistic: _StatisticF64,
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.float64, _Shape2Or3, tuple[int]]: ...
@overload  # ?d, count|std, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ArrayND[_AsF64 | _AsC128, _JustAnyShape],
    statistic: _StatisticF64,
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.float64, _Shape2Or3, tuple[int, int]]: ...
@overload  # ?d real
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ArrayND[_AsF64, _JustAnyShape],
    statistic: _StatisticPromoted | _StatFunc[np.float64, onp.ToFloat] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.float64, _Shape2Or3, tuple[int]]: ...
@overload  # ?d real, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ArrayND[_AsF64, _JustAnyShape],
    statistic: _StatisticPromoted | _StatFunc[np.float64, onp.ToFloat] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.float64, _Shape2Or3, tuple[int, int]]: ...
@overload  # ?d ~complex
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ArrayND[_AsC128, _JustAnyShape],
    statistic: _StatisticPromoted | _StatFunc[np.complex128, onp.ToComplex] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.complex128, _Shape2Or3, tuple[int]]: ...
@overload  # ?d ~complex, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ArrayND[_AsC128, _JustAnyShape],
    statistic: _StatisticPromoted | _StatFunc[np.complex128, onp.ToComplex] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.complex128, _Shape2Or3, tuple[int, int]]: ...
@overload  # ?d real, ~complex statistic
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ArrayND[_AsF64, _JustAnyShape],
    statistic: _StatFunc[np.float64, onp.ToJustComplex],
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.complex128, _Shape2Or3, tuple[int]]: ...
@overload  # ?d real, ~complex statistic, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ArrayND[_AsF64, _JustAnyShape],
    statistic: _StatFunc[np.float64, onp.ToJustComplex],
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.complex128, _Shape2Or3, tuple[int, int]]: ...
@overload  # 1d, count|std
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict1D[complex, _AsF64 | _AsC128],
    statistic: _StatisticF64,
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.float64, tuple[int, int], tuple[int]]: ...
@overload  # 1d, count|std, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict1D[complex, _AsF64 | _AsC128],
    statistic: _StatisticF64,
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.float64, tuple[int, int], tuple[int, int]]: ...
@overload  # 1d real
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict1D[float, _AsF64],
    statistic: _StatisticPromoted | _StatFunc[np.float64, onp.ToFloat] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.float64, tuple[int, int], tuple[int]]: ...
@overload  # 1d real, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict1D[float, _AsF64],
    statistic: _StatisticPromoted | _StatFunc[np.float64, onp.ToFloat] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.float64, tuple[int, int], tuple[int, int]]: ...
@overload  # 1d ~complex
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict1D[op.JustComplex, _AsC128],
    statistic: _StatisticPromoted | _StatFunc[np.complex128, onp.ToComplex] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.complex128, tuple[int, int], tuple[int]]: ...
@overload  # 1d ~complex, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict1D[op.JustComplex, _AsC128],
    statistic: _StatisticPromoted | _StatFunc[np.complex128, onp.ToComplex] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.complex128, tuple[int, int], tuple[int, int]]: ...
@overload  # 1d real, ~complex statistic
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict1D[float, _AsF64],
    statistic: _StatFunc[np.float64, onp.ToJustComplex],
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.complex128, tuple[int, int], tuple[int]]: ...
@overload  # 1d real, ~complex statistic, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict1D[float, _AsF64],
    statistic: _StatFunc[np.float64, onp.ToJustComplex],
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.complex128, tuple[int, int], tuple[int, int]]: ...
@overload  # 2d, count|std
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict2D[complex, _AsF64 | _AsC128],
    statistic: _StatisticF64,
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.float64, tuple[int, int, int], tuple[int]]: ...
@overload  # 2d, count|std, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict2D[complex, _AsF64 | _AsC128],
    statistic: _StatisticF64,
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.float64, tuple[int, int, int], tuple[int, int]]: ...
@overload  # 2d real
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict2D[float, _AsF64],
    statistic: _StatisticPromoted | _StatFunc[np.float64, onp.ToFloat] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.float64, tuple[int, int, int], tuple[int]]: ...
@overload  # 2d real, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict2D[float, _AsF64],
    statistic: _StatisticPromoted | _StatFunc[np.float64, onp.ToFloat] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.float64, tuple[int, int, int], tuple[int, int]]: ...
@overload  # 2d ~complex
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict2D[op.JustComplex, _AsC128],
    statistic: _StatisticPromoted | _StatFunc[np.complex128, onp.ToComplex] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.complex128, tuple[int, int, int], tuple[int]]: ...
@overload  # 2d ~complex, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict2D[op.JustComplex, _AsC128],
    statistic: _StatisticPromoted | _StatFunc[np.complex128, onp.ToComplex] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.complex128, tuple[int, int, int], tuple[int, int]]: ...
@overload  # 2d real, ~complex statistic
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict2D[float, _AsF64],
    statistic: _StatFunc[np.float64, onp.ToJustComplex],
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.complex128, tuple[int, int, int], tuple[int]]: ...
@overload  # 2d real, ~complex statistic, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToArrayStrict2D[float, _AsF64],
    statistic: _StatFunc[np.float64, onp.ToJustComplex],
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.complex128, tuple[int, int, int], tuple[int, int]]: ...
@overload  # values=None, count
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: None,
    statistic: Literal["count"],
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.float64, tuple[int, int], tuple[int]]: ...
@overload  # values=None, count, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: None,
    statistic: Literal["count"],
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.float64, tuple[int, int], tuple[int, int]]: ...
@overload  # fallback
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToComplexND | None,
    statistic: _Statistic | _StatFunc[Any, onp.ToComplex] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[np.float64 | np.complex128, _Shape2Or3, tuple[int]]: ...
@overload  # fallback, expanded (keyword)
def binned_statistic_2d(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    values: onp.ToComplexND | None,
    statistic: _Statistic | _StatFunc[Any, onp.ToComplex] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[np.float64 | np.complex128, _Shape2Or3, tuple[int, int]]: ...

#
@overload  # count|std
def binned_statistic_dd(
    sample: _ToSample,
    values: onp.ToArrayND[complex, _AsF64 | _AsC128],
    statistic: _StatisticF64,
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
    binned_statistic_result: BinnedStatisticddResult[Any, Any] | None = None,
) -> BinnedStatisticddResult[np.float64, tuple[int]]: ...
@overload  # count|std, expanded (keyword)
def binned_statistic_dd(
    sample: _ToSample,
    values: onp.ToArrayND[complex, _AsF64 | _AsC128],
    statistic: _StatisticF64,
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
    binned_statistic_result: BinnedStatisticddResult[Any, Any] | None = None,
) -> BinnedStatisticddResult[np.float64, tuple[int, int]]: ...
@overload  # real
def binned_statistic_dd(
    sample: _ToSample,
    values: onp.ToArrayND[float, _AsF64],
    statistic: _StatisticPromoted | _StatFunc[np.float64, onp.ToFloat] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
    binned_statistic_result: BinnedStatisticddResult[Any, Any] | None = None,
) -> BinnedStatisticddResult[np.float64, tuple[int]]: ...
@overload  # real, expanded (keyword)
def binned_statistic_dd(
    sample: _ToSample,
    values: onp.ToArrayND[float, _AsF64],
    statistic: _StatisticPromoted | _StatFunc[np.float64, onp.ToFloat] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
    binned_statistic_result: BinnedStatisticddResult[Any, Any] | None = None,
) -> BinnedStatisticddResult[np.float64, tuple[int, int]]: ...
@overload  # ~complex
def binned_statistic_dd(
    sample: _ToSample,
    values: onp.ToArrayND[op.JustComplex, _AsC128],
    statistic: _StatisticPromoted | _StatFunc[np.complex128, onp.ToComplex] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
    binned_statistic_result: BinnedStatisticddResult[Any, Any] | None = None,
) -> BinnedStatisticddResult[np.complex128, tuple[int]]: ...
@overload  # ~complex, expanded (keyword)
def binned_statistic_dd(
    sample: _ToSample,
    values: onp.ToArrayND[op.JustComplex, _AsC128],
    statistic: _StatisticPromoted | _StatFunc[np.complex128, onp.ToComplex] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
    binned_statistic_result: BinnedStatisticddResult[Any, Any] | None = None,
) -> BinnedStatisticddResult[np.complex128, tuple[int, int]]: ...
@overload  # real, ~complex statistic
def binned_statistic_dd(
    sample: _ToSample,
    values: onp.ToArrayND[float, _AsF64],
    statistic: _StatFunc[np.float64, onp.ToJustComplex],
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
    binned_statistic_result: BinnedStatisticddResult[Any, Any] | None = None,
) -> BinnedStatisticddResult[np.complex128, tuple[int]]: ...
@overload  # real, ~complex statistic, expanded (keyword)
def binned_statistic_dd(
    sample: _ToSample,
    values: onp.ToArrayND[float, _AsF64],
    statistic: _StatFunc[np.float64, onp.ToJustComplex],
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
    binned_statistic_result: BinnedStatisticddResult[Any, Any] | None = None,
) -> BinnedStatisticddResult[np.complex128, tuple[int, int]]: ...
@overload  # values=None, count
def binned_statistic_dd(
    sample: _ToSample,
    values: None,
    statistic: Literal["count"],
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
    binned_statistic_result: BinnedStatisticddResult[Any, Any] | None = None,
) -> BinnedStatisticddResult[np.float64, tuple[int]]: ...
@overload  # values=None, count, expanded (keyword)
def binned_statistic_dd(
    sample: _ToSample,
    values: None,
    statistic: Literal["count"],
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
    binned_statistic_result: BinnedStatisticddResult[Any, Any] | None = None,
) -> BinnedStatisticddResult[np.float64, tuple[int, int]]: ...
@overload  # fallback
def binned_statistic_dd(
    sample: _ToSample,
    values: onp.ToComplexND | None,
    statistic: _Statistic | _StatFunc[Any, onp.ToComplex] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    expand_binnumbers: Literal[False] = False,
    binned_statistic_result: BinnedStatisticddResult[Any, Any] | None = None,
) -> BinnedStatisticddResult[np.float64 | np.complex128, tuple[int]]: ...
@overload  # fallback, expanded (keyword)
def binned_statistic_dd(
    sample: _ToSample,
    values: onp.ToComplexND | None,
    statistic: _Statistic | _StatFunc[Any, onp.ToComplex] = "mean",
    bins: _ToBinsND = 10,
    range: _ToRangeND = None,
    *,
    expand_binnumbers: Literal[True],
    binned_statistic_result: BinnedStatisticddResult[Any, Any] | None = None,
) -> BinnedStatisticddResult[np.float64 | np.complex128, tuple[int, int]]: ...
