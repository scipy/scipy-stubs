from collections.abc import Callable, Sequence
from typing import Any, Generic, Literal, NamedTuple, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["binned_statistic", "binned_statistic_2d", "binned_statistic_dd"]

_Statistic: TypeAlias = Literal["mean", "std", "median", "count", "sum", "min", "max"]
_BinNumber_co = TypeVar("_BinNumber_co", bound=tuple[Any, ...], covariant=True, default=tuple[int] | tuple[int, int])

class BinnedStatisticResult(NamedTuple):
    statistic: onp.Array1D[npc.inexact]
    bin_edges: onp.Array1D[np.float64]
    binnumber: onp.Array1D[np.intp]

def binned_statistic(
    x: onp.ToComplex1D,
    values: onp.ToComplex1D | Sequence[onp.ToComplex1D],
    statistic: _Statistic | Callable[[onp.Array1D[np.float64]], onp.ToFloat] = "mean",
    bins: onp.ToInt | onp.ToFloat1D = 10,
    range: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
) -> BinnedStatisticResult: ...

class BinnedStatistic2dResult(NamedTuple, Generic[_BinNumber_co]):
    statistic: onp.Array2D[npc.inexact]
    x_edge: onp.Array1D[np.float64]
    y_edge: onp.Array1D[np.float64]
    binnumber: onp.Array[_BinNumber_co, np.intp]

#
@overload
def binned_statistic_2d(
    x: onp.ToComplex1D,
    y: onp.ToComplex1D,
    values: onp.ToComplex1D | Sequence[onp.ToComplex1D],
    statistic: _Statistic | Callable[[onp.ArrayND[np.float64]], onp.ToFloat] = "mean",
    bins: onp.ToInt | onp.ToFloat1D | Sequence[onp.ToFloat1D] = 10,
    range: tuple[int, int] | None = None,
    expand_binnumbers: Literal[False] = False,
) -> BinnedStatistic2dResult[tuple[int]]: ...
@overload
def binned_statistic_2d(
    x: onp.ToComplex1D,
    y: onp.ToComplex1D,
    values: onp.ToComplex1D | Sequence[onp.ToComplex1D],
    statistic: _Statistic | Callable[[onp.ArrayND[np.float64]], onp.ToFloat] = "mean",
    bins: onp.ToInt | onp.ToFloat1D | Sequence[onp.ToFloat1D] = 10,
    range: tuple[int, int] | None = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatistic2dResult[tuple[int, int]]: ...

class BinnedStatisticddResult(NamedTuple, Generic[_BinNumber_co]):
    statistic: onp.ArrayND[npc.inexact]
    bin_edges: list[onp.Array1D[np.float64]]
    binnumber: onp.Array[_BinNumber_co, np.intp]

#
@overload
def binned_statistic_dd(
    sample: onp.ToComplex2D,
    values: onp.ToComplex1D | Sequence[onp.ToComplex1D],
    statistic: _Statistic | Callable[[onp.ArrayND[np.float64]], onp.ToFloat] = "mean",
    bins: onp.ToInt | onp.ToFloat1D = 10,
    range: tuple[int, int] | None = None,
    expand_binnumbers: Literal[False] = False,
    binned_statistic_result: BinnedStatisticddResult | None = None,
) -> BinnedStatisticddResult[tuple[int]]: ...
@overload
def binned_statistic_dd(
    sample: onp.ToComplex2D,
    values: onp.ToComplex1D | Sequence[onp.ToComplex1D],
    statistic: _Statistic | Callable[[onp.ArrayND[np.float64]], onp.ToFloat] = "mean",
    bins: onp.ToInt | onp.ToFloat1D = 10,
    range: tuple[int, int] | None = None,
    binned_statistic_result: BinnedStatisticddResult | None = None,
    *,
    expand_binnumbers: Literal[True],
) -> BinnedStatisticddResult[tuple[int, int]]: ...
