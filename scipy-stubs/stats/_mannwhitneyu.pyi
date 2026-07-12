from typing import Any, Generic, Literal as L, Never, Self, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._resampling import PermutationMethod
from ._typing import Alternative, NanPolicy

###

# `Any` is used as shape-type for numpy<2.1 compatibility (because on numpy 2.0 it's invariant)
_StatisticT_co = TypeVar("_StatisticT_co", bound=npc.floating | onp.ArrayND[npc.floating], default=Any, covariant=True)
_PValueT_co = TypeVar("_PValueT_co", bound=np.float64 | onp.ArrayND[np.float64], default=Any, covariant=True)

type _AsFloat64 = npc.floating64 | npc.integer | np.bool

type _MannwhitneyuResult0D[FloatT: npc.floating] = MannwhitneyuResult[FloatT, np.float64]
type _MannwhitneyuResultND[FloatT: npc.floating, ShapeT: tuple[int, ...]] = MannwhitneyuResult[
    onp.ArrayND[FloatT, ShapeT], onp.ArrayND[np.float64, ShapeT]
]

type _MannWhitneyUMethod = L["auto", "asymptotic", "exact"] | PermutationMethod

type _AnyShape = tuple[Any, ...]
type _JustAnyShape = tuple[Never, Never, Never, Never]  # workaround for https://github.com/microsoft/pyright/issues/10232

###

# at runtime this is a dynamically created tuple subclass from `_make_tuple_bunch`, but its `extra_fields` aren't used.
class MannwhitneyuResult(tuple[_StatisticT_co, _PValueT_co], Generic[_StatisticT_co, _PValueT_co]):
    def __new__(_cls, /, statistic: _StatisticT_co, pvalue: _PValueT_co) -> Self: ...
    def __init__(self, /, statistic: _StatisticT_co, pvalue: _PValueT_co) -> None: ...

    #
    @property
    def statistic(self) -> _StatisticT_co: ...
    @property
    def pvalue(self) -> _PValueT_co: ...

    zstatistic: _StatisticT_co

    #
    def __getnewargs_ex__(self) -> tuple[tuple[_StatisticT_co, _PValueT_co], dict[str, Never]]: ...

# undocumented
def mwu_result_object[StatT: npc.floating | onp.ArrayND[npc.floating], PValT: np.float64 | onp.ArrayND[np.float64]](
    statistic: StatT, pvalue: PValT, zstatistic: object | None = None
) -> MannwhitneyuResult[StatT, PValT]: ...

#
@overload  # ?d ~f64
def mannwhitneyu(
    x: onp.ArrayND[_AsFloat64, _JustAnyShape],
    y: onp.ArrayND[_AsFloat64, _JustAnyShape],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResult0D[np.float64] | _MannwhitneyuResultND[np.float64, _AnyShape]: ...
@overload  # ?d ~T
def mannwhitneyu[FloatT: npc.floating](
    x: onp.ArrayND[FloatT, _JustAnyShape],
    y: onp.ArrayND[FloatT, _JustAnyShape],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResult0D[FloatT] | _MannwhitneyuResultND[FloatT, _AnyShape]: ...
@overload  # 1d ~f64
def mannwhitneyu(
    x: onp.ToArrayStrict1D[float, _AsFloat64],
    y: onp.ToArrayStrict1D[float, _AsFloat64],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResult0D[np.float64]: ...
@overload  # 1d ~T
def mannwhitneyu[FloatT: npc.floating](
    x: onp.ToArrayStrict1D[FloatT, FloatT],
    y: onp.ToArrayStrict1D[FloatT, FloatT],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResult0D[FloatT]: ...
@overload  # 2d ~f64
def mannwhitneyu(
    x: onp.ToArrayStrict2D[float, _AsFloat64],
    y: onp.ToArrayStrict2D[float, _AsFloat64],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResultND[np.float64, tuple[int]]: ...
@overload  # 2d ~T
def mannwhitneyu[FloatT: npc.floating](
    x: onp.ToArrayStrict2D[FloatT, FloatT],
    y: onp.ToArrayStrict2D[FloatT, FloatT],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResultND[FloatT, tuple[int]]: ...
@overload  # 3d ~f64
def mannwhitneyu(
    x: onp.ToArrayStrict3D[float, _AsFloat64],
    y: onp.ToArrayStrict3D[float, _AsFloat64],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResultND[np.float64, tuple[int, int]]: ...
@overload  # 3d ~T
def mannwhitneyu[FloatT: npc.floating](
    x: onp.ToArrayStrict3D[FloatT, FloatT],
    y: onp.ToArrayStrict3D[FloatT, FloatT],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResultND[FloatT, tuple[int, int]]: ...
@overload  # nd ~f64
def mannwhitneyu(
    x: onp.ToArrayND[float, _AsFloat64],
    y: onp.ToArrayND[float, _AsFloat64],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResult0D[np.float64] | _MannwhitneyuResultND[np.float64, _AnyShape]: ...
@overload  # nd ~f64, axis=None  (keyword)
def mannwhitneyu(
    x: onp.ToArrayND[float, _AsFloat64],
    y: onp.ToArrayND[float, _AsFloat64],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    *,
    axis: None,
    method: _MannWhitneyUMethod = "auto",
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResult0D[np.float64]: ...
@overload  # nd ~f64, keepdims=True
def mannwhitneyu(
    x: onp.ToArrayND[float, _AsFloat64],
    y: onp.ToArrayND[float, _AsFloat64],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int | None = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[True],
) -> _MannwhitneyuResultND[np.float64, _AnyShape]: ...
@overload  # nd ~T
def mannwhitneyu[FloatT: npc.floating](
    x: onp.ToArrayND[FloatT, FloatT],
    y: onp.ToArrayND[FloatT, FloatT],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResult0D[FloatT] | _MannwhitneyuResultND[FloatT, _AnyShape]: ...
@overload  # nd ~T, axis=None  (keyword)
def mannwhitneyu[FloatT: npc.floating](
    x: onp.ToArrayND[FloatT, FloatT],
    y: onp.ToArrayND[FloatT, FloatT],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    *,
    axis: None,
    method: _MannWhitneyUMethod = "auto",
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResult0D[FloatT]: ...
@overload  # nd ~T, keepdims=True
def mannwhitneyu[FloatT: npc.floating](
    x: onp.ToArrayND[FloatT, FloatT],
    y: onp.ToArrayND[FloatT, FloatT],
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int | None = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[True],
) -> _MannwhitneyuResultND[FloatT, _AnyShape]: ...
@overload  # nd +floating
def mannwhitneyu(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResult0D[npc.floating] | _MannwhitneyuResultND[npc.floating, _AnyShape]: ...
@overload  # nd +floating, axis=None  (keyword)
def mannwhitneyu(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    *,
    axis: None,
    method: _MannWhitneyUMethod = "auto",
    nan_policy: NanPolicy = "propagate",
    keepdims: L[False] = False,
) -> _MannwhitneyuResult0D[npc.floating]: ...
@overload  # nd +floating, keepdims=True
def mannwhitneyu(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int | None = 0,
    method: _MannWhitneyUMethod = "auto",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: L[True],
) -> _MannwhitneyuResultND[npc.floating, _AnyShape]: ...
