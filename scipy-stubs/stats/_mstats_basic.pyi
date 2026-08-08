# mypy's false positive `overload-overlap` errors here differ per numpy version
# mypy: disable-error-code=overload-overlap
from collections.abc import Callable
from typing import (
    Any,
    Concatenate,
    Final,
    Generic,
    Literal,
    NamedTuple,
    Never,
    Self,
    SupportsIndex,
    TypedDict,
    overload,
    override,
    type_check_only,
)
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc
from numpy._typing import _ArrayLike

from ._stats_mstats_common import SiegelslopesResult, TheilslopesResult
from ._stats_py import KstestResult, LinregressResult, SignificanceResult
from ._typing import Alternative, BaseBunch, NanPolicy

__all__ = [
    "argstoarray",
    "brunnermunzel",
    "count_tied_groups",
    "describe",
    "f_oneway",
    "find_repeats",
    "friedmanchisquare",
    "kendalltau",
    "kendalltau_seasonal",
    "kruskal",
    "kruskalwallis",
    "ks_1samp",
    "ks_2samp",
    "ks_twosamp",
    "kstest",
    "kurtosis",
    "kurtosistest",
    "linregress",
    "mannwhitneyu",
    "meppf",
    "mode",
    "moment",
    "mquantiles",
    "msign",
    "normaltest",
    "obrientransform",
    "pearsonr",
    "plotting_positions",
    "pointbiserialr",
    "rankdata",
    "scoreatpercentile",
    "sem",
    "sen_seasonal_slopes",
    "siegelslopes",
    "skew",
    "skewtest",
    "spearmanr",
    "theilslopes",
    "tmax",
    "tmean",
    "tmin",
    "trim",
    "trima",
    "trimboth",
    "trimmed_mean",
    "trimmed_std",
    "trimmed_stde",
    "trimmed_var",
    "trimr",
    "trimtail",
    "tsem",
    "ttest_1samp",
    "ttest_ind",
    "ttest_onesamp",
    "ttest_rel",
    "tvar",
    "variation",
    "winsorize",
]

###

type _MArrayOrND[ScalarT: np.generic] = ScalarT | onp.MArray[ScalarT]

type _AsF64 = np.float64 | npc.integer | np.bool

type _JustAnyShape = tuple[Never, Never, Never, Never]  # workaround for https://github.com/microsoft/pyright/issues/10232
type _ToFloatStrictND = onp.ArrayND[npc.floating | npc.integer | np.bool, _JustAnyShape]

# workaround for a strange bug in pyright's overlapping overload detection with `numpy<2.1`
type _WorkaroundForPyright = tuple[int] | tuple[Any, ...]

type _KendallTauMethod = Literal["auto", "asymptotic", "exact"]
type _TheilSlopesMethod = Literal["joint", "separate"]
type _SiegelSlopesMethod = Literal["hierarchical", "separate"]

type _KSMethod = Literal["auto", "exact", "asymp"]
type _KTestMethod = Literal[_KSMethod, "approx"]

type _Describe0D[MinMaxT: npc.number | np.bool, MeanT: npc.inexact, VarT: npc.inexact, SkewT: npc.inexact] = DescribeResult[
    tuple[()], MinMaxT, MeanT, VarT, SkewT, SkewT
]
type _Describe1D[MinMaxT: npc.number | np.bool, MeanT: npc.inexact, VarT: npc.inexact, SkewT: npc.inexact] = DescribeResult[
    tuple[int], MinMaxT, onp.MArray1D[MeanT], onp.MArray1D[VarT], SkewT, onp.MArray1D[SkewT]
]  # fmt: skip
type _DescribeND[MinMaxT: npc.number | np.bool, MeanT: npc.inexact, VarT: npc.inexact, SkewT: npc.inexact] = DescribeResult[
    tuple[Any, ...], MinMaxT, onp.MArray[MeanT] | Any, onp.MArray[VarT] | Any, SkewT, onp.MArray[SkewT] | Any
]  # fmt: skip

_NDT_f_co = TypeVar(
    "_NDT_f_co", covariant=True, bound=float | npc.floating | onp.ArrayND[npc.floating], default=onp.MArray[np.float64]
)
_NDT_fc_co = TypeVar(
    "_NDT_fc_co",
    covariant=True,
    bound=complex | _MArrayOrND[npc.inexact],
    default=_MArrayOrND[np.float64 | np.complex128],
)  # fmt: skip

_ShapeT_co = TypeVar("_ShapeT_co", covariant=True, bound=tuple[int, ...], default=tuple[Any, ...])
_MinMaxT_co = TypeVar("_MinMaxT_co", covariant=True, bound=npc.number | np.bool, default=Any)
_MeanT_co = TypeVar("_MeanT_co", covariant=True, bound=_MArrayOrND[npc.inexact], default=_MArrayOrND[Any])
_VarT_co = TypeVar("_VarT_co", covariant=True, bound=_MArrayOrND[npc.inexact], default=_MArrayOrND[Any])
_SkewT_co = TypeVar("_SkewT_co", covariant=True, bound=npc.inexact, default=Any)
_KurtT_co = TypeVar("_KurtT_co", covariant=True, bound=_MArrayOrND[npc.inexact], default=_MArrayOrND[Any])

@type_check_only
class _TestResult(NamedTuple, Generic[_NDT_f_co, _NDT_fc_co]):
    statistic: _NDT_fc_co
    pvalue: _NDT_f_co

_KendallTauSeasonalResult = TypedDict(
    "_KendallTauSeasonalResult",
    {
        "seasonal tau": _MArrayOrND[np.float64],
        "global tau": np.float64,
        "global tau (alt)": np.float64,
        "seasonal p-value": onp.ArrayND[np.float64],
        "global p-value (indep)": np.float64,
        "global p-value (dep)": np.float64,
        "chi2 total": onp.MArray[np.float64],
        "chi2 trend": onp.MArray[np.float64],
    },
)

###

trimdoc: Final[str] = ...

class ModeResult(NamedTuple):
    mode: onp.MArray[np.float64]
    count: onp.MArray[np.float64]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]

class DescribeResult(NamedTuple, Generic[_ShapeT_co, _MinMaxT_co, _MeanT_co, _VarT_co, _SkewT_co, _KurtT_co]):
    nobs: onp.Array[_ShapeT_co, np.int_]
    minmax: tuple[onp.MArray[_MinMaxT_co, _ShapeT_co], onp.MArray[_MinMaxT_co, _ShapeT_co]]
    mean: _MeanT_co
    variance: _VarT_co
    skewness: onp.MArray[_SkewT_co, _ShapeT_co]
    kurtosis: _KurtT_co

class PointbiserialrResult(NamedTuple):
    correlation: np.float64
    pvalue: np.float64

class Ttest_relResult(_TestResult[_NDT_f_co, _NDT_fc_co], Generic[_NDT_f_co, _NDT_fc_co]): ...
class Ttest_indResult(_TestResult[_NDT_f_co, _NDT_fc_co], Generic[_NDT_f_co, _NDT_fc_co]): ...
class Ttest_1sampResult(_TestResult[_NDT_f_co, _NDT_fc_co], Generic[_NDT_f_co, _NDT_fc_co]): ...
class SkewtestResult(_TestResult[_NDT_f_co, _NDT_fc_co], Generic[_NDT_f_co, _NDT_fc_co]): ...
class KurtosistestResult(_TestResult[_NDT_f_co, _NDT_fc_co], Generic[_NDT_f_co, _NDT_fc_co]): ...
class NormaltestResult(_TestResult[_NDT_f_co, _NDT_fc_co], Generic[_NDT_f_co, _NDT_fc_co]): ...
class MannwhitneyuResult(_TestResult[np.float64, np.float64]): ...
class F_onewayResult(_TestResult[np.float64, np.float64]): ...
class KruskalResult(_TestResult[np.float64, np.float64]): ...
class FriedmanchisquareResult(_TestResult[np.float64, np.float64]): ...
class BrunnerMunzelResult(_TestResult[np.float64, np.float64]): ...

class SenSeasonalSlopesResult(BaseBunch[onp.MArray[np.float64], np.float64]):
    @override
    def __new__(_cls, intra_slope: float, inter_slope: float) -> Self: ...  # pyrefly:ignore[bad-override]
    @override
    def __init__(self, /, intra_slope: float, inter_slope: float) -> None: ...  # pyrefly:ignore[bad-override]

    #
    @property
    def intra_slope(self, /) -> onp.MArray[np.float64]: ...
    @property
    def inter_slope(self, /) -> float: ...

# TODO(jorenham): Overloads for scalar vs. array
# TODO(jorenham): Overloads for specific dtypes

def argstoarray(*args: onp.ToFloatND) -> onp.MArray[np.float64]: ...
def find_repeats(arr: onp.ToFloatND) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.intp]]: ...
def count_tied_groups(x: onp.ToFloatND, use_missing: bool = False) -> dict[np.intp, np.intp | int]: ...
def rankdata(data: onp.ToFloatND, axis: SupportsIndex | None = None, use_missing: bool = False) -> onp.ArrayND[np.float64]: ...
def mode(a: onp.ToFloatND, axis: SupportsIndex | None = 0) -> ModeResult: ...

#
@overload
def msign[ScalarT: npc.number | np.timedelta64 | np.bool | np.object_](x: _ArrayLike[ScalarT]) -> onp.ArrayND[ScalarT]: ...
@overload
def msign(x: onp.ToComplexND) -> onp.ArrayND[npc.number | np.timedelta64 | np.bool | np.object_]: ...

#
def pearsonr(x: onp.ToFloatND, y: onp.ToFloatND) -> tuple[np.float64, np.float64]: ...

#
# NOTE: `y` is required with `axis=None` (default)
@overload  # ?d, ?d, axis=None (default)
def spearmanr(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    use_ties: bool = True,
    axis: None = None,
    nan_policy: NanPolicy = "propagate",
    alternative: Alternative = "two-sided",
) -> SignificanceResult[np.float64]: ...
@overload  # ?d, ?d, axis=<given>
def spearmanr(
    x: _ToFloatStrictND,
    y: _ToFloatStrictND,
    use_ties: bool = True,
    *,
    axis: SupportsIndex,
    nan_policy: NanPolicy = "propagate",
    alternative: Alternative = "two-sided",
) -> SignificanceResult[onp.Array2D[np.float64] | Any]: ...
@overload  # 1d, 1d, axis=<given>
def spearmanr(
    x: onp.ToFloatStrict1D,
    y: onp.ToFloatStrict1D,
    use_ties: bool = True,
    *,
    axis: SupportsIndex,
    nan_policy: NanPolicy = "propagate",
    alternative: Alternative = "two-sided",
) -> SignificanceResult[np.float64]: ...
@overload  # 2d, 2d, axis=<given>
def spearmanr(
    x: onp.ToFloatStrict2D,
    y: onp.ToFloatStrict2D,
    use_ties: bool = True,
    *,
    axis: SupportsIndex,
    nan_policy: NanPolicy = "propagate",
    alternative: Alternative = "two-sided",
) -> SignificanceResult[onp.Array2D[np.float64]]: ...
@overload  # fallback
def spearmanr(
    x: onp.ToFloatND,
    y: onp.ToFloatND | None = None,
    use_ties: bool = True,
    axis: SupportsIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    alternative: Alternative = "two-sided",
) -> SignificanceResult[onp.Array2D[np.float64] | Any]: ...

#
def kendalltau(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    use_ties: bool = True,
    use_missing: bool = False,
    method: _KendallTauMethod = "auto",
    alternative: Alternative = "two-sided",
) -> SignificanceResult: ...
def kendalltau_seasonal(x: onp.ToFloatND) -> _KendallTauSeasonalResult: ...
def pointbiserialr(x: onp.ToFloatND, y: onp.ToFloatND) -> PointbiserialrResult: ...

# NOTE: flattens input
def linregress(x: onp.ToFloatND, y: onp.ToFloatND | None = None) -> LinregressResult[np.float64]: ...

# NOTE: flattens input
def theilslopes(
    y: onp.ToFloatND, x: onp.ToFloatND | None = None, alpha: float | npc.floating = 0.95, method: _TheilSlopesMethod = "separate"
) -> TheilslopesResult[np.float64]: ...

#
def siegelslopes(
    y: onp.ToFloatND, x: onp.ToFloatND | None = None, method: _SiegelSlopesMethod = "hierarchical"
) -> SiegelslopesResult: ...
def sen_seasonal_slopes(x: onp.ToFloatND) -> SenSeasonalSlopesResult: ...

#
def ttest_1samp(
    a: onp.ToFloatND, popmean: onp.ToFloat | onp.ToFloatND, axis: SupportsIndex | None = 0, alternative: Alternative = "two-sided"
) -> Ttest_1sampResult: ...
def ttest_ind(
    a: onp.ToFloatND,
    b: onp.ToFloatND,
    axis: SupportsIndex | None = 0,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult: ...
def ttest_rel(
    a: onp.ToFloatND, b: onp.ToFloatND, axis: SupportsIndex | None = 0, alternative: Alternative = "two-sided"
) -> Ttest_relResult: ...
def mannwhitneyu(x: onp.ToFloatND, y: onp.ToFloatND, use_continuity: bool = True) -> MannwhitneyuResult: ...
def kruskal(arg0: onp.ToFloatND, arg1: onp.ToFloatND, /, *args: onp.ToFloatND) -> KruskalResult: ...

#
@overload
def ks_1samp(
    x: onp.ToFloatND,
    cdf: str | Callable[[float], onp.ToFloat],
    args: tuple[()] = (),
    alternative: Alternative = "two-sided",
    method: _KSMethod = "auto",
) -> KstestResult: ...
@overload
def ks_1samp(
    x: onp.ToFloatND,
    cdf: str | Callable[Concatenate[float, ...], onp.ToFloat],
    args: tuple[object, ...],
    alternative: Alternative = "two-sided",
    method: _KSMethod = "auto",
) -> KstestResult: ...

#
def ks_2samp(
    data1: onp.ToFloatND, data2: onp.ToFloatND, alternative: Alternative = "two-sided", method: _KSMethod = "auto"
) -> KstestResult: ...

#
@overload
def kstest(
    data1: onp.ToFloatND,
    data2: onp.ToFloatND | str | Callable[[float], onp.ToFloat],
    args: tuple[()] = (),
    alternative: Alternative = "two-sided",
    method: _KTestMethod = "auto",
) -> KstestResult: ...
@overload
def kstest(
    data1: onp.ToFloatND,
    data2: Callable[Concatenate[float, ...], onp.ToFloat],
    args: tuple[object, ...],
    alternative: Alternative = "two-sided",
    method: _KTestMethod = "auto",
) -> KstestResult: ...

#
@overload
def trima(
    a: onp.SequenceND[bool], limits: tuple[onp.ToInt, onp.ToInt] | None = None, inclusive: tuple[bool, bool] = (True, True)
) -> onp.MArray[np.bool]: ...
@overload
def trima(
    a: onp.SequenceND[op.JustInt], limits: tuple[onp.ToInt, onp.ToInt] | None = None, inclusive: tuple[bool, bool] = (True, True)
) -> onp.MArray[np.int_]: ...
@overload
def trima(
    a: onp.SequenceND[float], limits: tuple[onp.ToFloat, onp.ToFloat] | None = None, inclusive: tuple[bool, bool] = (True, True)
) -> onp.MArray[np.float64 | np.int_ | np.bool]: ...
@overload
def trima(
    a: onp.SequenceND[complex],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
) -> onp.MArray[np.complex128 | np.float64 | np.int_ | np.bool]: ...
@overload
def trima[ScalarT: npc.number | np.bool](
    a: _ArrayLike[ScalarT], limits: tuple[onp.ToComplex, onp.ToComplex] | None = None, inclusive: tuple[bool, bool] = (True, True)
) -> onp.MArray[ScalarT]: ...

#
@overload
def trimr(
    a: onp.SequenceND[op.JustInt | np.int_],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.int_]: ...
@overload
def trimr(
    a: onp.SequenceND[float],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.float64 | np.int_]: ...
@overload
def trimr(
    a: onp.SequenceND[complex],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.complex128 | np.float64 | np.int_]: ...
@overload
def trimr[ScalarT: npc.number | np.bool](
    a: _ArrayLike[ScalarT],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[ScalarT]: ...

#
@overload  # 1d bool
def trim(
    a: list[bool],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    relative: bool = False,
    axis: SupportsIndex | None = None,
) -> onp.MArray1D[np.bool]: ...
@overload  # ?d bool
def trim(
    a: onp.SequenceND[list[bool]],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    relative: bool = False,
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.bool]: ...
@overload  # 1d ~int
def trim(
    a: list[int],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    relative: bool = False,
    axis: SupportsIndex | None = None,
) -> onp.MArray1D[np.int_]: ...
@overload  # ?d ~int
def trim(
    a: onp.SequenceND[list[int]],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    relative: bool = False,
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.int_]: ...
@overload  # 1d ~float
def trim(
    a: list[float],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    relative: bool = False,
    axis: SupportsIndex | None = None,
) -> onp.MArray1D[np.float64]: ...
@overload  # ?d ~float
def trim(
    a: onp.SequenceND[list[float]],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    relative: bool = False,
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.float64]: ...
@overload  # 1d ~complex
def trim(
    a: list[complex],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    relative: bool = False,
    axis: SupportsIndex | None = None,
) -> onp.MArray1D[np.complex128]: ...
@overload  # ?d ~complex
def trim(
    a: onp.SequenceND[list[complex]],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    relative: bool = False,
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.complex128]: ...
@overload  # ?d T@+number
def trim[ShapeT: tuple[int, ...], ScalarT: npc.number | np.bool](
    a: onp.ArrayND[ScalarT, ShapeT],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    relative: bool = False,
    axis: SupportsIndex | None = None,
) -> onp.MArray[ScalarT, ShapeT]: ...
@overload  # Nd T@+number
def trim[ScalarT: npc.number | np.bool](
    a: onp.ToArrayND[ScalarT, ScalarT],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    relative: bool = False,
    axis: SupportsIndex | None = None,
) -> onp.MArray[ScalarT, _WorkaroundForPyright]: ...

#
@overload
def trimboth(
    data: onp.SequenceND[op.JustInt | np.int_],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.int_]: ...
@overload
def trimboth(
    data: onp.SequenceND[float],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.float64 | np.int_]: ...
@overload
def trimboth(
    data: onp.SequenceND[complex],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.complex128 | np.float64 | np.int_]: ...
@overload
def trimboth[ScalarT: npc.number | np.bool](
    data: _ArrayLike[ScalarT],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[ScalarT]: ...

#
@overload
def trimtail(
    data: onp.SequenceND[op.JustInt | np.int_],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.int_]: ...
@overload
def trimtail(
    data: onp.SequenceND[float],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.float64 | np.int_]: ...
@overload
def trimtail(
    data: onp.SequenceND[complex],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.complex128 | np.float64 | np.int_]: ...
@overload
def trimtail[ScalarT: npc.number | np.bool](
    data: _ArrayLike[ScalarT],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[ScalarT]: ...

#
# NOTE: f32/c64 promotes to f64/c128
@overload  # ?d ~f64, axis=None (default)
def trimmed_mean(
    a: onp.ToArrayND[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    axis: None = None,
) -> np.float64: ...
@overload  # ?d ~c128, axis=None (default)
def trimmed_mean(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    axis: None = None,
) -> np.complex128: ...
@overload  # ?d T@inexact, axis=None (default)
def trimmed_mean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayND[InexactT, InexactT],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    axis: None = None,
) -> InexactT: ...
@overload  # ?d ~f64, axis=<given>
def trimmed_mean(
    a: onp.ArrayND[_AsF64 | np.float32, _JustAnyShape],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray[np.float64] | Any: ...
@overload  # ?d ~c128, axis=<given>
def trimmed_mean(
    a: onp.ArrayND[np.complex128 | np.complex64, _JustAnyShape],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray[np.complex128] | Any: ...
@overload  # ?d T@inexact, axis=<given>
def trimmed_mean[InexactT: npc.inexact80 | np.float16](
    a: onp.ArrayND[InexactT, _JustAnyShape],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray[InexactT] | Any: ...
@overload  # 1d ~f64, axis=<given>
def trimmed_mean(
    a: onp.ToArrayStrict1D[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> np.float64: ...
@overload  # 1d ~c128, axis=<given>
def trimmed_mean(
    a: onp.ToArrayStrict1D[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> np.complex128: ...
@overload  # 1d T@inexact, axis=<given>
def trimmed_mean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayStrict1D[InexactT, InexactT],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> InexactT: ...
@overload  # 2d ~f64, axis=<given>
def trimmed_mean(
    a: onp.ToArrayStrict2D[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray1D[np.float64]: ...
@overload  # 2d ~c128, axis=<given>
def trimmed_mean(
    a: onp.ToArrayStrict2D[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray1D[np.complex128]: ...
@overload  # 2d T@inexact, axis=<given>
def trimmed_mean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayStrict2D[InexactT, InexactT],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray1D[InexactT]: ...
@overload  # 3d ~f64, axis=<given>
def trimmed_mean(
    a: onp.ToArrayStrict3D[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray2D[np.float64]: ...
@overload  # 3d ~c128, axis=<given>
def trimmed_mean(
    a: onp.ToArrayStrict3D[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray2D[np.complex128]: ...
@overload  # 3d T@inexact, axis=<given>
def trimmed_mean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayStrict3D[InexactT, InexactT],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray2D[InexactT]: ...
@overload  # Nd ~f64
def trimmed_mean(
    a: onp.ToArrayND[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.float64] | Any: ...
@overload  # Nd ~c128
def trimmed_mean(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.complex128] | Any: ...
@overload  # Nd T@inexact
def trimmed_mean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayND[InexactT, InexactT],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    axis: SupportsIndex | None = None,
) -> onp.MArray[InexactT] | Any: ...

#
def trimmed_var(
    a: onp.ToComplexND,
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    axis: SupportsIndex | None = None,
    ddof: onp.ToInt = 0,
) -> _MArrayOrND[np.float64]: ...

#
def trimmed_std(
    a: onp.ToComplexND,
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: bool = True,
    axis: SupportsIndex | None = None,
    ddof: onp.ToInt = 0,
) -> _MArrayOrND[np.float64]: ...

#
def trimmed_stde(
    a: onp.ToComplexND,
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    axis: SupportsIndex | None = None,
) -> _MArrayOrND[np.float64]: ...

#
# NOTE: f32/c64 promotes to f64/c128
@overload  # ?d ~f64, axis=None (default)
def tmean(
    a: onp.ToArrayND[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: None = None,
) -> np.float64: ...
@overload  # ?d ~c128, axis=None (default)
def tmean(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: None = None,
) -> np.complex128: ...
@overload  # ?d T@inexact, axis=None (default)
def tmean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayND[InexactT, InexactT],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: None = None,
) -> InexactT: ...
@overload  # ?d +integer | ~f32, axis=<given>
def tmean(
    a: onp.ArrayND[_AsF64 | np.float32, _JustAnyShape],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    *,
    axis: SupportsIndex,
) -> onp.MArray[np.float64] | Any: ...
@overload  # ?d ~c64, axis=<given>
def tmean(
    a: onp.ArrayND[np.complex128 | np.complex64, _JustAnyShape],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    *,
    axis: SupportsIndex,
) -> onp.MArray[np.complex128] | Any: ...
@overload  # ?d T@inexact, axis=<given>
def tmean[InexactT: npc.inexact80 | np.float16](
    a: onp.ArrayND[InexactT, _JustAnyShape],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    *,
    axis: SupportsIndex,
) -> onp.MArray[InexactT] | Any: ...
@overload  # 1d +f64 | ~f32, axis=<given>
def tmean(
    a: onp.ToArrayStrict1D[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    *,
    axis: SupportsIndex,
) -> np.float64: ...
@overload  # 1d ~complex | ~c64, axis=<given>
def tmean(
    a: onp.ToArrayStrict1D[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    *,
    axis: SupportsIndex,
) -> np.complex128: ...
@overload  # 1d T@inexact, axis=<given>
def tmean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayStrict1D[InexactT, InexactT],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    *,
    axis: SupportsIndex,
) -> InexactT: ...
@overload  # 2d ~f64, axis=<given>
def tmean(
    a: onp.ToArrayStrict2D[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    *,
    axis: SupportsIndex,
) -> onp.MArray1D[np.float64]: ...
@overload  # 2d ~c128, axis=<given>
def tmean(
    a: onp.ToArrayStrict2D[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    *,
    axis: SupportsIndex,
) -> onp.MArray1D[np.complex128]: ...
@overload  # 2d T@inexact, axis=<given>
def tmean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayStrict2D[InexactT, InexactT],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    *,
    axis: SupportsIndex,
) -> onp.MArray1D[InexactT]: ...
@overload  # Nd ~f64
def tmean(
    a: onp.ToArrayND[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.float64] | Any: ...
@overload  # Nd ~c128
def tmean(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.complex128] | Any: ...
@overload  # Nd T@inexact
def tmean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayND[InexactT, InexactT],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[InexactT] | Any: ...

#
def tvar(
    a: onp.MArray[npc.floating | npc.integer],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = 0,
    ddof: onp.ToInt = 1,
) -> _MArrayOrND[npc.floating]: ...

#
@overload
def tmin(
    a: onp.SequenceND[op.JustInt | np.int_],
    lowerlimit: onp.ToFloat | None = None,
    axis: SupportsIndex | None = 0,
    inclusive: bool = True,
) -> _MArrayOrND[np.int_]: ...
@overload
def tmin(
    a: onp.SequenceND[float], lowerlimit: onp.ToFloat | None = None, axis: SupportsIndex | None = 0, inclusive: bool = True
) -> _MArrayOrND[np.float64 | np.int_]: ...
@overload
def tmin(
    a: onp.SequenceND[complex], lowerlimit: onp.ToComplex | None = None, axis: SupportsIndex | None = 0, inclusive: bool = True
) -> _MArrayOrND[np.complex128 | np.float64 | np.int_]: ...
@overload
def tmin[ScalarT: npc.number | np.bool](
    a: _ArrayLike[ScalarT], lowerlimit: onp.ToComplex | None = None, axis: SupportsIndex | None = 0, inclusive: bool = True
) -> _MArrayOrND[ScalarT]: ...

#
@overload
def tmax(
    a: onp.SequenceND[op.JustInt | np.int_],
    upperlimit: onp.ToFloat | None = None,
    axis: SupportsIndex | None = 0,
    inclusive: bool = True,
) -> _MArrayOrND[np.int_]: ...
@overload
def tmax(
    a: onp.SequenceND[float], upperlimit: onp.ToFloat | None = None, axis: SupportsIndex | None = 0, inclusive: bool = True
) -> _MArrayOrND[np.float64 | np.int_]: ...
@overload
def tmax(
    a: onp.SequenceND[complex], upperlimit: onp.ToComplex | None = None, axis: SupportsIndex | None = 0, inclusive: bool = True
) -> _MArrayOrND[np.complex128 | np.float64 | np.int_]: ...
@overload
def tmax[ScalarT: npc.number | np.bool](
    a: _ArrayLike[ScalarT], upperlimit: onp.ToComplex | None = None, axis: SupportsIndex | None = 0, inclusive: bool = True
) -> _MArrayOrND[ScalarT]: ...

#
def tsem(
    a: onp.ToComplexND,
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = 0,
    ddof: onp.ToInt = 1,
) -> _MArrayOrND[np.float64]: ...

#
@overload
def winsorize(
    a: onp.ToIntND,
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    inplace: bool = False,
    axis: SupportsIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> onp.MArray[np.int_]: ...
@overload
def winsorize[FloatingT: npc.floating](
    a: _ArrayLike[FloatingT],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    inplace: bool = False,
    axis: SupportsIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> onp.MArray[FloatingT]: ...
@overload
def winsorize(
    a: onp.ToFloatND,
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    inplace: bool = False,
    axis: SupportsIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> onp.MArray[npc.floating | np.int_]: ...
@overload
def winsorize(
    a: onp.ToComplexND,
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    inplace: bool = False,
    axis: SupportsIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> onp.MArray[np.complex128 | npc.floating | np.int_]: ...

# NOTE: f16/f32 and c64 promote to f64 and c128, unless `moment <= 1`
@overload  # ?d ~f64, axis=None (positional)
def moment(a: onp.ToFloat64_ND, moment: onp.ToInt, axis: None) -> np.float64: ...
@overload  # ?d ~c128, axis=None (positional)
def moment(a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64], moment: onp.ToInt, axis: None) -> np.complex128: ...
@overload  # ?d T@inexact80, axis=None (positional)
def moment[InexactT: npc.inexact80](a: onp.ToArrayND[InexactT, InexactT], moment: onp.ToInt, axis: None) -> InexactT: ...
@overload  # ?d ~f64, axis=None (keyword)
def moment(a: onp.ToFloat64_ND, moment: onp.ToInt = 1, *, axis: None) -> np.float64: ...
@overload  # ?d ~c128, axis=None (keyword)
def moment(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64], moment: onp.ToInt = 1, *, axis: None
) -> np.complex128: ...
@overload  # ?d T@inexact80, axis=None (keyword)
def moment[InexactT: npc.inexact80](a: onp.ToArrayND[InexactT, InexactT], moment: onp.ToInt = 1, *, axis: None) -> InexactT: ...
@overload  # ?d ~f64, axis=<given> (default)
def moment(
    a: onp.ArrayND[_AsF64 | np.float32 | np.float16, _JustAnyShape], moment: onp.ToInt = 1, axis: SupportsIndex = 0
) -> onp.MArray[np.float64] | Any: ...
@overload  # ?d ~c128, axis=<given> (default)
def moment(
    a: onp.ArrayND[np.complex128 | np.complex64, _JustAnyShape], moment: onp.ToInt = 1, axis: SupportsIndex = 0
) -> onp.MArray[np.complex128] | Any: ...
@overload  # ?d T@inexact80, axis=<given> (default)
def moment[InexactT: npc.inexact80](
    a: onp.ArrayND[InexactT, _JustAnyShape], moment: onp.ToInt = 1, axis: SupportsIndex = 0
) -> onp.MArray[InexactT] | Any: ...
@overload  # 1d ~f64, axis=<given> (default)
def moment(a: onp.ToFloat64Strict1D, moment: onp.ToInt = 1, axis: SupportsIndex = 0) -> np.float64: ...
@overload  # 1d ~c128, axis=<given> (default)
def moment(
    a: onp.ToArrayStrict1D[op.JustComplex, np.complex128 | np.complex64], moment: onp.ToInt = 1, axis: SupportsIndex = 0
) -> np.complex128: ...
@overload  # 1d T@inexact80, axis=<given> (default)
def moment[InexactT: npc.inexact80](
    a: onp.ToArrayStrict1D[InexactT, InexactT], moment: onp.ToInt = 1, axis: SupportsIndex = 0
) -> InexactT: ...
@overload  # 2d ~f64, axis=<given> (default)
def moment(a: onp.ToFloat64Strict2D, moment: onp.ToInt = 1, axis: SupportsIndex = 0) -> onp.MArray1D[np.float64]: ...
@overload  # 2d ~c128, axis=<given> (default)
def moment(
    a: onp.ToArrayStrict2D[op.JustComplex, np.complex128 | np.complex64], moment: onp.ToInt = 1, axis: SupportsIndex = 0
) -> onp.MArray1D[np.complex128]: ...
@overload  # 2d T@inexact80, axis=<given> (default)
def moment[InexactT: npc.inexact80](
    a: onp.ToArrayStrict2D[InexactT, InexactT], moment: onp.ToInt = 1, axis: SupportsIndex = 0
) -> onp.MArray1D[InexactT]: ...
@overload  # 3d ~f64, axis=<given> (default)
def moment(a: onp.ToFloat64Strict3D, moment: onp.ToInt = 1, axis: SupportsIndex = 0) -> onp.MArray2D[np.float64]: ...
@overload  # 3d ~c128, axis=<given> (default)
def moment(
    a: onp.ToArrayStrict3D[op.JustComplex, np.complex128 | np.complex64], moment: onp.ToInt = 1, axis: SupportsIndex = 0
) -> onp.MArray2D[np.complex128]: ...
@overload  # 3d T@inexact80, axis=<given> (default)
def moment[InexactT: npc.inexact80](
    a: onp.ToArrayStrict3D[InexactT, InexactT], moment: onp.ToInt = 1, axis: SupportsIndex = 0
) -> onp.MArray2D[InexactT]: ...
@overload  # ?d ~f64, moment: 1d
def moment(a: onp.ToFloat64_ND, moment: onp.ToIntND, axis: SupportsIndex | None = 0) -> onp.MArray[np.float64]: ...
@overload  # ?d ~c128, moment: 1d
def moment(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64], moment: onp.ToIntND, axis: SupportsIndex | None = 0
) -> onp.MArray[np.complex128]: ...
@overload  # ?d T@inexact80, moment: 1d
def moment[InexactT: npc.inexact80](
    a: onp.ToArrayND[InexactT, InexactT], moment: onp.ToIntND, axis: SupportsIndex | None = 0
) -> onp.MArray[InexactT]: ...
@overload  # Nd ~f64
def moment(
    a: onp.ToFloat64_ND, moment: onp.ToInt | onp.ToIntND = 1, axis: SupportsIndex | None = 0
) -> onp.MArray[np.float64] | Any: ...
@overload  # Nd ~c128
def moment(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64],
    moment: onp.ToInt | onp.ToIntND = 1,
    axis: SupportsIndex | None = 0,
) -> onp.MArray[np.complex128] | Any: ...
@overload  # Nd T@inexact80
def moment[InexactT: npc.inexact80](
    a: onp.ToArrayND[InexactT, InexactT], moment: onp.ToInt | onp.ToIntND = 1, axis: SupportsIndex | None = 0
) -> onp.MArray[InexactT] | Any: ...

# NOTE: f16/f32 and c64 promote only if `a` is masked
@overload  # ?d ~f64, axis=None
def variation(a: onp.ToArrayND[float, _AsF64], axis: None, ddof: onp.ToInt = 0) -> np.float64: ...
@overload  # ?d ~c128, axis=None
def variation(a: onp.ToJustComplex128_ND, axis: None, ddof: onp.ToInt = 0) -> np.complex128: ...
@overload  # ?d T@inexact, axis=None
def variation[InexactT: npc.inexact](a: onp.ToArrayND[InexactT, InexactT], axis: None, ddof: onp.ToInt = 0) -> InexactT: ...
@overload  # ?d ~f64, axis=<given> (default)
def variation(
    a: onp.ArrayND[_AsF64, _JustAnyShape], axis: SupportsIndex = 0, ddof: onp.ToInt = 0
) -> onp.MArray[np.float64] | Any: ...
@overload  # ?d ~c128, axis=<given> (default)
def variation(
    a: onp.ArrayND[np.complex128, _JustAnyShape], axis: SupportsIndex = 0, ddof: onp.ToInt = 0
) -> onp.MArray[np.complex128] | Any: ...
@overload  # ?d T@inexact, axis=<given> (default)
def variation[InexactT: npc.inexact](
    a: onp.ArrayND[InexactT, _JustAnyShape], axis: SupportsIndex = 0, ddof: onp.ToInt = 0
) -> onp.MArray[InexactT] | Any: ...
@overload  # 1d ~f64, axis=<given> (default)
def variation(a: onp.ToArrayStrict1D[float, _AsF64], axis: SupportsIndex = 0, ddof: onp.ToInt = 0) -> np.float64: ...
@overload  # 1d ~c128, axis=<given> (default)
def variation(a: onp.ToJustComplex128Strict1D, axis: SupportsIndex = 0, ddof: onp.ToInt = 0) -> np.complex128: ...
@overload  # 1d T@inexact, axis=<given> (default)
def variation[InexactT: npc.inexact](
    a: onp.ToArrayStrict1D[InexactT, InexactT], axis: SupportsIndex = 0, ddof: onp.ToInt = 0
) -> InexactT: ...
@overload  # 2d ~f64, axis=<given> (default)
def variation(
    a: onp.ToArrayStrict2D[float, _AsF64], axis: SupportsIndex = 0, ddof: onp.ToInt = 0
) -> onp.MArray1D[np.float64]: ...
@overload  # 2d ~c128, axis=<given> (default)
def variation(a: onp.ToJustComplex128Strict2D, axis: SupportsIndex = 0, ddof: onp.ToInt = 0) -> onp.MArray1D[np.complex128]: ...
@overload  # 2d T@inexact, axis=<given> (default)
def variation[InexactT: npc.inexact](
    a: onp.ToArrayStrict2D[InexactT, InexactT], axis: SupportsIndex = 0, ddof: onp.ToInt = 0
) -> onp.MArray1D[InexactT]: ...
@overload  # 3d ~f64, axis=<given> (default)
def variation(
    a: onp.ToArrayStrict3D[float, _AsF64], axis: SupportsIndex = 0, ddof: onp.ToInt = 0
) -> onp.MArray2D[np.float64]: ...
@overload  # 3d ~c128, axis=<given> (default)
def variation(a: onp.ToJustComplex128Strict3D, axis: SupportsIndex = 0, ddof: onp.ToInt = 0) -> onp.MArray2D[np.complex128]: ...
@overload  # 3d T@inexact, axis=<given> (default)
def variation[InexactT: npc.inexact](
    a: onp.ToArrayStrict3D[InexactT, InexactT], axis: SupportsIndex = 0, ddof: onp.ToInt = 0
) -> onp.MArray2D[InexactT]: ...
@overload  # Nd ~f64
def variation(
    a: onp.ToArrayND[float, _AsF64], axis: SupportsIndex | None = 0, ddof: onp.ToInt = 0
) -> onp.MArray[np.float64] | Any: ...
@overload  # Nd ~c128
def variation(
    a: onp.ToJustComplex128_ND, axis: SupportsIndex | None = 0, ddof: onp.ToInt = 0
) -> onp.MArray[np.complex128] | Any: ...
@overload  # Nd T@inexact
def variation[InexactT: npc.inexact](
    a: onp.ToArrayND[InexactT, InexactT], axis: SupportsIndex | None = 0, ddof: onp.ToInt = 0
) -> onp.MArray[InexactT] | Any: ...

#
@overload  # ?d ~f64, axis=None
def skew(a: onp.ToFloat64_ND, axis: None, bias: bool = True) -> onp.MArray0D[np.float64]: ...
@overload  # ?d ~c128, axis=None
def skew(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64], axis: None, bias: bool = True
) -> onp.MArray0D[np.complex128]: ...
@overload  # ?d T@inexact80, axis=None
def skew[InexactT: npc.inexact80](
    a: onp.ToArrayND[InexactT, InexactT], axis: None, bias: bool = True
) -> onp.MArray0D[InexactT]: ...
@overload  # ?d ~f64, axis=<given> (default)
def skew(
    a: onp.ArrayND[_AsF64 | np.float32 | np.float16, _JustAnyShape], axis: SupportsIndex = 0, bias: bool = True
) -> onp.MArray[np.float64] | Any: ...
@overload  # ?d ~c128, axis=<given> (default)
def skew(
    a: onp.ArrayND[np.complex128 | np.complex64, _JustAnyShape], axis: SupportsIndex = 0, bias: bool = True
) -> onp.MArray[np.complex128] | Any: ...
@overload  # ?d T@inexact80, axis=<given> (default)
def skew[InexactT: npc.inexact80](
    a: onp.ArrayND[InexactT, _JustAnyShape], axis: SupportsIndex = 0, bias: bool = True
) -> onp.MArray[InexactT] | Any: ...
@overload  # 1d ~f64, axis=<given> (default)
def skew(a: onp.ToFloat64Strict1D, axis: SupportsIndex = 0, bias: bool = True) -> onp.MArray0D[np.float64]: ...
@overload  # 1d ~c128, axis=<given> (default)
def skew(
    a: onp.ToArrayStrict1D[op.JustComplex, np.complex128 | np.complex64], axis: SupportsIndex = 0, bias: bool = True
) -> onp.MArray0D[np.complex128]: ...
@overload  # 1d T@inexact80, axis=<given> (default)
def skew[InexactT: npc.inexact80](
    a: onp.ToArrayStrict1D[InexactT, InexactT], axis: SupportsIndex = 0, bias: bool = True
) -> onp.MArray0D[InexactT]: ...
@overload  # 2d ~f64, axis=<given> (default)
def skew(a: onp.ToFloat64Strict2D, axis: SupportsIndex = 0, bias: bool = True) -> onp.MArray1D[np.float64]: ...
@overload  # 2d ~c128, axis=<given> (default)
def skew(
    a: onp.ToArrayStrict2D[op.JustComplex, np.complex128 | np.complex64], axis: SupportsIndex = 0, bias: bool = True
) -> onp.MArray1D[np.complex128]: ...
@overload  # 2d T@inexact80, axis=<given> (default)
def skew[InexactT: npc.inexact80](
    a: onp.ToArrayStrict2D[InexactT, InexactT], axis: SupportsIndex = 0, bias: bool = True
) -> onp.MArray1D[InexactT]: ...
@overload  # 3d ~f64, axis=<given> (default)
def skew(a: onp.ToFloat64Strict3D, axis: SupportsIndex = 0, bias: bool = True) -> onp.MArray2D[np.float64]: ...
@overload  # 3d ~c128, axis=<given> (default)
def skew(
    a: onp.ToArrayStrict3D[op.JustComplex, np.complex128 | np.complex64], axis: SupportsIndex = 0, bias: bool = True
) -> onp.MArray2D[np.complex128]: ...
@overload  # 3d T@inexact80, axis=<given> (default)
def skew[InexactT: npc.inexact80](
    a: onp.ToArrayStrict3D[InexactT, InexactT], axis: SupportsIndex = 0, bias: bool = True
) -> onp.MArray2D[InexactT]: ...
@overload  # Nd ~f64
def skew(a: onp.ToFloat64_ND, axis: SupportsIndex | None = 0, bias: bool = True) -> onp.MArray[np.float64] | Any: ...
@overload  # Nd ~c128
def skew(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64], axis: SupportsIndex | None = 0, bias: bool = True
) -> onp.MArray[np.complex128] | Any: ...
@overload  # Nd T@inexact80
def skew[InexactT: npc.inexact80](
    a: onp.ToArrayND[InexactT, InexactT], axis: SupportsIndex | None = 0, bias: bool = True
) -> onp.MArray[InexactT] | Any: ...

#
@overload  # ?d ~f64, axis=None, fisher=True
def kurtosis(a: onp.ToFloat64_ND, axis: None, fisher: Literal[True] = True, bias: bool = True) -> np.float64: ...
@overload  # ?d ~c128, axis=None, fisher=True
def kurtosis(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64], axis: None, fisher: Literal[True] = True, bias: bool = True
) -> np.complex128: ...
@overload  # ?d T@inexact80, axis=None, fisher=True
def kurtosis[InexactT: npc.inexact80](
    a: onp.ToArrayND[InexactT, InexactT], axis: None, fisher: Literal[True] = True, bias: bool = True
) -> InexactT: ...
@overload  # ?d ~f64, axis=<given> (default)
def kurtosis(
    a: onp.ArrayND[_AsF64 | np.float32 | np.float16, _JustAnyShape],
    axis: SupportsIndex = 0,
    fisher: bool = True,
    bias: bool = True,
) -> onp.MArray[np.float64] | Any: ...
@overload  # ?d ~c128, axis=<given> (default)
def kurtosis(
    a: onp.ArrayND[np.complex128 | np.complex64, _JustAnyShape], axis: SupportsIndex = 0, fisher: bool = True, bias: bool = True
) -> onp.MArray[np.complex128] | Any: ...
@overload  # ?d T@inexact80, axis=<given> (default)
def kurtosis[InexactT: npc.inexact80](
    a: onp.ArrayND[InexactT, _JustAnyShape], axis: SupportsIndex = 0, fisher: bool = True, bias: bool = True
) -> onp.MArray[InexactT] | Any: ...
@overload  # 1d ~f64, fisher=True (default)
def kurtosis(
    a: onp.ToFloat64Strict1D, axis: SupportsIndex = 0, fisher: Literal[True] = True, bias: bool = True
) -> np.float64: ...
@overload  # 1d ~c128, fisher=True (default)
def kurtosis(
    a: onp.ToArrayStrict1D[op.JustComplex, np.complex128 | np.complex64],
    axis: SupportsIndex = 0,
    fisher: Literal[True] = True,
    bias: bool = True,
) -> np.complex128: ...
@overload  # 1d T@inexact80, fisher=True (default)
def kurtosis[InexactT: npc.inexact80](
    a: onp.ToArrayStrict1D[InexactT, InexactT], axis: SupportsIndex = 0, fisher: Literal[True] = True, bias: bool = True
) -> InexactT: ...
@overload  # 2d ~f64, axis=<given> (default)
def kurtosis(
    a: onp.ToFloat64Strict2D, axis: SupportsIndex = 0, fisher: bool = True, bias: bool = True
) -> onp.MArray1D[np.float64]: ...
@overload  # 2d ~c128, axis=<given> (default)
def kurtosis(
    a: onp.ToArrayStrict2D[op.JustComplex, np.complex128 | np.complex64],
    axis: SupportsIndex = 0,
    fisher: bool = True,
    bias: bool = True,
) -> onp.MArray1D[np.complex128]: ...
@overload  # 2d T@inexact80, axis=<given> (default)
def kurtosis[InexactT: npc.inexact80](
    a: onp.ToArrayStrict2D[InexactT, InexactT], axis: SupportsIndex = 0, fisher: bool = True, bias: bool = True
) -> onp.MArray1D[InexactT]: ...
@overload  # 3d ~f64, axis=<given> (default)
def kurtosis(
    a: onp.ToFloat64Strict3D, axis: SupportsIndex = 0, fisher: bool = True, bias: bool = True
) -> onp.MArray2D[np.float64]: ...
@overload  # 3d ~c128, axis=<given> (default)
def kurtosis(
    a: onp.ToArrayStrict3D[op.JustComplex, np.complex128 | np.complex64],
    axis: SupportsIndex = 0,
    fisher: bool = True,
    bias: bool = True,
) -> onp.MArray2D[np.complex128]: ...
@overload  # 3d T@inexact80, axis=<given> (default)
def kurtosis[InexactT: npc.inexact80](
    a: onp.ToArrayStrict3D[InexactT, InexactT], axis: SupportsIndex = 0, fisher: bool = True, bias: bool = True
) -> onp.MArray2D[InexactT]: ...
@overload  # Nd ~f64
def kurtosis(
    a: onp.ToFloat64_ND, axis: SupportsIndex | None = 0, fisher: bool = True, bias: bool = True
) -> onp.MArray[np.float64] | Any: ...
@overload  # Nd ~c128
def kurtosis(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64],
    axis: SupportsIndex | None = 0,
    fisher: bool = True,
    bias: bool = True,
) -> onp.MArray[np.complex128] | Any: ...
@overload  # Nd T@inexact80
def kurtosis[InexactT: npc.inexact80](
    a: onp.ToArrayND[InexactT, InexactT], axis: SupportsIndex | None = 0, fisher: bool = True, bias: bool = True
) -> onp.MArray[InexactT] | Any: ...

#
@overload  # ?d bool, axis=None
def describe(
    a: onp.ToArrayND[bool, np.bool], axis: None, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[np.bool, np.float64, np.float64, np.float64]: ...
@overload  # ?d T@integer, axis=None
def describe[ScalarT: npc.integer](
    a: onp.ToArrayND[ScalarT, ScalarT], axis: None, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[ScalarT, np.float64, np.float64, np.float64]: ...
@overload  # ?d ~int, axis=None
def describe(
    a: onp.ToJustInt64_ND, axis: None, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[np.int_, np.float64, np.float64, np.float64]: ...
@overload  # ?d T@float32, axis=None
def describe[ScalarT: np.float32 | np.float16](
    a: onp.ToArrayND[ScalarT, ScalarT], axis: None, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[ScalarT, ScalarT, ScalarT, np.float64]: ...
@overload  # ?d ~f64, axis=None
def describe(
    a: onp.ToJustFloat64_ND, axis: None, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[np.float64, np.float64, np.float64, np.float64]: ...
@overload  # ?d T@inexact80, axis=None
def describe[ScalarT: npc.inexact80](
    a: onp.ToArrayND[ScalarT, ScalarT], axis: None, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[ScalarT, ScalarT, np.longdouble, ScalarT]: ...
@overload  # ?d c64, axis=None
def describe(
    a: onp.ToJustComplex64_ND, axis: None, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[np.complex64, np.complex64, np.float32, np.complex128]: ...
@overload  # ?d ~c128, axis=None
def describe(
    a: onp.ToJustComplex128_ND, axis: None, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[np.complex128, np.complex128, np.float64, np.complex128]: ...
@overload  # ?d bool, axis=<given> (default)
def describe(
    a: onp.ArrayND[np.bool, _JustAnyShape], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[np.bool, np.float64, np.float64, np.float64]: ...
@overload  # ?d T@integer, axis=<given> (default)
def describe[ScalarT: npc.integer](
    a: onp.ArrayND[ScalarT, _JustAnyShape], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[ScalarT, np.float64, np.float64, np.float64]: ...
@overload  # ?d T@float32, axis=<given> (default)
def describe[ScalarT: np.float32 | np.float16](
    a: onp.ArrayND[ScalarT, _JustAnyShape], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[ScalarT, ScalarT, ScalarT, np.float64]: ...
@overload  # ?d ~f64, axis=<given> (default)
def describe(
    a: onp.ArrayND[np.float64, _JustAnyShape], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[np.float64, np.float64, np.float64, np.float64]: ...
@overload  # ?d T@inexact80, axis=<given> (default)
def describe[ScalarT: npc.inexact80](
    a: onp.ArrayND[ScalarT, _JustAnyShape], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[ScalarT, ScalarT, np.longdouble, ScalarT]: ...
@overload  # ?d c64, axis=<given> (default)
def describe(
    a: onp.ArrayND[np.complex64, _JustAnyShape], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[np.complex64, np.complex64, np.float32, np.complex128]: ...
@overload  # ?d ~c128, axis=<given> (default)
def describe(
    a: onp.ArrayND[np.complex128, _JustAnyShape], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[np.complex128, np.complex128, np.float64, np.complex128]: ...
@overload  # 1d bool, axis=<given> (default)
def describe(
    a: onp.ToArrayStrict1D[bool, np.bool], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[np.bool, np.float64, np.float64, np.float64]: ...
@overload  # 1d T@integer, axis=<given> (default)
def describe[ScalarT: npc.integer](
    a: onp.ToArrayStrict1D[ScalarT, ScalarT], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[ScalarT, np.float64, np.float64, np.float64]: ...
@overload  # 1d ~int, axis=<given> (default)
def describe(
    a: onp.ToJustInt64Strict1D, axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[np.int_, np.float64, np.float64, np.float64]: ...
@overload  # 1d T@float32, axis=<given> (default)
def describe[ScalarT: np.float32 | np.float16](
    a: onp.ToArrayStrict1D[ScalarT, ScalarT], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[ScalarT, ScalarT, ScalarT, np.float64]: ...
@overload  # 1d ~f64, axis=<given> (default)
def describe(
    a: onp.ToJustFloat64Strict1D, axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[np.float64, np.float64, np.float64, np.float64]: ...
@overload  # 1d T@inexact80, axis=<given> (default)
def describe[ScalarT: npc.inexact80](
    a: onp.ToArrayStrict1D[ScalarT, ScalarT], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[ScalarT, ScalarT, np.longdouble, ScalarT]: ...
@overload  # 1d c64, axis=<given> (default)
def describe(
    a: onp.ToJustComplex64Strict1D, axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[np.complex64, np.complex64, np.float32, np.complex128]: ...
@overload  # 1d ~c128, axis=<given> (default)
def describe(
    a: onp.ToJustComplex128Strict1D, axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe0D[np.complex128, np.complex128, np.float64, np.complex128]: ...
@overload  # 2d bool, axis=<given> (default)
def describe(
    a: onp.ToArrayStrict2D[bool, np.bool], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe1D[np.bool, np.float64, np.float64, np.float64]: ...
@overload  # 2d T@integer, axis=<given> (default)
def describe[ScalarT: npc.integer](
    a: onp.ToArrayStrict2D[ScalarT, ScalarT], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe1D[ScalarT, np.float64, np.float64, np.float64]: ...
@overload  # 2d ~int, axis=<given> (default)
def describe(
    a: onp.ToJustInt64Strict2D, axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe1D[np.int_, np.float64, np.float64, np.float64]: ...
@overload  # 2d T@float32, axis=<given> (default)
def describe[ScalarT: np.float32 | np.float16](
    a: onp.ToArrayStrict2D[ScalarT, ScalarT], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe1D[ScalarT, ScalarT, ScalarT, np.float64]: ...
@overload  # 2d ~f64, axis=<given> (default)
def describe(
    a: onp.ToJustFloat64Strict2D, axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe1D[np.float64, np.float64, np.float64, np.float64]: ...
@overload  # 2d T@inexact80, axis=<given> (default)
def describe[ScalarT: npc.inexact80](
    a: onp.ToArrayStrict2D[ScalarT, ScalarT], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe1D[ScalarT, ScalarT, np.longdouble, ScalarT]: ...
@overload  # 2d c64, axis=<given> (default)
def describe(
    a: onp.ToJustComplex64Strict2D, axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe1D[np.complex64, np.complex64, np.float32, np.complex128]: ...
@overload  # 2d ~c128, axis=<given> (default)
def describe(
    a: onp.ToJustComplex128Strict2D, axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _Describe1D[np.complex128, np.complex128, np.float64, np.complex128]: ...
@overload  # Nd bool
def describe(
    a: onp.ToArrayND[bool, np.bool], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[np.bool, np.float64, np.float64, np.float64]: ...
@overload  # Nd T@integer
def describe[ScalarT: npc.integer](
    a: onp.ToArrayND[ScalarT, ScalarT], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[ScalarT, np.float64, np.float64, np.float64]: ...
@overload  # Nd ~int
def describe(
    a: onp.ToJustInt64_ND, axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[np.int_, np.float64, np.float64, np.float64]: ...
@overload  # Nd T@float32
def describe[ScalarT: np.float32 | np.float16](
    a: onp.ToArrayND[ScalarT, ScalarT], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[ScalarT, ScalarT, ScalarT, np.float64]: ...
@overload  # Nd ~f64
def describe(
    a: onp.ToJustFloat64_ND, axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[np.float64, np.float64, np.float64, np.float64]: ...
@overload  # Nd T@inexact80
def describe[ScalarT: npc.inexact80](
    a: onp.ToArrayND[ScalarT, ScalarT], axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[ScalarT, ScalarT, np.longdouble, ScalarT]: ...
@overload  # Nd c64
def describe(
    a: onp.ToJustComplex64_ND, axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[np.complex64, np.complex64, np.float32, np.complex128]: ...
@overload  # Nd ~c128
def describe(
    a: onp.ToJustComplex128_ND, axis: SupportsIndex = 0, ddof: onp.ToInt = 0, bias: bool = True
) -> _DescribeND[np.complex128, np.complex128, np.float64, np.complex128]: ...

#
@overload
def stde_median(data: onp.ToFloatND, axis: SupportsIndex | None = None) -> _MArrayOrND[npc.floating]: ...
@overload
def stde_median(data: onp.ToComplexND, axis: SupportsIndex | None = None) -> _MArrayOrND[npc.inexact]: ...

#
@overload
def skewtest(
    a: onp.ToFloatND, axis: SupportsIndex | None = 0, alternative: Alternative = "two-sided"
) -> SkewtestResult[_MArrayOrND[np.float64], _MArrayOrND[np.float64]]: ...
@overload
def skewtest(
    a: onp.ToComplexND, axis: SupportsIndex | None = 0, alternative: Alternative = "two-sided"
) -> SkewtestResult[_MArrayOrND[np.float64], _MArrayOrND[np.float64 | np.complex128]]: ...

#
@overload
def kurtosistest(
    a: onp.ToFloatND, axis: SupportsIndex | None = 0, alternative: Alternative = "two-sided"
) -> KurtosistestResult[_MArrayOrND[np.float64], _MArrayOrND[np.float64]]: ...
@overload
def kurtosistest(
    a: onp.ToComplexND, axis: SupportsIndex | None = 0, alternative: Alternative = "two-sided"
) -> KurtosistestResult[_MArrayOrND[np.float64], _MArrayOrND[np.float64 | np.complex128]]: ...

#
@overload  # ?d, axis=None
def normaltest(a: onp.ToFloatND, axis: None) -> NormaltestResult[np.float64, np.float64]: ...
@overload  # ?d, axis=<given> (default)
def normaltest(
    a: _ToFloatStrictND, axis: SupportsIndex = 0
) -> NormaltestResult[onp.ArrayND[np.float64] | Any, onp.MArray[np.float64] | Any]: ...
@overload  # 1d, axis=<given> (default)
def normaltest(a: onp.ToFloatStrict1D, axis: SupportsIndex = 0) -> NormaltestResult[np.float64, np.float64]: ...
@overload  # 2d, axis=<given> (default)
def normaltest(
    a: onp.ToFloatStrict2D, axis: SupportsIndex = 0
) -> NormaltestResult[onp.Array1D[np.float64], onp.MArray1D[np.float64]]: ...
@overload  # 3d, axis=<given> (default)
def normaltest(
    a: onp.ToFloatStrict3D, axis: SupportsIndex = 0
) -> NormaltestResult[onp.Array2D[np.float64], onp.MArray2D[np.float64]]: ...
@overload  # fallback
def normaltest(
    a: onp.ToFloatND, axis: SupportsIndex | None = 0
) -> NormaltestResult[onp.ArrayND[np.float64] | Any, onp.MArray[np.float64] | Any]: ...

#
def mquantiles(
    a: onp.ToFloatND,
    prob: onp.ToFloatND = (0.25, 0.5, 0.75),
    alphap: onp.ToFloat = 0.4,
    betap: onp.ToFloat = 0.4,
    axis: SupportsIndex | None = None,
    limit: tuple[onp.ToFloat, onp.ToFloat] | tuple[()] = (),
) -> onp.MArray[np.float64]: ...

#
def scoreatpercentile(
    data: onp.ToFloatND,
    per: onp.ToFloat,
    limit: tuple[onp.ToFloat, onp.ToFloat] | tuple[()] = (),
    alphap: onp.ToFloat = 0.4,
    betap: onp.ToFloat = 0.4,
) -> onp.MArray[np.float64]: ...

#
def plotting_positions(data: onp.ToFloatND, alpha: onp.ToFloat = 0.4, beta: onp.ToFloat = 0.4) -> onp.MArray[np.float64]: ...

#
def obrientransform(arg0: onp.ToFloatND, /, *args: onp.ToFloatND) -> onp.MArray[np.float64]: ...

#
def sem(a: onp.ToFloatND, axis: SupportsIndex | None = 0, ddof: onp.ToInt = 1) -> np.float64 | onp.MArray[np.float64]: ...

#
def f_oneway(arg0: onp.ToFloatND, arg1: onp.ToFloatND, /, *args: onp.ToFloatND) -> F_onewayResult: ...

#
def friedmanchisquare(arg0: onp.ToFloatND, *args: onp.ToFloatND) -> FriedmanchisquareResult: ...

#
def brunnermunzel(
    x: onp.ToFloatND, y: onp.ToFloatND, alternative: Alternative = "two-sided", distribution: Literal["t", "normal"] = "t"
) -> BrunnerMunzelResult: ...

#
ttest_onesamp = ttest_1samp
kruskalwallis = kruskal
ks_twosamp = ks_2samp
trim1 = trimtail
meppf = plotting_positions
