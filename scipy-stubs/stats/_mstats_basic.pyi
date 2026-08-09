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
from ._stats_py import KstestResult, LinregressResult, PearsonRResult, SignificanceResult
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
type _ToJustF64 = np.float64 | np.float32 | np.float16

type _ToLimits = onp.ToJustFloat64 | onp.ToFloat1D | tuple[onp.ToFloat | None, onp.ToFloat | None] | None
type _ToInclusive = tuple[op.CanBool, op.CanBool]
type _ToMinMax = tuple[onp.ToComplex | None, onp.ToComplex | None] | None

type _JustAnyShape = tuple[Never, Never, Never, Never]  # workaround for https://github.com/microsoft/pyright/issues/10232
type _ToFloatStrictND = onp.ArrayND[npc.floating | npc.integer | np.bool, _JustAnyShape]
type _ToComplexStrictND = onp.ArrayND[npc.number | np.bool, _JustAnyShape]
type _ToComplex128StrictND = onp.ArrayND[npc.inexact64 | npc.inexact32 | np.float16 | npc.integer | np.bool, _JustAnyShape]

# workaround for a strange bug in pyright's overlapping overload detection with `numpy<2.1`
type _WorkaroundForPyright = tuple[int] | tuple[Any, ...]

type _KendallTauMethod = Literal["auto", "asymptotic", "exact"]
type _TheilSlopesMethod = Literal["joint", "separate"]
type _SiegelSlopesMethod = Literal["hierarchical", "separate"]

type _KSMethod = Literal["auto", "exact", "asymp"]
type _KTestMethod = Literal[_KSMethod, "approx"]
type _ToCDF = str | Callable[[onp.ArrayND[np.float64]], onp.ToFloatND]

# we can't use a generic shape-type here due to a variance bug in pyright
type _KstestResult0 = KstestResult[np.float64, np.int8]
type _KstestResult1 = KstestResult[onp.Array1D[np.float64], onp.Array1D[np.int8]]
type _KstestResultAny = KstestResult[np.float64 | Any, np.int8 | Any]

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

_SlopeT_co = TypeVar("_SlopeT_co", covariant=True, bound=npc.inexact, default=Any)
_ModeT_co = TypeVar("_ModeT_co", covariant=True, bound=onp.ArrayND[Any], default=onp.ArrayND[np.float64 | Any])

@type_check_only
class _TestResult(NamedTuple, Generic[_NDT_f_co, _NDT_fc_co]):
    statistic: _NDT_fc_co
    pvalue: _NDT_f_co

_KendallTauSeasonalResult = TypedDict(
    "_KendallTauSeasonalResult",
    {
        "seasonal tau": onp.MArray1D[np.float64],
        "global tau": np.float64,
        "global tau (alt)": np.float64,
        "seasonal p-value": onp.Array1D[np.float64],
        "global p-value (indep)": np.float64,
        "global p-value (dep)": np.float64,
        "chi2 total": np.float64,
        "chi2 trend": np.float64,
    },
)

###

trimdoc: Final[str] = ...

class ModeResult(NamedTuple, Generic[_ModeT_co]):
    mode: _ModeT_co
    count: _ModeT_co  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]

class DescribeResult(NamedTuple, Generic[_ShapeT_co, _MinMaxT_co, _MeanT_co, _VarT_co, _SkewT_co, _KurtT_co]):
    nobs: onp.Array[_ShapeT_co, np.int_]
    minmax: tuple[onp.MArray[_MinMaxT_co, _ShapeT_co], onp.MArray[_MinMaxT_co, _ShapeT_co]]
    mean: _MeanT_co
    variance: _VarT_co
    skewness: onp.MArray[_SkewT_co, _ShapeT_co]
    kurtosis: _KurtT_co

class PointbiserialrResult(NamedTuple):
    correlation: np.float64
    pvalue: onp.MArray0D[np.float64]

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

class SenSeasonalSlopesResult(  # zuban: ignore[type-var]
    BaseBunch[onp.MArray1D[_SlopeT_co], _SlopeT_co], Generic[_SlopeT_co]
):
    @override
    def __new__(_cls, intra_slope: onp.MArray1D[_SlopeT_co], inter_slope: _SlopeT_co) -> Self: ...  # pyrefly:ignore[bad-override]
    @override
    def __init__(self, /, intra_slope: onp.MArray1D[_SlopeT_co], inter_slope: _SlopeT_co) -> None: ...  # pyrefly:ignore[bad-override]

    #
    @property
    def intra_slope(self, /) -> onp.MArray1D[_SlopeT_co]: ...
    @property
    def inter_slope(self, /) -> _SlopeT_co: ...

# TODO(jorenham): Overloads for scalar vs. array
# TODO(jorenham): Overloads for specific dtypes

def argstoarray(*args: onp.ToFloatND) -> onp.MArray[np.float64]: ...
def find_repeats(arr: onp.ToFloatND) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.intp]]: ...
def count_tied_groups(x: onp.ToFloatND, use_missing: bool = False) -> dict[np.intp, np.intp | int]: ...

#
@overload  # ?d
def rankdata(
    data: _ToComplexStrictND, axis: SupportsIndex | None = None, use_missing: bool = False
) -> onp.ArrayND[np.float64]: ...
@overload  # 1d
def rankdata(
    data: onp.ToComplexStrict1D, axis: SupportsIndex | None = None, use_missing: bool = False
) -> onp.Array1D[np.float64]: ...
@overload  # 2d
def rankdata(
    data: onp.ToComplexStrict2D, axis: SupportsIndex | None = None, use_missing: bool = False
) -> onp.Array2D[np.float64]: ...
@overload  # 3d
def rankdata(
    data: onp.ToComplexStrict3D, axis: SupportsIndex | None = None, use_missing: bool = False
) -> onp.Array3D[np.float64]: ...
@overload  # Nd
def rankdata(data: onp.ToComplexND, axis: SupportsIndex | None = None, use_missing: bool = False) -> onp.ArrayND[np.float64]: ...

#
@overload  # ?d ~f64, axis=None
def mode(a: onp.ToArrayND[op.JustFloat, _ToJustF64], axis: None) -> ModeResult[onp.Array1D[np.float64]]: ...
@overload  # ?d, axis=None
def mode(a: onp.ToFloatND, axis: None) -> ModeResult[onp.Array1D[np.float64 | Any]]: ...
@overload  # ?d, axis=<given> (default)
def mode(a: _ToFloatStrictND, axis: SupportsIndex = 0) -> ModeResult[onp.ArrayND[np.float64 | Any]]: ...
@overload  # 1d ~f64, axis=<given> (default)
def mode(a: onp.ToArrayStrict1D[op.JustFloat, _ToJustF64], axis: SupportsIndex = 0) -> ModeResult[onp.Array1D[np.float64]]: ...
@overload  # 1d, axis=<given> (default)
def mode(a: onp.ToFloatStrict1D, axis: SupportsIndex = 0) -> ModeResult[onp.Array1D[np.float64 | Any]]: ...
@overload  # 2d ~f64, axis=<given> (default)
def mode(a: onp.ToArrayStrict2D[op.JustFloat, _ToJustF64], axis: SupportsIndex = 0) -> ModeResult[onp.MArray2D[np.float64]]: ...
@overload  # 2d, axis=<given> (default)
def mode(a: onp.ToFloatStrict2D, axis: SupportsIndex = 0) -> ModeResult[onp.MArray2D[np.float64 | Any]]: ...
@overload  # 3d ~f64, axis=<given> (default)
def mode(a: onp.ToArrayStrict3D[op.JustFloat, _ToJustF64], axis: SupportsIndex = 0) -> ModeResult[onp.MArray3D[np.float64]]: ...
@overload  # 3d, axis=<given> (default)
def mode(a: onp.ToFloatStrict3D, axis: SupportsIndex = 0) -> ModeResult[onp.MArray3D[np.float64 | Any]]: ...
@overload  # fallback
def mode(a: onp.ToFloatND, axis: SupportsIndex | None = 0) -> ModeResult[onp.ArrayND[np.float64 | Any]]: ...

#
@overload
def msign[ScalarT: npc.number | np.timedelta64 | np.bool | np.object_](x: _ArrayLike[ScalarT]) -> onp.ArrayND[ScalarT]: ...
@overload
def msign(x: onp.ToComplexND) -> onp.ArrayND[npc.number | np.timedelta64 | np.bool | np.object_]: ...

# NOTE: flattens input
@overload  # ~f64 | +integer, +floating
def pearsonr(x: onp.ToJustFloat64_ND | onp.ToIntND, y: onp.ToFloatND) -> PearsonRResult[np.float64, np.float64]: ...
@overload  # +floating, ~f64 | +integer
def pearsonr(x: onp.ToFloatND, y: onp.ToJustFloat64_ND | onp.ToIntND) -> PearsonRResult[np.float64, np.float64]: ...
@overload  # ~f32, +float32
def pearsonr(x: onp.ToJustFloat32_ND, y: onp.ToFloat32_ND) -> PearsonRResult[np.float32, np.float64]: ...
@overload  # +float32, ~f32
def pearsonr(x: onp.ToFloat32_ND, y: onp.ToJustFloat32_ND) -> PearsonRResult[np.float32, np.float64]: ...
@overload  # +floating, +floating
def pearsonr(x: onp.ToFloatND, y: onp.ToFloatND) -> PearsonRResult[npc.floating, np.float64]: ...

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

# NOTE: flattens input
def kendalltau(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    use_ties: bool = True,
    use_missing: bool = False,
    method: _KendallTauMethod = "auto",
    alternative: Alternative = "two-sided",
) -> SignificanceResult[np.float64]: ...

#
def kendalltau_seasonal(x: onp.ToFloatND) -> _KendallTauSeasonalResult: ...

#
def pointbiserialr(x: onp.ToFloatND, y: onp.ToFloatND) -> PointbiserialrResult: ...

# NOTE: flattens input
def linregress(x: onp.ToFloatND, y: onp.ToFloatND | None = None) -> LinregressResult[np.float64]: ...

# NOTE: flattens input
def theilslopes(
    y: onp.ToFloatND, x: onp.ToFloatND | None = None, alpha: float | npc.floating = 0.95, method: _TheilSlopesMethod = "separate"
) -> TheilslopesResult[np.float64]: ...

# NOTE: flattens input
def siegelslopes(
    y: onp.ToFloatND, x: onp.ToFloatND | None = None, method: _SiegelSlopesMethod = "hierarchical"
) -> SiegelslopesResult[np.float64]: ...

#
@overload  # ~f64
def sen_seasonal_slopes(x: onp.ToFloat64_ND) -> SenSeasonalSlopesResult[np.float64]: ...
@overload  # ~c128
def sen_seasonal_slopes(x: onp.ToJustComplex128_ND | onp.ToJustComplex64_ND) -> SenSeasonalSlopesResult[np.complex128]: ...
@overload  # T@inexact80
def sen_seasonal_slopes[ScalarT: npc.inexact80](x: onp.ToArrayND[ScalarT, ScalarT]) -> SenSeasonalSlopesResult[ScalarT]: ...

#
@overload  # ?d, axis=None
def ttest_1samp(
    a: onp.ToFloatND, popmean: onp.ToFloat | onp.ToFloatND, axis: None, alternative: Alternative = "two-sided"
) -> Ttest_1sampResult[np.float64, np.float64]: ...
@overload  # ?d ~c128, axis=None
def ttest_1samp(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64],
    popmean: onp.ToComplex | onp.ToComplexND,
    axis: None,
    alternative: Alternative = "two-sided",
) -> Ttest_1sampResult[np.float64, np.complex128]: ...
@overload  # ?d, axis=<given> (default)
def ttest_1samp(
    a: _ToFloatStrictND, popmean: onp.ToFloat | onp.ToFloatND, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> Ttest_1sampResult[onp.MArray[np.float64] | Any, onp.MArray[np.float64] | Any]: ...
@overload  # ?d ~c128, axis=<given> (default)
def ttest_1samp(
    a: onp.ArrayND[np.complex128 | np.complex64, _JustAnyShape],
    popmean: onp.ToComplex | onp.ToComplexND,
    axis: SupportsIndex = 0,
    alternative: Alternative = "two-sided",
) -> Ttest_1sampResult[onp.MArray[np.float64] | Any, onp.MArray[np.complex128] | Any]: ...
@overload  # 1d, axis=<given> (default)
def ttest_1samp(
    a: onp.ToFloatStrict1D, popmean: onp.ToFloat | onp.ToFloatND, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> Ttest_1sampResult[np.float64, np.float64]: ...
@overload  # 1d ~c128, axis=<given> (default)
def ttest_1samp(
    a: onp.ToArrayStrict1D[op.JustComplex, np.complex128 | np.complex64],
    popmean: onp.ToComplex | onp.ToComplexND,
    axis: SupportsIndex = 0,
    alternative: Alternative = "two-sided",
) -> Ttest_1sampResult[np.float64, np.complex128]: ...
@overload  # 2d, axis=<given> (default)
def ttest_1samp(
    a: onp.ToFloatStrict2D, popmean: onp.ToFloat | onp.ToFloatND, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> Ttest_1sampResult[onp.MArray1D[np.float64], onp.MArray1D[np.float64]]: ...
@overload  # 2d ~c128, axis=<given> (default)
def ttest_1samp(
    a: onp.ToArrayStrict2D[op.JustComplex, np.complex128 | np.complex64],
    popmean: onp.ToComplex | onp.ToComplexND,
    axis: SupportsIndex = 0,
    alternative: Alternative = "two-sided",
) -> Ttest_1sampResult[onp.MArray1D[np.float64], onp.MArray1D[np.complex128]]: ...
@overload  # 3d, axis=<given> (default)
def ttest_1samp(
    a: onp.ToFloatStrict3D, popmean: onp.ToFloat | onp.ToFloatND, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> Ttest_1sampResult[onp.MArray2D[np.float64], onp.MArray2D[np.float64]]: ...
@overload  # 3d ~c128, axis=<given> (default)
def ttest_1samp(
    a: onp.ToArrayStrict3D[op.JustComplex, np.complex128 | np.complex64],
    popmean: onp.ToComplex | onp.ToComplexND,
    axis: SupportsIndex = 0,
    alternative: Alternative = "two-sided",
) -> Ttest_1sampResult[onp.MArray2D[np.float64], onp.MArray2D[np.complex128]]: ...
@overload  # fallback
def ttest_1samp(
    a: onp.ToComplexND,
    popmean: onp.ToComplex | onp.ToComplexND,
    axis: SupportsIndex | None = 0,
    alternative: Alternative = "two-sided",
) -> Ttest_1sampResult[onp.MArray[np.float64] | Any, onp.MArray[np.float64] | Any]: ...

#
@overload  # ?d, ?d, axis=None
def ttest_ind(
    a: onp.ToFloatND, b: onp.ToFloatND, axis: None, equal_var: bool = True, alternative: Alternative = "two-sided"
) -> Ttest_indResult[np.float64, np.float64]: ...
@overload  # ?d ~c128, ?d, axis=None
def ttest_ind(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64],
    b: onp.ToComplexND,
    axis: None,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult[np.float64, np.complex128]: ...
@overload  # ?d, ?d ~c128, axis=None
def ttest_ind(
    a: onp.ToComplexND,
    b: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64],
    axis: None,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult[np.float64, np.complex128]: ...
@overload  # ?d, ?d, axis=<given> (default)
def ttest_ind(
    a: _ToFloatStrictND,
    b: _ToFloatStrictND,
    axis: SupportsIndex = 0,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult[onp.MArray[np.float64] | Any, onp.MArray[np.float64] | Any]: ...
@overload  # ?d ~c128, ?d, axis=<given> (default)
def ttest_ind(
    a: onp.ArrayND[np.complex128 | np.complex64, _JustAnyShape],
    b: _ToComplexStrictND,
    axis: SupportsIndex = 0,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult[onp.MArray[np.float64] | Any, onp.MArray[np.complex128] | Any]: ...
@overload  # ?d, ?d ~c128, axis=<given> (default)
def ttest_ind(
    a: _ToComplexStrictND,
    b: onp.ArrayND[np.complex128 | np.complex64, _JustAnyShape],
    axis: SupportsIndex = 0,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult[onp.MArray[np.float64] | Any, onp.MArray[np.complex128] | Any]: ...
@overload  # 1d, 1d, axis=<given> (default)
def ttest_ind(
    a: onp.ToFloatStrict1D,
    b: onp.ToFloatStrict1D,
    axis: SupportsIndex = 0,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult[np.float64, np.float64]: ...
@overload  # 1d ~c128, 1d, axis=<given> (default)
def ttest_ind(
    a: onp.ToArrayStrict1D[op.JustComplex, np.complex128 | np.complex64],
    b: onp.ToComplexStrict1D,
    axis: SupportsIndex = 0,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult[np.float64, np.complex128]: ...
@overload  # 1d, 1d ~c128, axis=<given> (default)
def ttest_ind(
    a: onp.ToComplexStrict1D,
    b: onp.ToArrayStrict1D[op.JustComplex, np.complex128 | np.complex64],
    axis: SupportsIndex = 0,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult[np.float64, np.complex128]: ...
@overload  # 2d, 2d, axis=<given> (default)
def ttest_ind(
    a: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict2D,
    axis: SupportsIndex = 0,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult[onp.MArray1D[np.float64], onp.MArray1D[np.float64]]: ...
@overload  # 2d ~c128, 2d, axis=<given> (default)
def ttest_ind(
    a: onp.ToArrayStrict2D[op.JustComplex, np.complex128 | np.complex64],
    b: onp.ToComplexStrict2D,
    axis: SupportsIndex = 0,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult[onp.MArray1D[np.float64], onp.MArray1D[np.complex128]]: ...
@overload  # 2d, 2d ~c128, axis=<given> (default)
def ttest_ind(
    a: onp.ToComplexStrict2D,
    b: onp.ToArrayStrict2D[op.JustComplex, np.complex128 | np.complex64],
    axis: SupportsIndex = 0,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult[onp.MArray1D[np.float64], onp.MArray1D[np.complex128]]: ...
@overload  # fallback
def ttest_ind(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    axis: SupportsIndex | None = 0,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult[onp.MArray[np.float64] | Any, onp.MArray[np.float64] | Any]: ...

#
@overload  # ?d, ?d, axis=None
def ttest_rel(
    a: onp.ToFloatND, b: onp.ToFloatND, axis: None, alternative: Alternative = "two-sided"
) -> Ttest_relResult[np.float64, np.float64]: ...
@overload  # ?d, ?d, axis=<given> (default)
def ttest_rel(
    a: _ToFloatStrictND, b: _ToFloatStrictND, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> Ttest_relResult[onp.MArray[np.float64] | Any, onp.MArray[np.float64] | Any]: ...
@overload  # 1d, 1d, axis=<given> (default)
def ttest_rel(
    a: onp.ToFloatStrict1D, b: onp.ToFloatStrict1D, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> Ttest_relResult[np.float64, np.float64]: ...
@overload  # 2d, 2d, axis=<given> (default)
def ttest_rel(
    a: onp.ToFloatStrict2D, b: onp.ToFloatStrict2D, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> Ttest_relResult[onp.MArray1D[np.float64], onp.MArray1D[np.float64]]: ...
@overload  # fallback
def ttest_rel(
    a: onp.ToFloatND, b: onp.ToFloatND, axis: SupportsIndex | None = 0, alternative: Alternative = "two-sided"
) -> Ttest_relResult[onp.MArray[np.float64] | Any, onp.MArray[np.float64] | Any]: ...

#
def mannwhitneyu(x: onp.ToFloatND, y: onp.ToFloatND, use_continuity: bool = True) -> MannwhitneyuResult: ...

#
def kruskal(arg0: onp.ToFloat1D, /, *args: onp.ToFloat1D) -> KruskalResult: ...

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
@overload  # ?d, ?d | 1d
def ks_2samp(
    data1: _ToFloatStrictND,
    data2: _ToFloatStrictND | onp.ToFloatStrict1D,
    alternative: Alternative = "two-sided",
    method: _KSMethod = "auto",
) -> _KstestResultAny: ...
@overload  # ?d | 1d, ?d
def ks_2samp(
    data1: _ToFloatStrictND | onp.ToFloatStrict1D,
    data2: _ToFloatStrictND,
    alternative: Alternative = "two-sided",
    method: _KSMethod = "auto",
) -> _KstestResultAny: ...
@overload  # 1d, 1d
def ks_2samp(
    data1: onp.ToFloatStrict1D, data2: onp.ToFloatStrict1D, alternative: Alternative = "two-sided", method: _KSMethod = "auto"
) -> _KstestResult0: ...
@overload  # 2d, <=2d
def ks_2samp(
    data1: onp.ToFloatStrict2D,
    data2: onp.ToFloatStrict1D | onp.ToFloatStrict2D,
    alternative: Alternative = "two-sided",
    method: _KSMethod = "auto",
) -> _KstestResult1: ...
@overload  # <=2d, 2d
def ks_2samp(
    data1: onp.ToFloatStrict1D | onp.ToFloatStrict2D,
    data2: onp.ToFloatStrict2D,
    alternative: Alternative = "two-sided",
    method: _KSMethod = "auto",
) -> _KstestResult1: ...
@overload  # fallback
def ks_2samp(
    data1: onp.ToFloatND, data2: onp.ToFloatND, alternative: Alternative = "two-sided", method: _KSMethod = "auto"
) -> _KstestResultAny: ...

#
@overload  # ?d, ?d | 1d
def kstest(
    data1: _ToFloatStrictND,
    data2: _ToFloatStrictND | onp.ToFloatStrict1D | _ToCDF,
    args: tuple[()] = (),
    alternative: Alternative = "two-sided",
    method: _KTestMethod = "auto",
) -> _KstestResultAny: ...
@overload  # ?d | 1d, ?d
def kstest(
    data1: _ToFloatStrictND | onp.ToFloatStrict1D,
    data2: _ToFloatStrictND,
    args: tuple[()] = (),
    alternative: Alternative = "two-sided",
    method: _KTestMethod = "auto",
) -> _KstestResultAny: ...
@overload  # 1d, 1d
def kstest(
    data1: onp.ToFloatStrict1D,
    data2: onp.ToFloatStrict1D | _ToCDF,
    args: tuple[()] = (),
    alternative: Alternative = "two-sided",
    method: _KTestMethod = "auto",
) -> _KstestResult0: ...
@overload  # 2d, <=2d
def kstest(
    data1: onp.ToFloatStrict2D,
    data2: onp.ToFloatStrict1D | onp.ToFloatStrict2D | _ToCDF,
    args: tuple[()] = (),
    alternative: Alternative = "two-sided",
    method: _KTestMethod = "auto",
) -> _KstestResult1: ...
@overload  # <=2d, 2d
def kstest(
    data1: onp.ToFloatStrict1D | onp.ToFloatStrict2D,
    data2: onp.ToFloatStrict2D,
    args: tuple[()] = (),
    alternative: Alternative = "two-sided",
    method: _KTestMethod = "auto",
) -> _KstestResult1: ...
@overload  # fallback
def kstest(
    data1: onp.ToFloatND,
    data2: onp.ToFloatND | _ToCDF,
    args: tuple[()] = (),
    alternative: Alternative = "two-sided",
    method: _KTestMethod = "auto",
) -> _KstestResultAny: ...
@overload  # 1d, args=<given>
def kstest(
    data1: onp.ToFloatStrict1D,
    data2: Callable[Concatenate[onp.ArrayND[np.float64], ...], onp.ToFloatND],
    args: tuple[object, ...],
    alternative: Alternative = "two-sided",
    method: _KTestMethod = "auto",
) -> _KstestResult0: ...
@overload  # 2d, args=<given>
def kstest(
    data1: onp.ToFloatStrict2D,
    data2: Callable[Concatenate[onp.ArrayND[np.float64], ...], onp.ToFloatND],
    args: tuple[object, ...],
    alternative: Alternative = "two-sided",
    method: _KTestMethod = "auto",
) -> _KstestResult1: ...
@overload  # fallback, args=<given>
def kstest(
    data1: onp.ToFloatND,
    data2: Callable[Concatenate[onp.ArrayND[np.float64], ...], onp.ToFloatND],
    args: tuple[object, ...],
    alternative: Alternative = "two-sided",
    method: _KTestMethod = "auto",
) -> _KstestResultAny: ...

#
@overload  # 1d bool
def trima(a: list[bool], limits: _ToMinMax = None, inclusive: _ToInclusive = (True, True)) -> onp.MArray1D[np.bool]: ...
@overload  # ?d bool
def trima(
    a: onp.SequenceND[list[bool]], limits: _ToMinMax = None, inclusive: _ToInclusive = (True, True)
) -> onp.MArray[np.bool]: ...
@overload  # 1d ~int
def trima(a: list[int], limits: _ToMinMax = None, inclusive: _ToInclusive = (True, True)) -> onp.MArray1D[np.int_]: ...
@overload  # ?d ~int
def trima(
    a: onp.SequenceND[list[int]], limits: _ToMinMax = None, inclusive: _ToInclusive = (True, True)
) -> onp.MArray[np.int_]: ...
@overload  # 1d ~float
def trima(a: list[float], limits: _ToMinMax = None, inclusive: _ToInclusive = (True, True)) -> onp.MArray1D[np.float64]: ...
@overload  # ?d ~float
def trima(
    a: onp.SequenceND[list[float]], limits: _ToMinMax = None, inclusive: _ToInclusive = (True, True)
) -> onp.MArray[np.float64]: ...
@overload  # 1d ~complex
def trima(a: list[complex], limits: _ToMinMax = None, inclusive: _ToInclusive = (True, True)) -> onp.MArray1D[np.complex128]: ...
@overload  # ?d ~complex
def trima(
    a: onp.SequenceND[list[complex]], limits: _ToMinMax = None, inclusive: _ToInclusive = (True, True)
) -> onp.MArray[np.complex128]: ...
@overload  # ?d T@+number
def trima[ShapeT: tuple[int, ...], ScalarT: npc.number | np.bool](
    a: onp.ArrayND[ScalarT, ShapeT], limits: _ToMinMax = None, inclusive: _ToInclusive = (True, True)
) -> onp.MArray[ScalarT, ShapeT]: ...
@overload  # Nd T@+number
def trima[ScalarT: npc.number | np.bool](
    a: onp.ToArrayND[ScalarT, ScalarT], limits: _ToMinMax = None, inclusive: _ToInclusive = (True, True)
) -> onp.MArray[ScalarT, _WorkaroundForPyright]: ...

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
@overload  # 1d bool
def trimboth(
    data: list[bool],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray1D[np.bool]: ...
@overload  # ?d bool
def trimboth(
    data: onp.SequenceND[list[bool]],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.bool]: ...
@overload  # 1d ~int
def trimboth(
    data: list[int],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray1D[np.int_]: ...
@overload  # ?d ~int
def trimboth(
    data: onp.SequenceND[list[int]],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.int_]: ...
@overload  # 1d ~float
def trimboth(
    data: list[float],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray1D[np.float64]: ...
@overload  # ?d ~float
def trimboth(
    data: onp.SequenceND[list[float]],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.float64]: ...
@overload  # 1d ~complex
def trimboth(
    data: list[complex],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray1D[np.complex128]: ...
@overload  # ?d ~complex
def trimboth(
    data: onp.SequenceND[list[complex]],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.complex128]: ...
@overload  # ?d T@+number
def trimboth[ShapeT: tuple[int, ...], ScalarT: npc.number | np.bool](
    data: onp.ArrayND[ScalarT, ShapeT],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[ScalarT, ShapeT]: ...
@overload  # Nd T@+number
def trimboth[ScalarT: npc.number | np.bool](
    data: onp.ToArrayND[ScalarT, ScalarT],
    proportiontocut: float | npc.floating = 0.2,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[ScalarT, _WorkaroundForPyright]: ...

#
@overload  # 1d bool
def trimtail(
    data: list[bool],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray1D[np.bool]: ...
@overload  # ?d bool
def trimtail(
    data: onp.SequenceND[list[bool]],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.bool]: ...
@overload  # 1d ~int
def trimtail(
    data: list[int],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray1D[np.int_]: ...
@overload  # ?d ~int
def trimtail(
    data: onp.SequenceND[list[int]],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.int_]: ...
@overload  # 1d ~float
def trimtail(
    data: list[float],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray1D[np.float64]: ...
@overload  # ?d ~float
def trimtail(
    data: onp.SequenceND[list[float]],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.float64]: ...
@overload  # 1d ~complex
def trimtail(
    data: list[complex],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray1D[np.complex128]: ...
@overload  # ?d ~complex
def trimtail(
    data: onp.SequenceND[list[complex]],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.complex128]: ...
@overload  # Nd T@+number
def trimtail[ShapeT: tuple[int, ...], ScalarT: npc.number | np.bool](
    data: onp.ArrayND[ScalarT, ShapeT],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[ScalarT, ShapeT]: ...
@overload  # ?d T@+number
def trimtail[ScalarT: npc.number | np.bool](
    data: onp.ToArrayND[ScalarT, ScalarT],
    proportiontocut: float | npc.floating = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = None,
) -> onp.MArray[ScalarT, _WorkaroundForPyright]: ...

#
# NOTE: f32/c64 promotes to f64/c128
@overload  # ?d ~f64, axis=None (default)
def trimmed_mean(
    a: onp.ToArrayND[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: None = None,
) -> np.float64: ...
@overload  # ?d ~c128, axis=None (default)
def trimmed_mean(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: None = None,
) -> np.complex128: ...
@overload  # ?d T@inexact, axis=None (default)
def trimmed_mean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayND[InexactT, InexactT],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: None = None,
) -> InexactT: ...
@overload  # ?d ~f64, axis=<given>
def trimmed_mean(
    a: onp.ArrayND[_AsF64 | np.float32, _JustAnyShape],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray[np.float64] | Any: ...
@overload  # ?d ~c128, axis=<given>
def trimmed_mean(
    a: onp.ArrayND[np.complex128 | np.complex64, _JustAnyShape],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray[np.complex128] | Any: ...
@overload  # ?d T@inexact, axis=<given>
def trimmed_mean[InexactT: npc.inexact80 | np.float16](
    a: onp.ArrayND[InexactT, _JustAnyShape],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray[InexactT] | Any: ...
@overload  # 1d ~f64, axis=<given>
def trimmed_mean(
    a: onp.ToArrayStrict1D[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> np.float64: ...
@overload  # 1d ~c128, axis=<given>
def trimmed_mean(
    a: onp.ToArrayStrict1D[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> np.complex128: ...
@overload  # 1d T@inexact, axis=<given>
def trimmed_mean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayStrict1D[InexactT, InexactT],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> InexactT: ...
@overload  # 2d ~f64, axis=<given>
def trimmed_mean(
    a: onp.ToArrayStrict2D[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray1D[np.float64]: ...
@overload  # 2d ~c128, axis=<given>
def trimmed_mean(
    a: onp.ToArrayStrict2D[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray1D[np.complex128]: ...
@overload  # 2d T@inexact, axis=<given>
def trimmed_mean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayStrict2D[InexactT, InexactT],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray1D[InexactT]: ...
@overload  # 3d ~f64, axis=<given>
def trimmed_mean(
    a: onp.ToArrayStrict3D[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray2D[np.float64]: ...
@overload  # 3d ~c128, axis=<given>
def trimmed_mean(
    a: onp.ToArrayStrict3D[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray2D[np.complex128]: ...
@overload  # 3d T@inexact, axis=<given>
def trimmed_mean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayStrict3D[InexactT, InexactT],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
) -> onp.MArray2D[InexactT]: ...
@overload  # Nd ~f64
def trimmed_mean(
    a: onp.ToArrayND[float, _AsF64 | np.float32],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.float64] | Any: ...
@overload  # Nd ~c128
def trimmed_mean(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.complex128] | Any: ...
@overload  # Nd T@inexact
def trimmed_mean[InexactT: npc.inexact80 | np.float16](
    a: onp.ToArrayND[InexactT, InexactT],
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: SupportsIndex | None = None,
) -> onp.MArray[InexactT] | Any: ...

#
@overload  # ?d ~f64, axis=None (default)
def trimmed_var(
    a: onp.ToComplex128_ND,
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: None = None,
    ddof: onp.ToInt = 0,
) -> np.float64: ...
@overload  # ?d ~f80, axis=None (default)
def trimmed_var(
    a: onp.ToArrayND[npc.inexact80, npc.inexact80],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: None = None,
    ddof: onp.ToInt = 0,
) -> np.longdouble: ...
@overload  # ?d ~f64, axis=<given>
def trimmed_var(
    a: _ToComplex128StrictND,
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
    ddof: onp.ToInt = 0,
) -> onp.MArray[np.float64] | Any: ...
@overload  # ?d ~f80, axis=<given>
def trimmed_var(
    a: onp.ArrayND[npc.inexact80, _JustAnyShape],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
    ddof: onp.ToInt = 0,
) -> onp.MArray[np.longdouble] | Any: ...
@overload  # 1d ~f64, axis=<given>
def trimmed_var(
    a: onp.ToComplex128Strict1D,
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
    ddof: onp.ToInt = 0,
) -> np.float64: ...
@overload  # 1d ~f80, axis=<given>
def trimmed_var(
    a: onp.ToArrayStrict1D[npc.inexact80, npc.inexact80],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
    ddof: onp.ToInt = 0,
) -> np.longdouble: ...
@overload  # 2d ~f64, axis=<given>
def trimmed_var(
    a: onp.ToComplex128Strict2D,
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
    ddof: onp.ToInt = 0,
) -> onp.MArray1D[np.float64]: ...
@overload  # 2d ~f80, axis=<given>
def trimmed_var(
    a: onp.ToArrayStrict2D[npc.inexact80, npc.inexact80],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
    ddof: onp.ToInt = 0,
) -> onp.MArray1D[np.longdouble]: ...
@overload  # Nd ~f64
def trimmed_var(
    a: onp.ToComplex128_ND,
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: SupportsIndex | None = None,
    ddof: onp.ToInt = 0,
) -> onp.MArray[np.float64] | Any: ...
@overload  # Nd ~f80
def trimmed_var(
    a: onp.ToArrayND[npc.inexact80, npc.inexact80],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: SupportsIndex | None = None,
    ddof: onp.ToInt = 0,
) -> onp.MArray[np.longdouble] | Any: ...

#
@overload  # ?d ~f64, axis=None (default)
def trimmed_std(
    a: onp.ToComplex128_ND,
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: None = None,
    ddof: onp.ToInt = 0,
) -> np.float64: ...
@overload  # ?d ~f80, axis=None (default)
def trimmed_std(
    a: onp.ToArrayND[npc.inexact80, npc.inexact80],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: None = None,
    ddof: onp.ToInt = 0,
) -> np.longdouble: ...
@overload  # ?d ~f64, axis=<given>
def trimmed_std(
    a: _ToComplex128StrictND,
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
    ddof: onp.ToInt = 0,
) -> onp.MArray[np.float64] | Any: ...
@overload  # ?d ~f80, axis=<given>
def trimmed_std(
    a: onp.ArrayND[npc.inexact80, _JustAnyShape],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
    ddof: onp.ToInt = 0,
) -> onp.MArray[np.longdouble] | Any: ...
@overload  # 1d ~f64, axis=<given>
def trimmed_std(
    a: onp.ToComplex128Strict1D,
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
    ddof: onp.ToInt = 0,
) -> np.float64: ...
@overload  # 1d ~f80, axis=<given>
def trimmed_std(
    a: onp.ToArrayStrict1D[npc.inexact80, npc.inexact80],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
    ddof: onp.ToInt = 0,
) -> np.longdouble: ...
@overload  # 2d ~f64, axis=<given>
def trimmed_std(
    a: onp.ToComplex128Strict2D,
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
    ddof: onp.ToInt = 0,
) -> onp.MArray1D[np.float64]: ...
@overload  # 2d ~f80, axis=<given>
def trimmed_std(
    a: onp.ToArrayStrict2D[npc.inexact80, npc.inexact80],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    *,
    axis: SupportsIndex,
    ddof: onp.ToInt = 0,
) -> onp.MArray1D[np.longdouble]: ...
@overload  # Nd ~f64
def trimmed_std(
    a: onp.ToComplex128_ND,
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: SupportsIndex | None = None,
    ddof: onp.ToInt = 0,
) -> onp.MArray[np.float64] | Any: ...
@overload  # Nd ~f80
def trimmed_std(
    a: onp.ToArrayND[npc.inexact80, npc.inexact80],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    relative: bool = True,
    axis: SupportsIndex | None = None,
    ddof: onp.ToInt = 0,
) -> onp.MArray[np.longdouble] | Any: ...

#
@overload  # ?d ~f64, axis=None (default)
def trimmed_stde(
    a: onp.ToComplex128_ND, limits: _ToLimits = (0.1, 0.1), inclusive: _ToInclusive = (1, 1), axis: None = None
) -> np.float64: ...
@overload  # ?d ~f80, axis=None (default)
def trimmed_stde(
    a: onp.ToArrayND[npc.inexact80, npc.inexact80],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    axis: None = None,
) -> np.longdouble: ...
@overload  # ?d ~f64, axis=<given>
def trimmed_stde(
    a: _ToComplex128StrictND, limits: _ToLimits = (0.1, 0.1), inclusive: _ToInclusive = (1, 1), *, axis: SupportsIndex
) -> onp.MArray[np.float64] | Any: ...
@overload  # ?d ~f80, axis=<given>
def trimmed_stde(
    a: onp.ArrayND[npc.inexact80, _JustAnyShape],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    *,
    axis: SupportsIndex,
) -> onp.MArray[np.longdouble] | Any: ...
@overload  # 1d ~f64, axis=<given>
def trimmed_stde(
    a: onp.ToComplex128Strict1D, limits: _ToLimits = (0.1, 0.1), inclusive: _ToInclusive = (1, 1), *, axis: SupportsIndex
) -> onp.MArray0D[np.float64]: ...
@overload  # 1d ~f80, axis=<given>
def trimmed_stde(
    a: onp.ToArrayStrict1D[npc.inexact80, npc.inexact80],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    *,
    axis: SupportsIndex,
) -> onp.MArray0D[np.longdouble]: ...
@overload  # 2d ~f64, axis=<given>
def trimmed_stde(
    a: onp.ToComplex128Strict2D, limits: _ToLimits = (0.1, 0.1), inclusive: _ToInclusive = (1, 1), *, axis: SupportsIndex
) -> onp.MArray1D[np.float64]: ...
@overload  # 2d ~f80, axis=<given>
def trimmed_stde(
    a: onp.ToArrayStrict2D[npc.inexact80, npc.inexact80],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    *,
    axis: SupportsIndex,
) -> onp.MArray1D[np.longdouble]: ...
@overload  # Nd ~f64
def trimmed_stde(
    a: onp.ToComplex128_ND, limits: _ToLimits = (0.1, 0.1), inclusive: _ToInclusive = (1, 1), axis: SupportsIndex | None = None
) -> onp.MArray[np.float64] | Any: ...
@overload  # Nd ~f80
def trimmed_stde(
    a: onp.ToArrayND[npc.inexact80, npc.inexact80],
    limits: _ToLimits = (0.1, 0.1),
    inclusive: _ToInclusive = (1, 1),
    axis: SupportsIndex | None = None,
) -> onp.MArray[np.longdouble] | Any: ...

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
@overload  # limits=None (default), requires a mask
def tvar(
    a: onp.MArray[npc.number | np.bool],
    limits: None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = 0,
    ddof: onp.ToInt = 1,
) -> np.float64: ...
@overload  # limits=<given>
def tvar(
    a: onp.ArrayND[npc.number | np.bool],
    limits: tuple[onp.ToFloat | None, onp.ToFloat | None],
    inclusive: tuple[bool, bool] = (True, True),
    axis: SupportsIndex | None = 0,
    ddof: onp.ToInt = 1,
) -> np.float64: ...

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
@overload  # ?d ~f64
def tsem(
    a: onp.ToComplex128_ND,
    limits: tuple[onp.ToFloat | None, onp.ToFloat | None] | None = None,
    inclusive: _ToInclusive = (True, True),
    axis: SupportsIndex | None = 0,
    ddof: onp.ToInt = 1,
) -> np.float64: ...
@overload  # ?d ~f80
def tsem(
    a: onp.ToArrayND[npc.inexact80, npc.inexact80],
    limits: tuple[onp.ToFloat | None, onp.ToFloat | None] | None = None,
    inclusive: _ToInclusive = (True, True),
    axis: SupportsIndex | None = 0,
    ddof: onp.ToInt = 1,
) -> np.longdouble: ...
@overload  # ?d
def tsem(
    a: onp.ToComplexND,
    limits: tuple[onp.ToFloat | None, onp.ToFloat | None] | None = None,
    inclusive: _ToInclusive = (True, True),
    axis: SupportsIndex | None = 0,
    ddof: onp.ToInt = 1,
) -> np.float64 | Any: ...

# NOTE: rejects array-likes: the `nan_policy` check requires `a.shape`
def winsorize[ShapeT: tuple[int, ...], ScalarT: npc.number | np.bool](
    a: onp.ArrayND[ScalarT, ShapeT],
    limits: onp.ToJustFloat64 | tuple[onp.ToFloat | None, onp.ToFloat | None] | onp.ToFloat1D | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    inplace: bool = False,
    axis: SupportsIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> onp.MArray[ScalarT, ShapeT]: ...

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

# undocumented
@overload
def stde_median(data: onp.ToFloatND, axis: SupportsIndex | None = None) -> _MArrayOrND[npc.floating]: ...
@overload
def stde_median(data: onp.ToComplexND, axis: SupportsIndex | None = None) -> _MArrayOrND[npc.inexact]: ...

#
@overload  # ?d, axis=None
def skewtest(a: onp.ToFloatND, axis: None, alternative: Alternative = "two-sided") -> SkewtestResult[np.float64, np.float64]: ...
@overload  # ?d ~c128, axis=None
def skewtest(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64], axis: None, alternative: Alternative = "two-sided"
) -> SkewtestResult[np.float64, np.complex128]: ...
@overload  # ?d, axis=<given> (default)
def skewtest(
    a: _ToFloatStrictND, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> SkewtestResult[onp.ArrayND[np.float64] | Any, onp.MArray[np.float64] | Any]: ...
@overload  # ?d ~c128, axis=<given> (default)
def skewtest(
    a: onp.ArrayND[np.complex128 | np.complex64, _JustAnyShape], axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> SkewtestResult[onp.ArrayND[np.float64] | Any, onp.MArray[np.complex128] | Any]: ...
@overload  # 1d, axis=<given> (default)
def skewtest(
    a: onp.ToFloatStrict1D, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> SkewtestResult[np.float64, np.float64]: ...
@overload  # 1d ~c128, axis=<given> (default)
def skewtest(
    a: onp.ToArrayStrict1D[op.JustComplex, np.complex128 | np.complex64],
    axis: SupportsIndex = 0,
    alternative: Alternative = "two-sided",
) -> SkewtestResult[np.float64, np.complex128]: ...
@overload  # 2d, axis=<given> (default)
def skewtest(
    a: onp.ToFloatStrict2D, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> SkewtestResult[onp.Array1D[np.float64], onp.MArray1D[np.float64]]: ...
@overload  # 2d ~c128, axis=<given> (default)
def skewtest(
    a: onp.ToArrayStrict2D[op.JustComplex, np.complex128 | np.complex64],
    axis: SupportsIndex = 0,
    alternative: Alternative = "two-sided",
) -> SkewtestResult[onp.Array1D[np.float64], onp.MArray1D[np.complex128]]: ...
@overload  # 3d, axis=<given> (default)
def skewtest(
    a: onp.ToFloatStrict3D, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> SkewtestResult[onp.Array2D[np.float64], onp.MArray2D[np.float64]]: ...
@overload  # 3d ~c128, axis=<given> (default)
def skewtest(
    a: onp.ToArrayStrict3D[op.JustComplex, np.complex128 | np.complex64],
    axis: SupportsIndex = 0,
    alternative: Alternative = "two-sided",
) -> SkewtestResult[onp.Array2D[np.float64], onp.MArray2D[np.complex128]]: ...
@overload  # fallback
def skewtest(
    a: onp.ToComplexND, axis: SupportsIndex | None = 0, alternative: Alternative = "two-sided"
) -> SkewtestResult[onp.ArrayND[np.float64] | Any, onp.MArray[np.float64] | Any]: ...

#
@overload  # ?d, axis=None
def kurtosistest(
    a: onp.ToFloatND, axis: None, alternative: Alternative = "two-sided"
) -> KurtosistestResult[np.float64, np.float64]: ...
@overload  # ?d ~c128, axis=None
def kurtosistest(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.complex64], axis: None, alternative: Alternative = "two-sided"
) -> KurtosistestResult[np.float64, np.complex128]: ...
@overload  # ?d, axis=<given> (default)
def kurtosistest(
    a: _ToFloatStrictND, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> KurtosistestResult[onp.ArrayND[np.float64] | Any, onp.MArray[np.float64] | Any]: ...
@overload  # ?d ~c128, axis=<given> (default)
def kurtosistest(
    a: onp.ArrayND[np.complex128 | np.complex64, _JustAnyShape], axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> KurtosistestResult[onp.ArrayND[np.float64] | Any, onp.MArray[np.complex128] | Any]: ...
@overload  # 1d, axis=<given> (default)
def kurtosistest(
    a: onp.ToFloatStrict1D, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> KurtosistestResult[np.float64, np.float64]: ...
@overload  # 1d ~c128, axis=<given> (default)
def kurtosistest(
    a: onp.ToArrayStrict1D[op.JustComplex, np.complex128 | np.complex64],
    axis: SupportsIndex = 0,
    alternative: Alternative = "two-sided",
) -> KurtosistestResult[np.float64, np.complex128]: ...
@overload  # 2d, axis=<given> (default)
def kurtosistest(
    a: onp.ToFloatStrict2D, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> KurtosistestResult[onp.Array1D[np.float64], onp.MArray1D[np.float64]]: ...
@overload  # 2d ~c128, axis=<given> (default)
def kurtosistest(
    a: onp.ToArrayStrict2D[op.JustComplex, np.complex128 | np.complex64],
    axis: SupportsIndex = 0,
    alternative: Alternative = "two-sided",
) -> KurtosistestResult[onp.Array1D[np.float64], onp.MArray1D[np.complex128]]: ...
@overload  # 3d, axis=<given> (default)
def kurtosistest(
    a: onp.ToFloatStrict3D, axis: SupportsIndex = 0, alternative: Alternative = "two-sided"
) -> KurtosistestResult[onp.Array2D[np.float64], onp.MArray2D[np.float64]]: ...
@overload  # 3d ~c128, axis=<given> (default)
def kurtosistest(
    a: onp.ToArrayStrict3D[op.JustComplex, np.complex128 | np.complex64],
    axis: SupportsIndex = 0,
    alternative: Alternative = "two-sided",
) -> KurtosistestResult[onp.Array2D[np.float64], onp.MArray2D[np.complex128]]: ...
@overload  # fallback
def kurtosistest(
    a: onp.ToComplexND, axis: SupportsIndex | None = 0, alternative: Alternative = "two-sided"
) -> KurtosistestResult[onp.ArrayND[np.float64] | Any, onp.MArray[np.float64] | Any]: ...

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
@overload  # ?d ~f64, axis=None (default)
def mquantiles(
    a: onp.ToFloat64_ND,
    prob: onp.ToFloat | onp.ToFloatND = (0.25, 0.5, 0.75),
    alphap: onp.ToFloat = 0.4,
    betap: onp.ToFloat = 0.4,
    axis: None = None,
    limit: tuple[onp.ToFloat, onp.ToFloat] | tuple[()] = (),
) -> onp.Array1D[np.float64]: ...
@overload  # ?d T@floating80, axis=None (default)
def mquantiles[FloatT: npc.floating80](
    a: onp.ToArrayND[FloatT, FloatT],
    prob: onp.ToFloat | onp.ToFloatND = (0.25, 0.5, 0.75),
    alphap: onp.ToFloat = 0.4,
    betap: onp.ToFloat = 0.4,
    axis: None = None,
    limit: tuple[onp.ToFloat, onp.ToFloat] | tuple[()] = (),
) -> onp.Array1D[FloatT]: ...
@overload  # ?d ~f64, axis=<given>
def mquantiles(
    a: onp.ArrayND[_AsF64 | np.float32 | np.float16, _JustAnyShape],
    prob: onp.ToFloat | onp.ToFloatND = (0.25, 0.5, 0.75),
    alphap: onp.ToFloat = 0.4,
    betap: onp.ToFloat = 0.4,
    *,
    axis: SupportsIndex,
    limit: tuple[onp.ToFloat, onp.ToFloat] | tuple[()] = (),
) -> onp.MArray[np.float64] | Any: ...
@overload  # ?d T@floating80, axis=<given>
def mquantiles[FloatT: npc.floating80](
    a: onp.ArrayND[FloatT, _JustAnyShape],
    prob: onp.ToFloat | onp.ToFloatND = (0.25, 0.5, 0.75),
    alphap: onp.ToFloat = 0.4,
    betap: onp.ToFloat = 0.4,
    *,
    axis: SupportsIndex,
    limit: tuple[onp.ToFloat, onp.ToFloat] | tuple[()] = (),
) -> onp.MArray[FloatT] | Any: ...
@overload  # 1d ~f64, axis=<given>
def mquantiles(
    a: onp.ToFloat64Strict1D,
    prob: onp.ToFloat | onp.ToFloatND = (0.25, 0.5, 0.75),
    alphap: onp.ToFloat = 0.4,
    betap: onp.ToFloat = 0.4,
    *,
    axis: SupportsIndex,
    limit: tuple[onp.ToFloat, onp.ToFloat] | tuple[()] = (),
) -> onp.MArray1D[np.float64]: ...
@overload  # 1d T@floating80, axis=<given>
def mquantiles[FloatT: npc.floating80](
    a: onp.ToArrayStrict1D[FloatT, FloatT],
    prob: onp.ToFloat | onp.ToFloatND = (0.25, 0.5, 0.75),
    alphap: onp.ToFloat = 0.4,
    betap: onp.ToFloat = 0.4,
    *,
    axis: SupportsIndex,
    limit: tuple[onp.ToFloat, onp.ToFloat] | tuple[()] = (),
) -> onp.MArray1D[FloatT]: ...
@overload  # 2d ~f64, axis=<given>
def mquantiles(
    a: onp.ToFloat64Strict2D,
    prob: onp.ToFloat | onp.ToFloatND = (0.25, 0.5, 0.75),
    alphap: onp.ToFloat = 0.4,
    betap: onp.ToFloat = 0.4,
    *,
    axis: SupportsIndex,
    limit: tuple[onp.ToFloat, onp.ToFloat] | tuple[()] = (),
) -> onp.MArray2D[np.float64]: ...
@overload  # 2d T@floating80, axis=<given>
def mquantiles[FloatT: npc.floating80](
    a: onp.ToArrayStrict2D[FloatT, FloatT],
    prob: onp.ToFloat | onp.ToFloatND = (0.25, 0.5, 0.75),
    alphap: onp.ToFloat = 0.4,
    betap: onp.ToFloat = 0.4,
    *,
    axis: SupportsIndex,
    limit: tuple[onp.ToFloat, onp.ToFloat] | tuple[()] = (),
) -> onp.MArray2D[FloatT]: ...

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
def f_oneway(arg0: onp.ToFloat1D, /, *args: onp.ToFloat1D) -> F_onewayResult: ...

#
def friedmanchisquare(
    arg0: onp.ToFloat1D, arg1: onp.ToFloat1D, arg2: onp.ToFloat1D, /, *args: onp.ToFloat1D
) -> FriedmanchisquareResult: ...

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
