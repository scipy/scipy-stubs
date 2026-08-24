from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Concatenate, Final, Generic, Literal, NamedTuple, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from ._common import ConfidenceInterval
from ._stats_py import SignificanceResult
from ._typing import Alternative, NanPolicy

__all__ = [
    "barnard_exact",
    "boschloo_exact",
    "cramervonmises",
    "cramervonmises_2samp",
    "epps_singleton_2samp",
    "poisson_means_test",
    "somersd",
    "tukey_hsd",
]

###

type _AsF64ND = onp.ToArrayND[float, npc.floating64 | npc.integer | np.bool]
type _AsF64Strict1D = onp.ToArrayStrict1D[float, npc.floating64 | npc.integer | np.bool]

type _ToCDF = str | Callable[Concatenate[float, ...], float | np.float32]
type _ToCDFArgs = tuple[onp.ToFloat, ...]

type _CV2Method = Literal["auto", "asymptotic", "exact"]

_FloatOrNDT = TypeVar("_FloatOrNDT", bound=np.float32 | np.float64 | onp.ArrayND[np.float64], default=Any)

###

class Epps_Singleton_2sampResult(NamedTuple, Generic[_FloatOrNDT]):
    statistic: _FloatOrNDT  # readonly
    pvalue: _FloatOrNDT  # readonly

class CramerVonMisesResult(Generic[_FloatOrNDT]):
    statistic: _FloatOrNDT  # readonly
    pvalue: _FloatOrNDT  # readonly
    def __init__(self, /, statistic: _FloatOrNDT, pvalue: _FloatOrNDT) -> None: ...

class TukeyHSDResult:
    statistic: Final[onp.Array2D[np.float64 | Any]]
    pvalue: Final[onp.Array2D[np.float64]]
    _ntreatments: Final[int]
    _df: Final[int]
    _stand_err: Final[float]

    def __init__(
        self,
        /,
        statistic: onp.Array2D[np.float64 | Any],
        pvalue: onp.Array2D[np.float64],
        _ntreatments: int,
        _df: int,
        _stand_err: float,
    ) -> None: ...

    _ci: ConfidenceInterval | None
    _ci_cl: float | None

    def confidence_interval(
        self, /, confidence_level: float | np.float64 = 0.95
    ) -> ConfidenceInterval[onp.Array2D[np.float64]]: ...

@dataclass
class BoschlooExactResult:
    statistic: Final[float]
    pvalue: Final[float]

@dataclass
class SomersDResult:
    statistic: Final[float]
    pvalue: Final[float]
    table: Final[onp.Array2D[np.int_]]

@dataclass
class BarnardExactResult:
    statistic: Final[float]
    pvalue: Final[float]

@overload  # +f64, 1d
def epps_singleton_2samp(
    x: _AsF64Strict1D,
    y: _AsF64Strict1D,
    t: onp.ToFloatStrict1D = (0.4, 0.8),
    *,
    axis: Literal[0, -1] | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: onp.ToFalse = False,
) -> Epps_Singleton_2sampResult[np.float64]: ...
@overload  # +f64, axis: None
def epps_singleton_2samp(
    x: _AsF64ND,
    y: _AsF64ND,
    t: onp.ToFloatND = (0.4, 0.8),
    *,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: onp.ToFalse = False,
) -> Epps_Singleton_2sampResult[np.float64]: ...
@overload  # ~f32, 1d
def epps_singleton_2samp(
    x: onp.ToJustFloat32Strict1D,
    y: onp.ToJustFloat32Strict1D,
    t: onp.ToFloatStrict1D = (0.4, 0.8),
    *,
    axis: Literal[0, -1] | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: onp.ToFalse = False,
) -> Epps_Singleton_2sampResult[np.float32]: ...
@overload  # ~f32, axis: None
def epps_singleton_2samp(
    x: onp.ToJustFloat32_ND,
    y: onp.ToJustFloat32_ND,
    t: onp.ToFloatND = (0.4, 0.8),
    *,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: onp.ToFalse = False,
) -> Epps_Singleton_2sampResult[np.float32]: ...
@overload  # keepdims: True
def epps_singleton_2samp(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    t: onp.ToFloatND = (0.4, 0.8),
    *,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: onp.ToTrue,
) -> Epps_Singleton_2sampResult[onp.ArrayND[np.float64]]: ...
@overload  # fallback
def epps_singleton_2samp(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    t: onp.ToFloatND = (0.4, 0.8),
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> Epps_Singleton_2sampResult: ...
@overload
def cramervonmises(
    rvs: onp.ToFloatStrict1D,
    cdf: _ToCDF,
    args: _ToCDFArgs = (),
    *,
    axis: Literal[0, -1] | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: onp.ToFalse = False,
) -> CramerVonMisesResult[np.float64]: ...
@overload
def cramervonmises(
    rvs: onp.ToFloatND,
    cdf: _ToCDF,
    args: _ToCDFArgs = (),
    *,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: onp.ToFalse = False,
) -> CramerVonMisesResult[np.float64]: ...
@overload
def cramervonmises(
    rvs: onp.ToFloatND,
    cdf: _ToCDF,
    args: _ToCDFArgs = (),
    *,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: onp.ToTrue,
) -> CramerVonMisesResult[onp.ArrayND[np.float64]]: ...
@overload
def cramervonmises(
    rvs: onp.ToFloatND,
    cdf: _ToCDF,
    args: _ToCDFArgs = (),
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> CramerVonMisesResult: ...

#
@overload  # 1d, +f64
def cramervonmises_2samp(
    x: _AsF64Strict1D,
    y: _AsF64Strict1D,
    method: _CV2Method = "auto",
    *,
    axis: Literal[0, -1] | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: onp.ToFalse = False,
) -> CramerVonMisesResult[np.float64]: ...
@overload  # +f64, axis: None
def cramervonmises_2samp(
    x: _AsF64ND,
    y: _AsF64ND,
    method: _CV2Method = "auto",
    *,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: onp.ToFalse = False,
) -> CramerVonMisesResult[np.float64]: ...
@overload  # 1d, ~f32
def cramervonmises_2samp(
    x: onp.ToJustFloat32Strict1D,
    y: onp.ToJustFloat32Strict1D,
    method: _CV2Method = "auto",
    *,
    axis: Literal[0, -1] | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: onp.ToFalse = False,
) -> CramerVonMisesResult[np.float32]: ...
@overload  # x~f32, axis: None
def cramervonmises_2samp(
    x: onp.ToJustFloat32_ND,
    y: onp.ToJustFloat32_ND,
    method: _CV2Method = "auto",
    *,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: onp.ToFalse = False,
) -> CramerVonMisesResult[np.float32]: ...
@overload  # keepdims: True
def cramervonmises_2samp(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    method: _CV2Method = "auto",
    *,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: onp.ToTrue,
) -> CramerVonMisesResult[onp.ArrayND[np.float64]]: ...
@overload  # fallback
def cramervonmises_2samp(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    method: _CV2Method = "auto",
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> CramerVonMisesResult: ...

#
def poisson_means_test(
    k1: int, n1: float, k2: int, n2: float, *, diff: float = 0, alternative: Alternative = "two-sided"
) -> SignificanceResult[np.float64]: ...

#
def somersd(
    x: onp.ToFloat1D | onp.ToFloat2D, y: onp.ToFloat1D | None = None, alternative: Alternative = "two-sided"
) -> SomersDResult: ...

#
def barnard_exact(
    table: onp.ToInt2D, alternative: Alternative = "two-sided", pooled: bool = True, n: op.JustInt = 32
) -> BarnardExactResult: ...

#
def boschloo_exact(table: onp.ToInt2D, alternative: Alternative = "two-sided", n: op.JustInt = 32) -> BoschlooExactResult: ...

#
def tukey_hsd(arg0: onp.ToFloatND, arg1: onp.ToFloatND, /, *args: onp.ToFloatND, equal_var: bool = True) -> TukeyHSDResult: ...
