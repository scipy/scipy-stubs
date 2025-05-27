from dataclasses import dataclass
from collections.abc import Callable
from typing import Concatenate, Final, Generic, Literal, NamedTuple, TypeAlias
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import Alternative, NanPolicy
from ._common import ConfidenceInterval
from ._stats_py import SignificanceResult

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

_Float: TypeAlias = float | np.float64
_Float2D: TypeAlias = onp.Array2D[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float64]
_FloatOrND: TypeAlias = _Float | _FloatND
_FloatOrNDT = TypeVar("_FloatOrNDT", bound=_FloatOrND, default=_FloatOrND)

###

class Epps_Singleton_2sampResult(NamedTuple, Generic[_FloatOrNDT]):
    statistic: _FloatOrNDT  # readonly
    pvalue: _FloatOrNDT  # readonly

def epps_singleton_2samp(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    t: onp.ToFloat1D = (0.4, 0.8),
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> Epps_Singleton_2sampResult: ...

class CramerVonMisesResult(Generic[_FloatOrNDT]):
    statistic: _FloatOrNDT  # readonly
    pvalue: _FloatOrNDT  # readonly
    def __init__(self, /, statistic: _FloatOrNDT, pvalue: _FloatOrNDT) -> None: ...

def cramervonmises(
    rvs: onp.ToFloat1D,
    cdf: str | Callable[Concatenate[float, ...], _Float | np.float32],
    args: tuple[onp.ToFloat, ...] = (),
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> CramerVonMisesResult: ...
def cramervonmises_2samp(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    method: Literal["auto", "asymptotic", "exact"] = "auto",
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> CramerVonMisesResult: ...

#
def poisson_means_test(
    k1: int, n1: _Float, k2: int, n2: _Float, *, diff: _Float = 0, alternative: Alternative = "two-sided"
) -> SignificanceResult[np.float64]: ...

#
@dataclass
class SomersDResult:
    statistic: Final[_Float]
    pvalue: Final[_Float]
    table: Final[onp.Array2D[np.float64]]

def somersd(
    x: onp.ToFloat1D | onp.ToFloat2D, y: onp.ToFloat1D | None = None, alternative: Alternative = "two-sided"
) -> SomersDResult: ...

#
@dataclass
class BarnardExactResult:
    statistic: Final[_Float]
    pvalue: Final[_Float]

def barnard_exact(
    table: onp.ToInt2D, alternative: Alternative = "two-sided", pooled: bool = True, n: int = 32
) -> BarnardExactResult: ...

#
@dataclass
class BoschlooExactResult:
    statistic: Final[_Float]
    pvalue: Final[_Float]

def boschloo_exact(table: onp.ToInt2D, alternative: Alternative = "two-sided", n: int = 32) -> BoschlooExactResult: ...

#
class TukeyHSDResult:
    statistic: Final[_Float2D]
    pvalue: Final[_Float2D]
    _ntreatments: Final[int]
    _df: Final[int]
    _stand_err: Final[float]
    def __init__(self, /, statistic: _Float2D, pvalue: _Float2D, _ntreatments: int, _df: int, _stand_err: float) -> None: ...

    #
    _ci: ConfidenceInterval | None
    _ci_cl: float | None
    def confidence_interval(self, /, confidence_level: op.JustFloat = 0.95) -> ConfidenceInterval: ...

def tukey_hsd(arg0: onp.ToFloatND, arg1: onp.ToFloatND, /, *args: onp.ToFloatND, equal_var: bool = True) -> TukeyHSDResult: ...
