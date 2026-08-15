from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, Self, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._censored_data import CensoredData
from ._common import ConfidenceInterval
from ._typing import Alternative

__all__ = ["ecdf", "logrank"]

###

type _EDFKind = Literal["cdf", "sf"]
type _CIMethod = Literal["linear", "log-log"]

type _Int1D = onp.Array1D[np.int_]
type _Float1D = onp.Array1D[np.float64]

_QuantileT_co = TypeVar("_QuantileT_co", bound=np.float64 | npc.floating80, default=np.float64, covariant=True)
_KwargsT_contra = TypeVar("_KwargsT_contra", contravariant=True)
_LineT = TypeVar("_LineT")

type _SampleData = onp.ToFloatND | CensoredData[np.float64]

@type_check_only
class _CanStep(Protocol[_KwargsT_contra, _LineT]):
    def step(self, x: _Float1D, y: _Float1D, /, **kwargs: _KwargsT_contra) -> list[_LineT]: ...

###

@dataclass
class EmpiricalDistributionFunction(Generic[_QuantileT_co]):
    # NOTE: the order of attributes matters
    quantiles: onp.Array1D[_QuantileT_co]
    probabilities: _Float1D
    _n: _Int1D
    _d: _Int1D
    _sf: _Float1D
    _kind: _EDFKind

    def __init__(self, /, q: onp.Array1D[_QuantileT_co], p: _Float1D, n: _Int1D, d: _Int1D, kind: _EDFKind) -> None: ...
    def evaluate(self, /, x: onp.ToFloatND) -> onp.ArrayND[np.float64]: ...
    @overload
    def plot(self, /, ax: None = None, **kwds: object) -> list[Any]: ...
    @overload
    def plot[KwargsT, LineT](self, /, ax: _CanStep[KwargsT, LineT], **kwds: KwargsT) -> list[LineT]: ...
    def confidence_interval(
        self, /, confidence_level: onp.ToFloat = 0.95, *, method: _CIMethod = "linear"
    ) -> ConfidenceInterval[Self]: ...

@dataclass
class ECDFResult(Generic[_QuantileT_co]):
    cdf: EmpiricalDistributionFunction[_QuantileT_co]
    sf: EmpiricalDistributionFunction[_QuantileT_co]

    def __init__(self, /, q: onp.Array1D[_QuantileT_co], cdf: _Float1D, sf: _Float1D, n: _Int1D, d: _Int1D) -> None: ...

@dataclass
class LogRankResult:
    statistic: np.float64
    pvalue: np.float64

@overload
def ecdf(sample: onp.ToFloat64_ND | CensoredData[np.float64]) -> ECDFResult[np.float64]: ...
@overload
def ecdf(sample: onp.ToJustLongDoubleND) -> ECDFResult[np.longdouble]: ...

#
def logrank(x: _SampleData, y: _SampleData, alternative: Alternative = "two-sided") -> LogRankResult: ...
