from types import ModuleType
from typing import Generic, Literal, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._common import ConfidenceInterval
from ._typing import Alternative

###

_NumT_co = TypeVar("_NumT_co", bound=int | onp.ArrayND[np.float64], default=int, covariant=True)
_StatT_co = TypeVar("_StatT_co", bound=np.float64 | onp.ArrayND[np.float64], default=np.float64, covariant=True)

###

class BinomTestResult(Generic[_NumT_co, _StatT_co]):
    k: _NumT_co
    n: _NumT_co
    alternative: Alternative
    statistic: _StatT_co
    pvalue: _StatT_co
    proportion_estimate: _StatT_co  # alias of `statistic` for backwards compatibility

    def __init__(
        self, /, k: _StatT_co, n: _StatT_co, alternative: Alternative, statistic: _StatT_co, pvalue: _StatT_co, xp: ModuleType
    ) -> None: ...
    def proportion_ci(
        self, /, confidence_level: float = 0.95, method: Literal["exact", "wilson", "wilsoncc"] = "exact"
    ) -> ConfidenceInterval[_StatT_co]: ...

@overload
def binomtest(k: int, n: int, p: float = 0.5, alternative: Alternative = "two-sided") -> BinomTestResult[int, np.float64]: ...
@overload
def binomtest(
    k: onp.ToArrayND[int, npc.integer] | int,
    n: onp.ToArrayND[int, npc.integer],
    p: onp.ToArrayND[float, npc.floating] | float = 0.5,
    alternative: Alternative = "two-sided",
) -> BinomTestResult[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]: ...
@overload
def binomtest(
    k: onp.ToArrayND[int, npc.integer],
    n: onp.ToArrayND[int, npc.integer] | int,
    p: onp.ToArrayND[float, npc.floating] | float = 0.5,
    alternative: Alternative = "two-sided",
) -> BinomTestResult[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]: ...
@overload
def binomtest(
    k: onp.ToArrayND[int, npc.integer] | int,
    n: onp.ToArrayND[int, npc.integer] | int,
    p: onp.ToArrayND[float, npc.floating],
    alternative: Alternative = "two-sided",
) -> BinomTestResult[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]: ...
