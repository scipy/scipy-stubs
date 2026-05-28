from types import ModuleType
from typing import Literal

from ._common import ConfidenceInterval
from ._typing import Alternative

###

class BinomTestResult:
    k: int
    n: int
    alternative: Alternative
    statistic: float
    pvalue: float

    def __init__(self, /, k: int, n: int, alternative: Alternative, statistic: float, pvalue: float, xp: ModuleType) -> None: ...
    def proportion_ci(
        self, /, confidence_level: float = 0.95, method: Literal["exact", "wilson", "wilsoncc"] = "exact"
    ) -> ConfidenceInterval: ...

def binomtest(k: int, n: int, p: float = 0.5, alternative: Alternative = "two-sided") -> BinomTestResult: ...
