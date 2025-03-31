from typing import Literal, TypeAlias

from ._common import ConfidenceInterval

_Alternative: TypeAlias = Literal["two-sided", "less", "greater"]

class BinomTestResult:
    k: int | bool
    n: int | bool
    alternative: _Alternative
    statistic: float | int | bool
    pvalue: float | int | bool

    def __init__(
        self,
        /,
        k: int | bool,
        n: int | bool,
        alternative: _Alternative,
        statistic: float | int | bool,
        pvalue: float | int | bool,
    ) -> None: ...
    def proportion_ci(
        self,
        /,
        confidence_level: float | int | bool = 0.95,
        method: Literal["exact", "wilson", "wilsoncc"] = "exact",
    ) -> ConfidenceInterval: ...

def binomtest(
    k: int | bool, n: int | bool, p: float | int | bool = 0.5, alternative: _Alternative = "two-sided"
) -> BinomTestResult: ...
