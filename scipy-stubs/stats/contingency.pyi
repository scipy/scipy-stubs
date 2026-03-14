from typing import Any, Generic, Literal, Self, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._crosstab import crosstab
from ._odds_ratio import odds_ratio
from ._relative_risk import relative_risk
from ._resampling import ResamplingMethod
from ._typing import BaseBunch, PowerDivergenceStatistic

__all__ = ["association", "chi2_contingency", "crosstab", "expected_freq", "margins", "odds_ratio", "relative_risk"]

###

_NumericScalarT = TypeVar("_NumericScalarT", bound=npc.number | np.timedelta64)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...], default=tuple[Any, ...])

_to_floating: TypeAlias = npc.floating | npc.integer | np.bool_  # noqa: PYI042

###

class Chi2ContingencyResult(BaseBunch[np.float64, np.float64, int, onp.ArrayND[np.float64, _ShapeT]], Generic[_ShapeT]):
    @property
    def statistic(self, /) -> np.float64: ...
    @property
    def pvalue(self, /) -> np.float64: ...
    @property
    def dof(self, /) -> int: ...
    @property
    def expected_freq(self, /) -> onp.ArrayND[np.float64, _ShapeT]: ...

    #
    def __new__(
        _cls, statistic: np.float64, pvalue: np.float64, dof: int, expected_freq: onp.ArrayND[np.float64, _ShapeT]
    ) -> Self: ...
    def __init__(
        self, /, statistic: np.float64, pvalue: np.float64, dof: int, expected_freq: onp.ArrayND[np.float64, _ShapeT]
    ) -> None: ...

#
@overload
def margins(a: onp.ArrayND[_NumericScalarT, _ShapeT]) -> list[onp.ArrayND[_NumericScalarT, _ShapeT]]: ...
@overload
def margins(a: onp.ArrayND[np.bool_]) -> list[onp.ArrayND[np.int_]]: ...

#
@overload
def expected_freq(observed: onp.ArrayND[_to_floating, _ShapeT]) -> onp.ArrayND[np.float64, _ShapeT]: ...
@overload
def expected_freq(observed: onp.ToFloatND) -> onp.ArrayND[np.float64]: ...

#
@overload
def chi2_contingency(  # pyright: ignore[reportOverlappingOverload]
    observed: onp.ArrayND[_to_floating, _ShapeT],
    correction: bool = True,
    lambda_: PowerDivergenceStatistic | float | None = None,
    *,
    method: ResamplingMethod | None = None,
) -> Chi2ContingencyResult[_ShapeT]: ...
@overload
def chi2_contingency(
    observed: onp.ToFloatND,
    correction: bool = True,
    lambda_: PowerDivergenceStatistic | float | None = None,
    *,
    method: ResamplingMethod | None = None,
) -> Chi2ContingencyResult: ...

#
def association(
    observed: onp.ToJustIntND,
    method: Literal["cramer", "tschuprow", "pearson"] = "cramer",
    correction: bool = False,
    lambda_: PowerDivergenceStatistic | float | None = None,
) -> float: ...
