from typing import Generic, Literal, NamedTuple, TypeAlias
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy._typing import Alternative
from ._resampling import PermutationMethod

_FloatOrArray: TypeAlias = float | int | bool | np.float64 | onp.ArrayND[np.float64]
_FloatOrArrayT = TypeVar("_FloatOrArrayT", bound=_FloatOrArray, default=_FloatOrArray)

class MannwhitneyuResult(NamedTuple, Generic[_FloatOrArrayT]):
    statistic: _FloatOrArrayT
    pvalue: _FloatOrArrayT

def mannwhitneyu(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int | bool = 0,
    method: Literal["auto", "asymptotic", "exact"] | PermutationMethod = "auto",
    *,
    nan_policy: Literal["propagate", "raise", "coerce", "omit"] = "propagate",
    keepdims: bool = False,
) -> MannwhitneyuResult: ...
