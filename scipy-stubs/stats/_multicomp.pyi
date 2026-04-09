from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._common import ConfidenceInterval
from ._typing import Alternative

__all__ = ["dunnett"]

_StatT_co = TypeVar("_StatT_co", bound=npc.floating, default=np.float64, covariant=True)
_ToFloatNoLong: TypeAlias = float | np.float16 | np.float32 | np.float64 | npc.integer
_ToFloatNoLong1D: TypeAlias = Sequence[_ToFloatNoLong] | onp.CanArrayND[np.float16 | np.float32 | np.float64 | npc.integer]
_ToLongDouble1D: TypeAlias = Sequence[np.longdouble] | onp.CanArrayND[np.longdouble]

@dataclass
class DunnettResult(Generic[_StatT_co]):
    statistic: onp.Array1D[_StatT_co]
    pvalue: onp.Array1D[np.float64]

    _alternative: Alternative
    _rho: onp.Array2D[np.float64]
    _df: int
    _std: np.float64
    _mean_samples: onp.Array1D[_StatT_co]
    _mean_control: _StatT_co  # runtime is a scalar with same dtype as statistic
    _n_samples: onp.Array1D[np.int_]
    _n_control: int
    _rng: np.random.Generator | np.random.RandomState

    _ci: ConfidenceInterval[_StatT_co] | None = None
    _ci_cl: float | npc.floating | None = None

    def confidence_interval(self, /, confidence_level: float | npc.floating = 0.95) -> ConfidenceInterval[_StatT_co]: ...

@overload
def dunnett(  # type: ignore[overload-overlap]
    sample: _ToLongDouble1D,
    *samples: onp.ToFloat1D,
    control: onp.ToFloat1D,
    alternative: Alternative = "two-sided",
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> DunnettResult[np.longdouble]: ...
@overload
def dunnett(  # type: ignore[overload-overlap]
    *samples: onp.ToFloat1D,
    control: _ToLongDouble1D,
    alternative: Alternative = "two-sided",
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> DunnettResult[np.longdouble]: ...
@overload
def dunnett(
    *samples: _ToFloatNoLong1D,
    control: _ToFloatNoLong1D,
    alternative: Alternative = "two-sided",
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> DunnettResult[np.float64]: ...
@overload
def dunnett(
    *samples: onp.ToFloat1D,
    control: onp.ToFloat1D,
    alternative: Alternative = "two-sided",
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> DunnettResult[np.float64 | np.longdouble]: ...
