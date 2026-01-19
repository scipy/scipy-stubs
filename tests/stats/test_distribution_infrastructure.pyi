# type-tests for "new" distribution infrastructure functions from `stats/_distribution_infrastructure.pyi`

from typing import type_check_only

import numpy as np

from scipy.stats import distributions, make_distribution
from scipy.stats._distribution_infrastructure import ContinuousDistribution

###

@type_check_only
class _DuckRV:
    @property
    def __make_distribution_version__(self) -> str: ...
    @property
    def parameters(self) -> dict[str, tuple[float, float]]: ...
    @property
    def support(self) -> tuple[float, float]: ...
    def pdf(self, x: float, /, *, quack: float) -> float: ...

@type_check_only
class _MultiDuckRV:
    @property
    def __make_distribution_version__(self) -> str: ...
    @property
    def parameters(self) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]: ...
    def process_parameters(self, quack: float | None = None, swim: float | None = None) -> dict[str, float]: ...
    @property
    def support(self) -> dict[str, tuple[float, float]]: ...
    def pdf(self, x: float, /, *, quack: float, swim: float) -> np.float64: ...

def _assert_continuous_distribution_type(dist: type[ContinuousDistribution], /) -> None: ...

###

# make_distribution
_assert_continuous_distribution_type(make_distribution(distributions.loguniform))
_assert_continuous_distribution_type(make_distribution(_DuckRV))
_assert_continuous_distribution_type(make_distribution(_MultiDuckRV))
