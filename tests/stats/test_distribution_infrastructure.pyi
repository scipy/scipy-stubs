# type-tests for "new" distribution infrastructure functions from `stats/_distribution_infrastructure.pyi`

from typing import assert_type, type_check_only

import numpy as np
import optype.numpy as onp

from scipy.stats import Uniform, distributions, make_distribution
from scipy.stats._distribution_infrastructure import ContinuousDistribution, TruncatedDistribution, truncate

###

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]

_uniform_0d_f64: Uniform[tuple[()], np.float64]
_uniform_1d_f64: Uniform[tuple[int], np.float64]
_uniform_2d_f64: Uniform[tuple[int, int], np.float64]

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

# truncate
assert_type(truncate(_uniform_0d_f64), TruncatedDistribution[Uniform[tuple[()], np.float64], tuple[()]])
assert_type(truncate(_uniform_0d_f64, 0, 1), TruncatedDistribution[Uniform[tuple[()], np.float64], tuple[()]])
assert_type(truncate(_uniform_1d_f64), TruncatedDistribution[Uniform[tuple[int], np.float64], tuple[int]])
assert_type(truncate(_uniform_1d_f64, 0, 1), TruncatedDistribution[Uniform[tuple[int], np.float64], tuple[int]])
assert_type(truncate(_uniform_1d_f64, _f64_1d, _f64_1d), TruncatedDistribution[Uniform[tuple[int], np.float64], tuple[int]])
assert_type(truncate(_uniform_2d_f64), TruncatedDistribution[Uniform[tuple[int, int], np.float64], tuple[int, int]])
assert_type(truncate(_uniform_2d_f64, 0, 1), TruncatedDistribution[Uniform[tuple[int, int], np.float64], tuple[int, int]])
assert_type(
    truncate(_uniform_2d_f64, _f64_1d, _f64_1d), TruncatedDistribution[Uniform[tuple[int, int], np.float64], tuple[int, int]]
)
assert_type(
    truncate(_uniform_2d_f64, _f64_2d, _f64_2d), TruncatedDistribution[Uniform[tuple[int, int], np.float64], tuple[int, int]]
)
