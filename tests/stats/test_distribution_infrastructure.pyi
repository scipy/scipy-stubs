# type-tests for "new" distribution infrastructure functions from `stats/_distribution_infrastructure.pyi`

from typing import TypeAlias, assert_type, type_check_only

import numpy as np
import optype.numpy as onp

from scipy.stats import Uniform, distributions, make_distribution, order_statistic
from scipy.stats._distribution_infrastructure import (
    ContinuousDistribution,
    OrderStatisticDistribution,
    TruncatedDistribution,
    truncate,
)

###

_0d: TypeAlias = tuple[()]  # noqa: PYI042
_1d: TypeAlias = tuple[int]  # noqa: PYI042
_2d: TypeAlias = tuple[int, int]  # noqa: PYI042

###

_i64_1d: onp.Array1D[np.int64]
_i64_2d: onp.Array2D[np.int64]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]

_uniform_0d_f64: Uniform[_0d, np.float64]
_uniform_1d_f64: Uniform[_1d, np.float64]
_uniform_2d_f64: Uniform[_2d, np.float64]

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
assert_type(truncate(_uniform_0d_f64), TruncatedDistribution[Uniform[_0d, np.float64], _0d])
assert_type(truncate(_uniform_0d_f64, 0, 1), TruncatedDistribution[Uniform[_0d, np.float64], _0d])
assert_type(truncate(_uniform_1d_f64), TruncatedDistribution[Uniform[_1d, np.float64], _1d])
assert_type(truncate(_uniform_1d_f64, 0, 1), TruncatedDistribution[Uniform[_1d, np.float64], _1d])
assert_type(truncate(_uniform_1d_f64, _f64_1d, _f64_1d), TruncatedDistribution[Uniform[_1d, np.float64], _1d])
assert_type(truncate(_uniform_2d_f64), TruncatedDistribution[Uniform[_2d, np.float64], _2d])
assert_type(truncate(_uniform_2d_f64, 0, 1), TruncatedDistribution[Uniform[_2d, np.float64], _2d])
assert_type(truncate(_uniform_2d_f64, _f64_1d, _f64_1d), TruncatedDistribution[Uniform[_2d, np.float64], _2d])
assert_type(truncate(_uniform_2d_f64, _f64_2d, _f64_2d), TruncatedDistribution[Uniform[_2d, np.float64], _2d])

# order_statistic
assert_type(order_statistic(_uniform_0d_f64, r=0, n=1), OrderStatisticDistribution[Uniform[_0d, np.float64], _0d])
assert_type(order_statistic(_uniform_1d_f64, r=0, n=1), OrderStatisticDistribution[Uniform[_1d, np.float64], _1d])
assert_type(order_statistic(_uniform_1d_f64, r=_i64_1d, n=_i64_1d), OrderStatisticDistribution[Uniform[_1d, np.float64], _1d])
assert_type(order_statistic(_uniform_2d_f64, r=0, n=1), OrderStatisticDistribution[Uniform[_2d, np.float64], _2d])
assert_type(order_statistic(_uniform_2d_f64, r=_i64_1d, n=_i64_1d), OrderStatisticDistribution[Uniform[_2d, np.float64], _2d])
assert_type(order_statistic(_uniform_2d_f64, r=_i64_2d, n=_i64_2d), OrderStatisticDistribution[Uniform[_2d, np.float64], _2d])
