# ruff: noqa: INP001
from __future__ import annotations

from typing import assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy import stats


def batch_mean(x: npt.NDArray[np.float64], axis: int = -1) -> npt.NDArray[np.float64]:
    """Vectorized statistic for dimensionally-aware bootstrap validation."""
    return np.mean(x, axis=axis)  # type: ignore[return-value]


def strict_rvs(
    size: int | tuple[int, ...] | None = None,
    *,
    random_state: int | None = None
) -> npt.NDArray[np.float64]:
    """Protocol-compliant RVS callable for monte_carlo_test validation."""
    rng = np.random.default_rng(random_state)
    return rng.standard_normal(size)


# Test 1: Vectorized Bootstrap
data_bs: tuple[npt.NDArray[np.float64], ...] = (np.ones((10, 5), dtype=np.float64),)
res_bs = stats.bootstrap(data_bs, batch_mean, vectorized=True)
assert_type(res_bs.confidence_interval.low, onp.ArrayND[np.float64])

# Test 2: Vectorized Permutation Test
x_perm = np.ones((10, 5), dtype=np.float64)
res_perm = stats.permutation_test((x_perm,), batch_mean, vectorized=True)
assert_type(res_perm.statistic, onp.ArrayND[np.float64])

# Test 3: Vectorized Monte Carlo Test
x_mc: npt.NDArray[np.float64] = np.ones((10, 5), dtype=np.float64)
res_mc = stats.monte_carlo_test(x_mc, strict_rvs, batch_mean, vectorized=True)
assert_type(res_mc.pvalue, onp.ArrayND[np.float64])
