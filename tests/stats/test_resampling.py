# ruff: noqa: INP001
from __future__ import annotations

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy import stats

# --- CI Compatibility Helpers ---
# Explicit signatures prevent 'Unknown' argument inference in strict CI environments.


def strict_mean(x: npt.NDArray[np.float64]) -> float:
    """Provides a strict (NDArray) -> float signature for scalar validation."""
    return float(np.mean(x))


def batch_mean(x: npt.NDArray[np.float64], axis: int = -1) -> npt.NDArray[np.float64]:
    """Provides a vectorized (NDArray, axis) -> NDArray signature."""
    return np.mean(x, axis=axis)  # type: ignore[return-value]


def strict_rvs(
    size: int | tuple[int, ...] | None = None,
    *,
    random_state: int | None = None
) -> npt.NDArray[np.float64]:
    """Compliant RVS callable satisfying the strict _RVSCallable protocol."""
    rng = np.random.default_rng(random_state)
    return rng.standard_normal(size)


# ----------------------------------------------------------------

# Test 1: Scalar Bootstrap
# Validates standard 1D inputs resolve to a scalar.
data_1d: tuple[npt.NDArray[np.float64], ...] = (np.array([1.0, 2.0], dtype=np.float64),)
res_scalar = stats.bootstrap(data_1d, strict_mean)
assert_type(res_scalar.confidence_interval.low, Any)

# Test 2: Vectorized Bootstrap (Validation for #1099)
# Confirms that vectorized=True correctly triggers the NDArray return overload.
data_2d: tuple[npt.NDArray[np.float64], ...] = (np.ones((10, 5), dtype=np.float64),)
res_vec = stats.bootstrap(data_2d, batch_mean, vectorized=True)
assert_type(res_vec.confidence_interval.low, onp.ArrayND[np.float64])

# Test 3: Monte Carlo Test
# Verifies p-value resolution using a compliant RVS callable.
x_mc: npt.NDArray[np.float64] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
res_mc = stats.monte_carlo_test(x_mc, strict_rvs, strict_mean)
assert_type(res_mc.pvalue, Any)
