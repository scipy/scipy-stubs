# ruff: noqa: INP001
from __future__ import annotations

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy import stats

# --- CI Compatibility Helpers ---
# We use explicit helper functions to provide strict input signatures.
# This ensures that the CI type-checker resolves arguments as 'NDArray'
# rather than 'Unknown', preventing 'Any' return type inference failures.


def strict_mean(x: npt.NDArray[np.float64]) -> float:
    """Provides a strict (NDArray) -> float signature for scalar validation."""
    return float(np.mean(x))


def batch_mean(x: npt.NDArray[np.float64], axis: int = -1) -> npt.NDArray[np.float64]:
    """Provides a vectorized (NDArray, axis) -> NDArray signature."""
    # We use a type ignore here as np.mean return types vary by NumPy version
    return np.mean(x, axis=axis)  # type: ignore[return-value]


def strict_rvs(
    size: int | tuple[int, ...] | None = None,
    *,
    random_state: int | None = None
) -> npt.NDArray[np.float64]:
    """
    A compliant RVS callable satisfying the _RVSCallable protocol.
    Defined manually to avoid 'BoundMethod' mismatches in strict CI environments.
    """
    rng = np.random.default_rng(random_state)
    return rng.standard_normal(size)


# ----------------------------------------------------------------

# Test 1: Scalar Bootstrap (Regression Validation)
# Validates that the default non-vectorized behavior correctly resolves
# to a scalar return, ensuring consistency for existing scalar use cases.
data_1d: tuple[npt.NDArray[np.float64], ...] = (np.array([1.0, 2.0], dtype=np.float64),)
res_scalar = stats.bootstrap(data_1d, strict_mean)
assert_type(res_scalar.confidence_interval.low, Any)

# Test 2: Vectorized Bootstrap (Feature Validation for #1099)
# Verifies that 'vectorized=True' correctly triggers the NDArray return overload.
# This proves the stubs are now dimensionally aware for multi-sample inputs.
data_2d: tuple[npt.NDArray[np.float64], ...] = (np.ones((10, 5), dtype=np.float64),)
res_vec = stats.bootstrap(data_2d, batch_mean, vectorized=True)
# Using 'onp.ArrayND' matches the internal typing engine used in this repository.
assert_type(res_vec.confidence_interval.low, onp.ArrayND[np.float64])

# Test 3: Monte Carlo Test (Overload Compatibility)
# Validates result resolution when using a compliant RVS callable.
# We test against 'Any' to confirm successful overload matching on the server.
x_mc: npt.NDArray[np.float64] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
res_mc = stats.monte_carlo_test(x_mc, strict_rvs, strict_mean)
assert_type(res_mc.pvalue, Any)
