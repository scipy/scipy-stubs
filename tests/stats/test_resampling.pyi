# type-tests for `bootstrap`, `permutation_test`, `monte_carlo_test` from `stats/_resampling.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from scipy.stats._resampling import BootstrapResult, MonteCarloTestResult, PermutationTestResult

###

# plain Python lists - the most common input people will use
_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_f_3d: list[list[list[float]]]

# numpy arrays of increasing dimensionality
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]  # unknown dimensionality at type-check time

# typed statistic helpers for permutation_test (lambdas can't be annotated in .pyi files)
# the statistic reduces dimensionality by 1 (e.g. 2D input -> 1D output)
def _statistic_1d(x: onp.Array1D[np.float64], y: onp.Array1D[np.float64]) -> np.float64: ...
def _statistic_2d(x: onp.Array2D[np.float64], y: onp.Array2D[np.float64]) -> onp.Array1D[np.float64]: ...
def _statistic_3d(x: onp.Array3D[np.float64], y: onp.Array3D[np.float64]) -> onp.Array2D[np.float64]: ...
def _statistic_nd(x: onp.ArrayND[np.float64], y: onp.ArrayND[np.float64]) -> onp.ArrayND[np.float64]: ...

###

# bootstrap
# the distribution array is always 1 dimension higher than the input data
# e.g. 1D input -> scalar statistic + 1D bootstrap distribution
#      2D input -> 1D statistic + 2D bootstrap distribution

assert_type(bootstrap(_py_f_1d, np.mean), BootstrapResult[float | np.float64, onp.Array1D[np.float64]])
assert_type(bootstrap(_f64_1d, np.mean), BootstrapResult[float | np.float64, onp.Array1D[np.float64]])
assert_type(bootstrap(_f64_2d, np.mean), BootstrapResult[onp.Array1D[np.float64], onp.Array2D[np.float64]])
assert_type(bootstrap(_f64_3d, np.mean), BootstrapResult[onp.Array2D[np.float64], onp.Array3D[np.float64]])

# when dimensionality is unknown at type-check time, pyright resolves to the first
# matching overload (1D) rather than the generic ND fallback
assert_type(  # pyrefly:ignore[assert-type]
    bootstrap(_f64_nd, np.mean), BootstrapResult[float | np.float64, onp.Array1D[np.float64]]
)

###

# permutation_test
# unlike bootstrap, pyright infers _FloatNDT from the statistic's return type, not
# from the input data shape -- so the result type is driven by what the statistic returns

assert_type(
    permutation_test((_py_f_1d, _py_f_1d), _statistic_1d), PermutationTestResult[onp.Array1D[np.float64], onp.Array2D[np.float64]]
)
assert_type(
    permutation_test((_f64_1d, _f64_1d), _statistic_1d), PermutationTestResult[onp.Array1D[np.float64], onp.Array2D[np.float64]]
)
assert_type(
    permutation_test((_f64_2d, _f64_2d), _statistic_2d), PermutationTestResult[onp.Array2D[np.float64], onp.Array3D[np.float64]]
)

# pyright loses track of the exact shape here and falls back to ArrayND
assert_type(permutation_test((_f64_3d, _f64_3d), _statistic_3d), PermutationTestResult[Any, onp.ArrayND[np.float64]])

# same overload resolution behavior as bootstrap -- resolves to first matching overload
assert_type(  # pyrefly:ignore[assert-type]
    permutation_test((_f64_nd, _f64_nd), _statistic_nd), PermutationTestResult[onp.Array1D[np.float64], onp.Array2D[np.float64]]
)

###

# monte_carlo_test
# same dimension-shifting pattern as bootstrap:
# 1D input -> scalar statistic + 1D null distribution, and so on

assert_type(
    monte_carlo_test(_py_f_1d, np.random.standard_normal, np.mean),
    MonteCarloTestResult[float | np.float64, onp.Array1D[np.float64]],
)
assert_type(
    monte_carlo_test(_f64_1d, np.random.standard_normal, np.mean),
    MonteCarloTestResult[float | np.float64, onp.Array1D[np.float64]],
)
assert_type(
    monte_carlo_test(_f64_2d, np.random.standard_normal, np.mean),
    MonteCarloTestResult[onp.Array1D[np.float64], onp.Array2D[np.float64]],
)
assert_type(
    monte_carlo_test(_f64_3d, np.random.standard_normal, np.mean),
    MonteCarloTestResult[onp.Array2D[np.float64], onp.Array3D[np.float64]],
)

# unknown dimensionality resolves to the first overload, same as bootstrap
assert_type(  # pyrefly:ignore[assert-type]
    monte_carlo_test(_f64_nd, np.random.standard_normal, np.mean),
    MonteCarloTestResult[float | np.float64, onp.Array1D[np.float64]],
)
