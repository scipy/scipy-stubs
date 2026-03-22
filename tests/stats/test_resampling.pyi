# type-tests for `stats/_resampling.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import BootstrapMethod, MonteCarloMethod, PermutationMethod, bootstrap, monte_carlo_test, permutation_test, power
from scipy.stats._resampling import BootstrapResult, MonteCarloTestResult, PermutationTestResult, PowerResult

###

_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_f_3d: list[list[list[float]]]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

def _statistic_1d(x: onp.Array1D[np.float64], y: onp.Array1D[np.float64]) -> np.float64: ...
def _statistic_2d(x: onp.Array2D[np.float64], y: onp.Array2D[np.float64]) -> onp.Array1D[np.float64]: ...
def _statistic_3d(x: onp.Array3D[np.float64], y: onp.Array3D[np.float64]) -> onp.Array2D[np.float64]: ...
def _statistic_nd(x: onp.ArrayND[np.float64], y: onp.ArrayND[np.float64]) -> onp.ArrayND[np.float64]: ...

###

# bootstrap
assert_type(bootstrap(_py_f_1d, np.mean), BootstrapResult[float | np.float64, onp.Array1D[np.float64]])
assert_type(bootstrap(_f64_1d, np.mean), BootstrapResult[float | np.float64, onp.Array1D[np.float64]])
assert_type(bootstrap(_f64_2d, np.mean), BootstrapResult[onp.Array1D[np.float64], onp.Array2D[np.float64]])
assert_type(bootstrap(_f64_3d, np.mean), BootstrapResult[onp.Array2D[np.float64], onp.Array3D[np.float64]])
assert_type(  # pyrefly:ignore[assert-type]
    bootstrap(_f64_nd, np.mean), BootstrapResult[float | np.float64, onp.Array1D[np.float64]]
)

# permutation_test
assert_type(
    permutation_test((_py_f_1d, _py_f_1d), _statistic_1d), PermutationTestResult[onp.Array1D[np.float64], onp.Array2D[np.float64]]
)
assert_type(
    permutation_test((_f64_1d, _f64_1d), _statistic_1d), PermutationTestResult[onp.Array1D[np.float64], onp.Array2D[np.float64]]
)
assert_type(
    permutation_test((_f64_2d, _f64_2d), _statistic_2d), PermutationTestResult[onp.Array2D[np.float64], onp.Array3D[np.float64]]
)
assert_type(permutation_test((_f64_3d, _f64_3d), _statistic_3d), PermutationTestResult[Any, onp.ArrayND[np.float64]])
assert_type(  # pyrefly:ignore[assert-type]
    permutation_test((_f64_nd, _f64_nd), _statistic_nd), PermutationTestResult[onp.Array1D[np.float64], onp.Array2D[np.float64]]
)

# monte_carlo_test
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
assert_type(  # pyrefly:ignore[assert-type]
    monte_carlo_test(_f64_nd, np.random.standard_normal, np.mean),
    MonteCarloTestResult[float | np.float64, onp.Array1D[np.float64]],
)

# BootstrapMethod
assert_type(BootstrapMethod(), BootstrapMethod)
assert_type(BootstrapMethod(n_resamples=999, method="percentile"), BootstrapMethod)

# MonteCarloMethod
assert_type(MonteCarloMethod(), MonteCarloMethod)
assert_type(MonteCarloMethod(n_resamples=999), MonteCarloMethod)

# PermutationMethod
assert_type(PermutationMethod(), PermutationMethod)
assert_type(PermutationMethod(n_resamples=999), PermutationMethod)

# power
assert_type(power(_statistic_1d, np.random.standard_normal, [10, 20]), PowerResult[Any])
