# type-tests for `Covariance` from `stats/_covariance.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import Covariance
from scipy.stats._covariance import CovViaCholesky, CovViaDiagonal, CovViaEigendecomposition, CovViaPrecision

###

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]

###

assert_type(Covariance.from_diagonal(_f64_1d), CovViaDiagonal[np.float64])
assert_type(Covariance.from_precision(_f64_2d), CovViaPrecision)
assert_type(Covariance.from_cholesky(_f64_2d), CovViaCholesky)
assert_type(Covariance.from_eigendecomposition((_f64_1d, _f64_2d)), CovViaEigendecomposition)

_cov: Covariance[np.float64]
assert_type(_cov.log_pdet, np.float64)
assert_type(_cov.rank, np.int_)
assert_type(_cov.covariance, onp.Array2D[np.float64])
assert_type(_cov.shape, tuple[int, int])
