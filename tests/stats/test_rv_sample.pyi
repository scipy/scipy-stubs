from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import rv_discrete
from scipy.stats._distn_infrastructure import rv_sample

###

_i64_1d: onp.Array1D[np.int64]
_f64_1d: onp.Array1D[np.float64]
_py_f_1d: tuple[float, ...]
_n: int

# mypy fails because it (still) doesn't support `__new__` returning something that isn't `Self` (if there's an `__init__`)
assert_type(rv_discrete(values=(_i64_1d, _py_f_1d)), rv_sample[np.int64])  # type: ignore[assert-type]
assert_type(rv_discrete(values=(_f64_1d, _py_f_1d)), rv_sample[np.float64])  # type: ignore[assert-type]

###

_sample_i64: rv_sample[np.int64]
_sample_f64: rv_sample[np.float64]

assert_type(_sample_i64.rvs(), np.int64)
assert_type(_sample_i64.rvs(size=5), onp.Array1D[np.int64])
assert_type(_sample_i64.rvs(size=(_n,)), onp.Array1D[np.int64])
assert_type(_sample_i64.rvs(size=(_n, _n)), onp.Array2D[np.int64])
assert_type(_sample_i64.rvs(size=(_n, _n, _n)), onp.Array3D[np.int64])
assert_type(_sample_f64.rvs(), np.float64)
assert_type(_sample_f64.rvs(size=5), onp.Array1D[np.float64])
assert_type(_sample_f64.rvs(size=(_n, _n)), onp.Array2D[np.float64])
