from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import rv_continuous

###

_f64_nd: onp.ArrayND[np.float64]

###

mydist = rv_continuous(name="mydist")

###
# rvs

# scalar size (default)
assert_type(mydist.rvs(), np.float64)
assert_type(mydist.rvs(0.5), np.float64)
assert_type(mydist.rvs(0.5, 0.1), np.float64)
assert_type(mydist.rvs(0.5, 0.1, 1.2), np.float64)
assert_type(mydist.rvs(loc=0), np.float64)
assert_type(mydist.rvs(scale=1), np.float64)
assert_type(mydist.rvs(s=0.5), np.float64)

# custom size
assert_type(mydist.rvs(size=()), np.float64)
assert_type(mydist.rvs(size=4), onp.Array1D[np.float64])
assert_type(mydist.rvs(size=(4,)), onp.Array1D[np.float64])
assert_type(mydist.rvs(size=(4, 2)), onp.Array2D[np.float64])

# batching
assert_type(mydist.rvs(_f64_nd), onp.ArrayND[np.float64])
assert_type(mydist.rvs(0.5, _f64_nd), onp.ArrayND[np.float64])
assert_type(mydist.rvs(0.5, 0.1, _f64_nd), onp.ArrayND[np.float64])
assert_type(mydist.rvs(loc=_f64_nd), onp.ArrayND[np.float64])
assert_type(mydist.rvs(scale=_f64_nd), onp.ArrayND[np.float64])
assert_type(mydist.rvs(s=_f64_nd), onp.ArrayND[np.float64])
