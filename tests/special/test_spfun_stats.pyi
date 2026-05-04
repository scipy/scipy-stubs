from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.special import multigammaln

f_arr: onp.ArrayND[np.float64]

# multigammaln
assert_type(multigammaln(1.0, 2), np.float64)
assert_type(multigammaln(1.0, np.uint8(2)), np.float64)
assert_type(multigammaln(f_arr, 2), onp.ArrayND[np.float64])
assert_type(multigammaln(f_arr, np.uint8(2)), onp.ArrayND[np.float64])
