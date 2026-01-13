from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.integrate import lebedev_rule

# kinda trivial, but helps against regressions I guess...
assert_type(lebedev_rule(6), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64]])
assert_type(lebedev_rule(n=6), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64]])
