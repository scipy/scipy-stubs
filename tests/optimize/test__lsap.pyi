from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import linear_sum_assignment

###
# linear_sum_assignment

_cost_matrix: list[list[float]]

assert_type(linear_sum_assignment(_cost_matrix), tuple[onp.Array1D[np.intp], onp.Array1D[np.intp]])
assert_type(linear_sum_assignment(_cost_matrix, maximize=True), tuple[onp.Array1D[np.intp], onp.Array1D[np.intp]])
