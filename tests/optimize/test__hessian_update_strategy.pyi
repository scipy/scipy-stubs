from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import BFGS, SR1, HessianUpdateStrategy

_arr_1d: onp.Array1D[np.float64]

###
# HessianUpdateStrategy

_hus: HessianUpdateStrategy
assert_type(_hus.get_matrix(), onp.Array2D[np.float64])
assert_type(_hus.dot([1.0, 2.0]), onp.Array1D[np.float64])
assert_type(_hus @ [1.0, 2.0], onp.Array1D[np.float64])

###
# BFGS

_bfgs: BFGS
assert_type(BFGS(), BFGS)
assert_type(BFGS(exception_strategy="skip_update"), BFGS)
assert_type(BFGS(exception_strategy="damp_update", min_curvature=1e-4), BFGS)
assert_type(_bfgs.get_matrix(), onp.Array2D[np.float64])
assert_type(_bfgs.dot(_arr_1d), onp.Array1D[np.float64])

###
# SR1

_sr1: SR1
assert_type(SR1(), SR1)
assert_type(SR1(min_denominator=1e-10), SR1)
assert_type(_sr1.get_matrix(), onp.Array2D[np.float64])
assert_type(_sr1.dot(_arr_1d), onp.Array1D[np.float64])
