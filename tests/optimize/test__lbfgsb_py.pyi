from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import LbfgsInvHessProduct

###

_f64_2d: onp.Array2D[np.float64]

###
# LbfgsInvHessProduct

_prod = LbfgsInvHessProduct(_f64_2d, _f64_2d)
assert_type(_prod.sk, onp.Array2D[np.float64])
assert_type(_prod.yk, onp.Array2D[np.float64])
assert_type(_prod.n_corrs, int)
assert_type(_prod.todense(), onp.Array2D[np.float64])
