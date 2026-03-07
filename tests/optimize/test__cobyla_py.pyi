from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import fmin_cobyla

###
# fmin_cobyla

def _func(x: onp.Array1D[np.float64]) -> float: ...
def _con(x: onp.Array1D[np.float64]) -> float: ...

assert_type(fmin_cobyla(_func, [0.5, 0.5], [_con]), onp.Array1D[np.float64])
