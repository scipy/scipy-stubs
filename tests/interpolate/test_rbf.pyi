# type-tests for `interpolate/_rbf.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import Rbf

_f64_1d: onp.Array1D[np.float64]
_rbf: Rbf

assert_type(Rbf(_f64_1d, _f64_1d, _f64_1d), Rbf)
assert_type(_rbf(_f64_1d), onp.ArrayND[np.float64])
assert_type(_rbf.N, int)
assert_type(_rbf.di, onp.Array1D[np.float64])
assert_type(_rbf.xi, onp.Array2D[np.float64])
assert_type(_rbf.nodes, onp.Array1D[np.float64])
assert_type(_rbf.A, onp.Array2D[np.float64])
