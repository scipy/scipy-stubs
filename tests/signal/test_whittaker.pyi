# type-tests for `signal/_whittaker.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import whittaker_henderson
from scipy.signal._whittaker import _WhittakerHendersonResult

###

_py_f_1d: list[float]
_f64_1d: onp.Array1D[np.float64]

###

assert_type(whittaker_henderson(_py_f_1d), _WhittakerHendersonResult)
assert_type(whittaker_henderson(_f64_1d, lamb=1.0, order=3, weights=_f64_1d), _WhittakerHendersonResult)

result = whittaker_henderson(_f64_1d)
assert_type(result.x, onp.Array1D[np.float64])
assert_type(result.lamb, float)
