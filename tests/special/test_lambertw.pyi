from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.special import lambertw

k_arr: onp.ArrayND[np.intp]

assert_type(lambertw(1.0), np.complex128)
assert_type(lambertw(1j), np.complex128)
assert_type(lambertw(1.0, k=0), np.complex128)
assert_type(lambertw(1.0, k_arr), onp.ArrayND[np.complex128])
