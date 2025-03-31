from typing_extensions import assert_type

import numpy as np
import optype.numpy as onp
from scipy.interpolate import BSpline

# based on the example from
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html

k: int | bool
t: list[int | bool]
c: list[int | bool]
spl = BSpline(t, c, k)
assert_type(spl, BSpline[np.float64])
assert_type(spl(2.5), onp.ArrayND[np.float64])

c_complex: list[complex | float | int | bool]
spl_complex = BSpline(t, c_complex, k)
assert_type(spl_complex, BSpline[np.complex128])
assert_type(spl_complex(2.5), onp.ArrayND[np.complex128])
