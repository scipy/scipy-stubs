from typing import Any
from typing_extensions import assert_type

import numpy as np
import optype.numpy as onp
from scipy.interpolate import BSpline

# based on the example from
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html

k: int
t: list[int]
c: list[int]
spl = BSpline(t, c, k)
assert_type(spl(2.5), onp.ArrayND[np.floating[Any]])
