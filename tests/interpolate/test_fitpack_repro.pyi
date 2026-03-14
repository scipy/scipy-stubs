from collections.abc import Generator
from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.interpolate import BSpline, generate_knots, make_splprep, make_splrep

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]

###
# generate_knots

assert_type(generate_knots(_f64_1d, _f64_1d), Generator[onp.Array1D[np.float64]])

###
# make_splrep

assert_type(make_splrep(_f64_1d, _f64_1d), BSpline[np.float64])

###
# make_splprep

assert_type(make_splprep(_f64_2d), tuple[BSpline[np.float64], onp.Array1D[npc.floating]])
