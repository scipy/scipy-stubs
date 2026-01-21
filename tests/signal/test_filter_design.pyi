# type-tests for `signal/_max_len_seq.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import butter

###

# butter

assert_type(butter(8, 0.1), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(butter(8, 0.1, output="ba"), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(butter(8, 0.1, output="zpk"), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128], float])
assert_type(butter(8, 0.1, output="sos"), onp.Array2D[np.float64])
