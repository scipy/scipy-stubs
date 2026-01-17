from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.io import FortranFile

_i32_1d: onp.Array1D[np.int32]

###

f = FortranFile("test.unf", "w")
assert_type(f, FortranFile)
assert_type(f.write_record(_i32_1d), None)
assert_type(f.read_record(), onp.Array1D[np.void])
assert_type(f.read_ints(), onp.Array1D[np.int32])
assert_type(f.read_ints(np.int16), onp.Array1D[np.int16])
assert_type(f.read_reals(), onp.Array1D[np.float64])
assert_type(f.read_reals(np.float32), onp.Array1D[np.float32])
assert_type(f.close(), None)
