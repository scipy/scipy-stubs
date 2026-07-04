# type-tests for `odr/_odrpack.pyi`

from typing_extensions import assert_type

import numpy as np
import optype.numpy as onp
from scipy.odr import Output

###
_out: Output
assert_type(_out.iwork, onp.Array1D[np.int32 | np.int64])
