# type-tests for `signal/_max_len_seq.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import max_len_seq

###

_i64_1d: onp.Array1D[np.int64]

###

assert_type(max_len_seq(4), tuple[onp.Array1D[np.int8], onp.Array1D[np.int8]])
assert_type(max_len_seq(4, _i64_1d), tuple[onp.Array1D[np.int8], onp.Array1D[np.int8]])
assert_type(max_len_seq(4, length=3), tuple[onp.Array1D[np.int8], onp.Array1D[np.int8]])
assert_type(max_len_seq(4, taps=_i64_1d), tuple[onp.Array1D[np.int8], onp.Array1D[np.int8]])
