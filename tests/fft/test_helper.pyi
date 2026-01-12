from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.fft import fftfreq, next_fast_len, prev_fast_len, rfftfreq

###

# next_fast_len (same as prev_fast_len)
assert_type(next_fast_len(42), int)
assert_type(next_fast_len(42, True), int)
assert_type(next_fast_len(target=42, real=True), int)

# prev_fast_len (same as next_fast_len)
assert_type(prev_fast_len(42), int)
assert_type(prev_fast_len(42, True), int)
assert_type(prev_fast_len(target=42, real=True), int)

# fftfreq (same as rfftfreq)
assert_type(fftfreq(8), onp.Array1D[np.float64])
assert_type(fftfreq(8, 2.0), onp.Array1D[np.float64])
assert_type(fftfreq(8, d=2.0), onp.Array1D[np.float64])

# rfftfreq (same as fftfreq)
assert_type(rfftfreq(8), onp.Array1D[np.float64])
assert_type(rfftfreq(8, 2.0), onp.Array1D[np.float64])
assert_type(rfftfreq(8, d=2.0), onp.Array1D[np.float64])
