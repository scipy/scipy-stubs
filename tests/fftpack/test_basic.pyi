# type-tests for `fftpack/_basic.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.fftpack import fft, fft2, fftn, rfft

###

_f32_nd: onp.ArrayND[np.float32]
_f64_nd: onp.ArrayND[np.float64]

###

# fft (same as ifft)
assert_type(fft(_f64_nd), onp.ArrayND[np.complex128])
assert_type(fft(_f32_nd), onp.ArrayND[np.complex128 | Any])

# rfft (same as irfft)
assert_type(rfft(_f64_nd), onp.ArrayND[np.float64])
assert_type(rfft(_f32_nd), onp.ArrayND[np.float64 | Any])

# fft2 (same as ifft2)
assert_type(fft2(_f64_nd), onp.ArrayND[np.complex128])
assert_type(fft2(_f32_nd), onp.ArrayND[np.complex128 | Any])

# fftn (same as ifftn)
assert_type(fftn(_f64_nd), onp.ArrayND[np.complex128])
assert_type(fftn(_f32_nd), onp.ArrayND[np.complex128 | Any])
