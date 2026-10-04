# type-tests for `fftpack/_pseudo_diffs.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.fftpack import cs_diff, diff, hilbert, shift, tilbert

###

_c64_1d: onp.Array1D[np.complex64]

###

# diff
assert_type(diff(_c64_1d), onp.Array1D[np.complex128 | Any])

# tilbert (same as itilbert)
assert_type(tilbert(_c64_1d, 1.0), onp.Array1D[np.complex128 | Any])

# hilbert (same as ihilbert)
assert_type(hilbert(_c64_1d), onp.Array1D[np.complex128 | Any])

# cs_diff (same as sc_diff, ss_diff, cc_diff)
assert_type(cs_diff(_c64_1d, 1.0, 2.0), onp.Array1D[np.complex128 | Any])

# shift
assert_type(shift(_c64_1d, 1.0), onp.Array1D[np.complex128 | Any])
