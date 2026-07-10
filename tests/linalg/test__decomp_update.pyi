# type-tests for `linalg/_decomp_update.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import qr_delete, qr_insert, qr_update

###

f64_nd: onp.ArrayND[np.float64]
c128_nd: onp.ArrayND[np.complex128]

type _FloatQR = tuple[onp.ArrayND[np.float32 | np.float64], onp.ArrayND[np.float32 | np.float64]]
type _ComplexQR = (
    tuple[onp.ArrayND[np.float32 | np.float64], onp.ArrayND[np.float32 | np.float64]]
    | tuple[onp.ArrayND[np.complex64 | np.complex128], onp.ArrayND[np.complex64 | np.complex128]]
)

###
# qr_delete

assert_type(qr_delete(f64_nd, f64_nd, 0), _FloatQR)
assert_type(qr_delete(c128_nd, c128_nd, 0), _ComplexQR)

###
# qr_insert

assert_type(qr_insert(f64_nd, f64_nd, f64_nd, 0), _FloatQR)
assert_type(qr_insert(c128_nd, c128_nd, c128_nd, 0), _ComplexQR)

###
# qr_update

assert_type(qr_update(f64_nd, f64_nd, f64_nd, f64_nd), _FloatQR)
assert_type(qr_update(c128_nd, c128_nd, c128_nd, c128_nd), _ComplexQR)
