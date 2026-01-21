# type-tests for `signal/_max_len_seq.pyi`

from typing import Any, Literal, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.signal import besselap, buttap, butter, buttord, cheb1ap, cheb1ord, cheb2ap, cheb2ord, ellipap, ellipord

###

_f64_1d: onp.Array1D[np.float64]
_f80_1d: onp.Array1D[npc.floating80]

###

# butter

assert_type(butter(8, 0.1), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(butter(8, 0.1, output="ba"), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(butter(8, 0.1, output="zpk"), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128], float])
assert_type(butter(8, 0.1, output="sos"), onp.Array2D[np.float64])

# buttord

assert_type(buttord(0.2, 0.3, 3, 40), tuple[int, np.float64])
assert_type(buttord(0.2, _f64_1d, 3, 40), tuple[int, np.float64])
assert_type(buttord(0.2, _f80_1d, 3, 40), tuple[int, np.longdouble])
assert_type(buttord(_f64_1d, 0.3, 3, 40), tuple[int, onp.Array1D[np.float64]])

# cheb1ord

assert_type(cheb1ord(0.2, 0.3, 3, 40), tuple[int, np.float64])
assert_type(cheb1ord(0.2, _f64_1d, 3, 40), tuple[int, np.float64])
assert_type(cheb1ord(0.2, _f80_1d, 3, 40), tuple[int, np.longdouble])
assert_type(cheb1ord(_f64_1d, 0.3, 3, 40), tuple[int, onp.Array1D[np.float64]])

# cheb2ord

assert_type(cheb2ord(0.2, 0.3, 3, 40), tuple[int, np.float64])
assert_type(cheb2ord(0.2, _f64_1d, 3, 40), tuple[int, np.float64])
assert_type(cheb2ord(0.2, _f80_1d, 3, 40), tuple[int, np.longdouble])
assert_type(cheb2ord(_f64_1d, 0.3, 3, 40), tuple[int, onp.Array1D[np.float64]])

# ellipord

assert_type(ellipord(0.2, 0.3, 3, 40), tuple[int, np.float64])
assert_type(ellipord(0.2, _f64_1d, 3, 40), tuple[int, np.float64])
assert_type(ellipord(_f64_1d, 0.3, 3, 40), tuple[int, onp.Array1D[np.float64]])

# buttap

assert_type(buttap(4), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128], Literal[1]])
assert_type(buttap(4, xp=np), tuple[Any, Any, Literal[1]])

# cheb1ap

assert_type(cheb1ap(4, 0.1), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128], float])
assert_type(cheb1ap(4, 0.1, xp=np), tuple[Any, Any, float])

# cheb2ap

assert_type(cheb2ap(4, 0.1), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], float])
assert_type(cheb2ap(4, 0.1, xp=np), tuple[Any, Any, float])

# ellipap

assert_type(ellipap(4, 0.1, 0.2), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], float])
assert_type(ellipap(4, 0.1, 0.2, xp=np), tuple[Any, Any, float])

# besselap

assert_type(besselap(4), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128], float])
assert_type(besselap(4, xp=np), tuple[Any, Any, float])
