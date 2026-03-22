# type-tests for `signal/_lti_conversion.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.signal import abcd_normalize, cont2discrete, dlti, lti, ss2tf, ss2zpk, tf2ss, zpk2ss
from scipy.signal._ltisys import (
    StateSpaceContinuous,
    StateSpaceDiscrete,
    TransferFunctionContinuous,
    TransferFunctionDiscrete,
    ZerosPolesGainContinuous,
    ZerosPolesGainDiscrete,
)

###

_f16_2d: onp.Array2D[np.float16]
_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_c64_1d: onp.Array1D[np.complex64]
_c64_2d: onp.Array2D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]
_c128_2d: onp.Array2D[np.complex128]

_tf_c_f64: TransferFunctionContinuous[np.float64]
_zpk_c_f64: ZerosPolesGainContinuous[np.float64, np.float64]
_ss_c_f64: StateSpaceContinuous[np.float64, np.float64]
_lti_c_f64: lti[np.float64, np.float64]

###

# abcd_normalize

assert_type(abcd_normalize(_f64_2d, _f64_2d, _f64_2d, _f64_2d), tuple[onp.Array2D[npc.floating], onp.Array1D[npc.floating]])
assert_type(abcd_normalize(_c128_2d, _c128_2d, _c128_2d, _c128_2d), tuple[onp.Array2D[npc.inexact], onp.Array1D[npc.inexact]])

# tf2ss

assert_type(
    tf2ss(_f64_2d, _f32_1d),
    tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64]],
)
assert_type(
    tf2ss(_f64_2d, _f64_1d),
    tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64]],
)
assert_type(
    tf2ss(_c128_2d, _c64_1d),
    tuple[onp.Array2D[np.complex128], onp.Array2D[np.float64], onp.Array2D[np.complex128], onp.Array2D[np.complex128]],
)
assert_type(
    tf2ss(_c128_2d, _c128_1d),
    tuple[onp.Array2D[np.complex128], onp.Array2D[np.float64], onp.Array2D[np.complex128], onp.Array2D[np.complex128]],
)

# ss2tf

assert_type(ss2tf(_f64_2d, _f64_2d, _f64_2d, _f64_2d), tuple[onp.Array2D[npc.floating], onp.Array1D[npc.floating]])
assert_type(ss2tf(_c128_2d, _c128_2d, _c128_2d, _c128_2d), tuple[onp.Array2D[npc.inexact], onp.Array1D[npc.inexact]])

# zpk2ss

assert_type(
    zpk2ss(_f64_1d, _f64_1d, 1.0),
    tuple[onp.Array2D[npc.floating], onp.Array2D[npc.floating], onp.Array2D[npc.floating], onp.Array2D[npc.floating]],
)
assert_type(
    zpk2ss(_c128_1d, _c128_1d, 1.0),
    tuple[onp.Array2D[npc.inexact], onp.Array2D[npc.inexact], onp.Array2D[npc.inexact], onp.Array2D[npc.inexact]],
)

# ss2zpk

assert_type(
    ss2zpk(_f64_2d, _f64_2d, _f64_2d, _f64_2d), tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating], float | np.float64]
)
assert_type(
    ss2zpk(_c128_2d, _c128_2d, _c128_2d, _c128_2d), tuple[onp.Array1D[npc.inexact], onp.Array1D[npc.inexact], float | np.float64]
)

# cont2discrete

assert_type(cont2discrete((_f64_2d, _f64_1d), 0.01), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64], float])
assert_type(
    cont2discrete((_f64_1d, _f64_1d, 0.5), 0.01), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64, float]
)
assert_type(
    cont2discrete((_f64_2d, _f64_2d, _f64_2d, _f64_2d), 0.01),
    tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64], float],
)
assert_type(cont2discrete((_c128_2d, _c128_1d), 0.01), tuple[onp.Array2D[np.complex128 | Any], onp.Array1D[np.float64], float])
assert_type(
    cont2discrete((_c128_1d, _c128_1d, 1j), 0.01),
    tuple[onp.Array1D[np.complex128 | Any], onp.Array1D[np.complex128 | Any], np.complex128 | Any, float],
)

assert_type(
    cont2discrete((_f16_2d, _f16_2d, _f16_2d, _f16_2d), 0.01),
    tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float16], onp.Array2D[np.float16], float],
)
assert_type(
    cont2discrete((_f32_2d, _f32_2d, _f32_2d, _f32_2d), 0.01),
    tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float32], onp.Array2D[np.float32], float],
)
assert_type(
    cont2discrete((_c64_2d, _c64_2d, _c64_2d, _c64_2d), 0.01),
    tuple[onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex64], onp.Array2D[np.complex64], float],
)
assert_type(
    cont2discrete((_c128_2d, _c128_2d, _c128_2d, _c128_2d), 0.01),
    tuple[onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex128], float],
)

assert_type(cont2discrete(_tf_c_f64, 0.01), TransferFunctionDiscrete[np.float64])
assert_type(cont2discrete(_zpk_c_f64, 0.01), ZerosPolesGainDiscrete[np.float64, np.float64])
assert_type(cont2discrete(_ss_c_f64, 0.01), StateSpaceDiscrete[np.float64, np.float64])
assert_type(cont2discrete(_lti_c_f64, 0.01), dlti[np.float64, np.float64])
