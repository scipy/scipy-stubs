# type-tests for `signal/_lti_conversion.pyi`

from typing import Any, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.signal._lti_conversion import abcd_normalize, cont2discrete, ss2tf, ss2zpk, tf2ss, zpk2ss
from scipy.signal._ltisys import (
    StateSpaceContinuous,
    StateSpaceDiscrete,
    TransferFunctionContinuous,
    TransferFunctionDiscrete,
    ZerosPolesGainContinuous,
    ZerosPolesGainDiscrete,
    dlti,
    lti,
)

###

f16_2d: onp.Array2D[np.float16]
f32_2d: onp.Array2D[np.float32]
f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
c64_2d: onp.Array2D[np.complex64]
c128_1d: onp.Array1D[np.complex128]
c128_2d: onp.Array2D[np.complex128]

_SS_f: TypeAlias = tuple[
    onp.Array2D[npc.floating], onp.Array2D[npc.floating], onp.Array2D[npc.floating], onp.Array2D[npc.floating]
]
_SS: TypeAlias = tuple[onp.Array2D[npc.inexact], onp.Array2D[npc.inexact], onp.Array2D[npc.inexact], onp.Array2D[npc.inexact]]
_TF_f: TypeAlias = tuple[onp.Array2D[npc.floating], onp.Array1D[npc.floating]]
_TF: TypeAlias = tuple[onp.Array2D[npc.inexact], onp.Array1D[npc.inexact]]
_ZPK_f: TypeAlias = tuple[onp.Array1D[npc.floating], onp.Array1D[npc.floating], float | np.float64]
_ZPK: TypeAlias = tuple[onp.Array1D[npc.inexact], onp.Array1D[npc.inexact], float | np.float64]

###
# abcd_normalize

assert_type(abcd_normalize(f64_2d, f64_2d, f64_2d, f64_2d), _TF_f)
assert_type(abcd_normalize(c128_2d, c128_2d, c128_2d, c128_2d), _TF)

###
# tf2ss

assert_type(tf2ss(f64_2d, f64_1d), _SS_f)
assert_type(tf2ss(c128_2d, c128_1d), _SS)

###
# ss2tf

assert_type(ss2tf(f64_2d, f64_2d, f64_2d, f64_2d), _TF_f)
assert_type(ss2tf(c128_2d, c128_2d, c128_2d, c128_2d), _TF)

###
# zpk2ss

assert_type(zpk2ss(f64_1d, f64_1d, 1.0), _SS_f)
assert_type(zpk2ss(c128_1d, c128_1d, 1.0), _SS)

###
# ss2zpk

assert_type(ss2zpk(f64_2d, f64_2d, f64_2d, f64_2d), _ZPK_f)
assert_type(ss2zpk(c128_2d, c128_2d, c128_2d, c128_2d), _ZPK)

###
# cont2discrete

assert_type(cont2discrete((f64_2d, f64_1d), 0.01), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64], float])
assert_type(
    cont2discrete((f64_1d, f64_1d, 0.5), 0.01), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64, float]
)
assert_type(
    cont2discrete((f64_2d, f64_2d, f64_2d, f64_2d), 0.01),
    tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64], float],
)

assert_type(cont2discrete((c128_2d, c128_1d), 0.01), tuple[onp.Array2D[np.complex128 | Any], onp.Array1D[np.float64], float])
assert_type(
    cont2discrete((c128_1d, c128_1d, 1j), 0.01),
    tuple[onp.Array1D[np.complex128 | Any], onp.Array1D[np.complex128 | Any], np.complex128 | Any, float],
)

assert_type(
    cont2discrete((f16_2d, f16_2d, f16_2d, f16_2d), 0.01),
    tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float16], onp.Array2D[np.float16], float],
)
assert_type(
    cont2discrete((f32_2d, f32_2d, f32_2d, f32_2d), 0.01),
    tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float32], onp.Array2D[np.float32], float],
)
assert_type(
    cont2discrete((c64_2d, c64_2d, c64_2d, c64_2d), 0.01),
    tuple[onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex64], onp.Array2D[np.complex64], float],
)
assert_type(
    cont2discrete((c128_2d, c128_2d, c128_2d, c128_2d), 0.01),
    tuple[onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex128], float],
)

tf_cont: TransferFunctionContinuous[np.float64]
zpk_cont: ZerosPolesGainContinuous[np.float64, np.float64]
ss_cont: StateSpaceContinuous[np.float64, np.float64]
lti_sys: lti[np.float64, np.float64]

assert_type(cont2discrete(tf_cont, 0.01), TransferFunctionDiscrete[np.float64])
assert_type(cont2discrete(zpk_cont, 0.01), ZerosPolesGainDiscrete[np.float64, np.float64])
assert_type(cont2discrete(ss_cont, 0.01), StateSpaceDiscrete[np.float64, np.float64])
assert_type(cont2discrete(lti_sys, 0.01), dlti[np.float64, np.float64])
