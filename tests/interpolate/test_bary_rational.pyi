# type-tests for `interpolate/_bary_rational.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.interpolate import AAA, FloaterHormannInterpolator

###

_i64_1d: onp.Array1D[np.int64]
_f16_1d: onp.Array1D[np.float16]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f80_1d: onp.Array1D[npc.floating80]
_c64_1d: onp.Array1D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]
_c160_1d: onp.Array1D[npc.complexfloating160]

###

# AAA

_aaa: AAA[np.float32]

assert_type(AAA(_i64_1d, _i64_1d), AAA[np.float64])
assert_type(AAA(_f16_1d, _f16_1d), AAA[np.float64])
assert_type(AAA(_f32_1d, _f32_1d), AAA[np.float32])
assert_type(AAA(_f64_1d, _f64_1d), AAA[np.float64])
assert_type(AAA(_f80_1d, _f80_1d), AAA[npc.floating80])
assert_type(AAA(_c64_1d, _c64_1d), AAA[np.complex64])
assert_type(AAA(_c128_1d, _c128_1d), AAA[np.complex128])
assert_type(AAA(_c160_1d, _c160_1d), AAA[npc.complexfloating160])

assert_type(_aaa(1), onp.Array1D[np.float32])
assert_type(_aaa(1.0), onp.Array1D[np.float64])
assert_type(_aaa(1j), onp.Array1D[np.complex128])

assert_type(_aaa.weights, onp.Array1D[np.float32])
assert_type(_aaa.support_values, onp.Array1D[np.float32])
assert_type(_aaa.support_values, onp.Array1D[np.float32])
assert_type(_aaa.residues(), onp.Array1D[np.float32])
assert_type(_aaa.poles(), onp.Array1D[np.complex64])
assert_type(_aaa.roots(), onp.Array1D[np.complex64])

# FloaterHormannInterpolator

_fhi: FloaterHormannInterpolator[np.float32, tuple[int, int]]

assert_type(FloaterHormannInterpolator(_i64_1d, _i64_1d), FloaterHormannInterpolator[np.float64, tuple[int]])
assert_type(FloaterHormannInterpolator(_f16_1d, _f16_1d), FloaterHormannInterpolator[np.float64, tuple[int]])
assert_type(FloaterHormannInterpolator(_f32_1d, _f32_1d), FloaterHormannInterpolator[np.float32, tuple[int]])
assert_type(FloaterHormannInterpolator(_f64_1d, _f64_1d), FloaterHormannInterpolator[np.float64, tuple[int]])
# pyrefly: ignore[assert-type]
assert_type(FloaterHormannInterpolator(_f80_1d, _f80_1d), FloaterHormannInterpolator[npc.floating80, tuple[int]])
assert_type(FloaterHormannInterpolator(_c64_1d, _c64_1d), FloaterHormannInterpolator[np.complex64, tuple[int]])
assert_type(FloaterHormannInterpolator(_c128_1d, _c128_1d), FloaterHormannInterpolator[np.complex128, tuple[int]])
# pyrefly: ignore[assert-type]
assert_type(FloaterHormannInterpolator(_c160_1d, _c160_1d), FloaterHormannInterpolator[npc.complexfloating160, tuple[int]])

assert_type(_fhi(1), onp.Array2D[np.float32])
assert_type(_fhi(1.0), onp.Array2D[np.float64])
assert_type(_fhi(1j), onp.Array2D[np.complex128])

assert_type(_fhi.residues(), onp.Array2D[np.float32])
assert_type(_fhi.poles(), onp.Array2D[np.complex64])
assert_type(_fhi.roots(), onp.Array2D[np.complex64])
