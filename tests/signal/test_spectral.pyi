# type-tests for `signal/_spectral_py.pyi`

from typing import Literal, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import check_COLA, check_NOLA, coherence, csd, istft, lombscargle, periodogram, spectrogram, stft, welch

###

_F64_1D: TypeAlias = onp.Array1D[np.float64]
_F32_ND: TypeAlias = onp.ArrayND[np.float32]
_F64_ND: TypeAlias = onp.ArrayND[np.float64]
_F80_ND: TypeAlias = onp.ArrayND[np.float96 | np.float128]
_C64_ND: TypeAlias = onp.ArrayND[np.complex64]
_C128_ND: TypeAlias = onp.ArrayND[np.complex128]
_C160_ND: TypeAlias = onp.ArrayND[np.complex192 | np.complex256]

###

_i64_1d: onp.Array1D[np.int64]
_f16_1d: onp.Array1D[np.float16]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f80_1d: onp.Array1D[np.float96 | np.float128]
_c64_1d: onp.Array1D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]
_c160_1d: onp.Array1D[np.complex192 | np.complex256]

_mode_real: Literal["psd", "magnitude", "angle", "phase"]

###

# lombscargle

assert_type(lombscargle(_i64_1d, _i64_1d, _i64_1d), _F64_1D)
assert_type(lombscargle(_f32_1d, _f32_1d, _f32_1d), _F64_1D)
assert_type(lombscargle(_f64_1d, _f64_1d, _f64_1d), _F64_1D)
assert_type(lombscargle(_f64_1d, _f64_1d, _f64_1d, precenter=False), _F64_1D)  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(lombscargle(_f64_1d, _f64_1d, _f64_1d, precenter=True), _F64_1D)  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

# periodogram

assert_type(periodogram(_i64_1d), tuple[_F64_1D, _F64_ND])
assert_type(periodogram(_f16_1d), tuple[_F64_1D, _F32_ND])
assert_type(periodogram(_f32_1d), tuple[_F64_1D, _F32_ND])
assert_type(periodogram(_f64_1d), tuple[_F64_1D, _F64_ND])
assert_type(periodogram(_f80_1d), tuple[_F64_1D, _F80_ND])
assert_type(periodogram(_c64_1d), tuple[_F64_1D, _F32_ND])
assert_type(periodogram(_c128_1d), tuple[_F64_1D, _F64_ND])
assert_type(periodogram(_c160_1d), tuple[_F64_1D, _F80_ND])

# welch

assert_type(welch(_i64_1d), tuple[_F64_1D, _F64_ND])
assert_type(welch(_f16_1d), tuple[_F64_1D, _F32_ND])
assert_type(welch(_f32_1d), tuple[_F64_1D, _F32_ND])
assert_type(welch(_f64_1d), tuple[_F64_1D, _F64_ND])
assert_type(welch(_f80_1d), tuple[_F64_1D, _F80_ND])
assert_type(welch(_c64_1d), tuple[_F64_1D, _F32_ND])
assert_type(welch(_c128_1d), tuple[_F64_1D, _F64_ND])
assert_type(welch(_c160_1d), tuple[_F64_1D, _F80_ND])

# csd

assert_type(csd(_i64_1d, _i64_1d), tuple[_F64_1D, _C128_ND])
assert_type(csd(_f16_1d, _f16_1d), tuple[_F64_1D, _C64_ND])
assert_type(csd(_f32_1d, _f32_1d), tuple[_F64_1D, _C64_ND])
assert_type(csd(_f64_1d, _f64_1d), tuple[_F64_1D, _C128_ND])
assert_type(csd(_f80_1d, _f80_1d), tuple[_F64_1D, _C160_ND])
assert_type(csd(_c64_1d, _c64_1d), tuple[_F64_1D, _C64_ND])
assert_type(csd(_c128_1d, _c128_1d), tuple[_F64_1D, _C128_ND])
assert_type(csd(_c160_1d, _c160_1d), tuple[_F64_1D, _C160_ND])

# spectrogram

assert_type(spectrogram(_f64_1d), tuple[_F64_1D, _F64_1D, _F64_ND])
assert_type(spectrogram(_f64_1d, mode=_mode_real), tuple[_F64_1D, _F64_1D, _F64_ND])
assert_type(spectrogram(_f64_1d, mode="complex"), tuple[_F64_1D, _F64_1D, _C128_ND])
assert_type(
    spectrogram(_f64_1d, 1.0, ("tukey", 2.5), None, None, None, "constant", True, "density", -1, mode="complex"),
    tuple[_F64_1D, _F64_1D, _C128_ND],
)

# check_{COLA,NOLA}

assert_type(check_COLA(256, 128, 256), np.bool_)
assert_type(check_NOLA(256, 128, 256), np.bool_)

# stft

assert_type(stft(_i64_1d), tuple[_F64_1D, _F64_1D, _C128_ND])
assert_type(stft(_f16_1d), tuple[_F64_1D, _F64_1D, _C64_ND])
assert_type(stft(_f32_1d), tuple[_F64_1D, _F64_1D, _C64_ND])
assert_type(stft(_f64_1d), tuple[_F64_1D, _F64_1D, _C128_ND])
assert_type(stft(_f80_1d), tuple[_F64_1D, _F64_1D, _C160_ND])
assert_type(stft(_c64_1d), tuple[_F64_1D, _F64_1D, _C64_ND])
assert_type(stft(_c128_1d), tuple[_F64_1D, _F64_1D, _C128_ND])
assert_type(stft(_c160_1d), tuple[_F64_1D, _F64_1D, _C160_ND])

# istft

assert_type(istft(_i64_1d), tuple[_F64_1D, _F64_ND])
assert_type(istft(_f16_1d), tuple[_F64_1D, _F32_ND])
assert_type(istft(_f32_1d), tuple[_F64_1D, _F32_ND])
assert_type(istft(_f64_1d), tuple[_F64_1D, _F64_ND])
assert_type(istft(_f80_1d), tuple[_F64_1D, _F80_ND])
assert_type(istft(_c64_1d), tuple[_F64_1D, _F32_ND])
assert_type(istft(_c128_1d), tuple[_F64_1D, _F64_ND])
assert_type(istft(_c160_1d), tuple[_F64_1D, _F80_ND])

assert_type(istft(_i64_1d, input_onesided=False), tuple[_F64_1D, _C128_ND])
assert_type(istft(_f16_1d, input_onesided=False), tuple[_F64_1D, _C64_ND])
assert_type(istft(_f32_1d, input_onesided=False), tuple[_F64_1D, _C64_ND])
assert_type(istft(_f64_1d, input_onesided=False), tuple[_F64_1D, _C128_ND])
assert_type(istft(_f80_1d, input_onesided=False), tuple[_F64_1D, _C160_ND])
assert_type(istft(_c64_1d, input_onesided=False), tuple[_F64_1D, _C64_ND])
assert_type(istft(_c128_1d, input_onesided=False), tuple[_F64_1D, _C128_ND])
assert_type(istft(_c160_1d, input_onesided=False), tuple[_F64_1D, _C160_ND])

# coherence

assert_type(coherence(_i64_1d, _i64_1d), tuple[_F64_1D, _F64_ND])
assert_type(coherence(_f16_1d, _f16_1d), tuple[_F64_1D, _F32_ND])
assert_type(coherence(_f32_1d, _f32_1d), tuple[_F64_1D, _F32_ND])
assert_type(coherence(_f64_1d, _f64_1d), tuple[_F64_1D, _F64_ND])
assert_type(coherence(_f80_1d, _f80_1d), tuple[_F64_1D, _F80_ND])
assert_type(coherence(_c64_1d, _c64_1d), tuple[_F64_1D, _F32_ND])
assert_type(coherence(_c128_1d, _c128_1d), tuple[_F64_1D, _F64_ND])
assert_type(coherence(_c160_1d, _c160_1d), tuple[_F64_1D, _F80_ND])
