from typing import Literal, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import istft, spectrogram

###

_Float1D: TypeAlias = onp.Array1D[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float64]
_ComplexND: TypeAlias = onp.ArrayND[np.complex128]

###

_array_f8_1d: onp.Array[tuple[Literal[256]], np.float64]
_array_c16_1d: onp.Array[tuple[Literal[256]], np.complex128]
_spectrogram_mode_real: Literal["psd", "magnitude", "angle", "phase"]

###
# spectrogram

assert_type(spectrogram(_array_f8_1d), tuple[_Float1D, _Float1D, _FloatND])
assert_type(spectrogram(_array_f8_1d, mode=_spectrogram_mode_real), tuple[_Float1D, _Float1D, _FloatND])
assert_type(spectrogram(_array_f8_1d, mode="complex"), tuple[_Float1D, _Float1D, _ComplexND])
assert_type(
    spectrogram(_array_f8_1d, 1.0, ("tukey", 2.5), None, None, None, "constant", True, "density", -1, mode="complex"),
    tuple[_Float1D, _Float1D, _ComplexND],
)

###
# isft

assert_type(istft(_array_c16_1d), tuple[_Float1D, _FloatND])
assert_type(istft(_array_c16_1d, input_onesided=True), tuple[_Float1D, _FloatND])
assert_type(istft(_array_c16_1d, 1.0, "hann", 256, 128, 256, input_onesided=False), tuple[_Float1D, _ComplexND])
assert_type(
    istft(_array_c16_1d, input_onesided=False, fs=1.0, window="hann", nperseg=256, noverlap=128, nfft=256),
    tuple[_Float1D, _ComplexND],
)
assert_type(
    istft(
        _array_c16_1d,
        fs=2.0,
        window=("tukey", 0.25),
        nperseg=256,
        noverlap=128,
        nfft=256,
        input_onesided=True,
        boundary=False,
        time_axis=-1,
        freq_axis=0,
        scaling="spectrum",
    ),
    tuple[_Float1D, _FloatND],
)
assert_type(
    istft(
        _array_c16_1d,
        fs=2.0,
        window=("tukey", 0.25),
        nperseg=256,
        noverlap=128,
        nfft=256,
        input_onesided=False,
        boundary=False,
        time_axis=0,
        freq_axis=1,
        scaling="spectrum",
    ),
    tuple[_Float1D, _ComplexND],
)
