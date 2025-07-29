from typing import Literal, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.signal import istft, spectrogram

_DoubleND: TypeAlias = onp.ArrayND[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float32 | np.float64 | np.longdouble]
_ComplexND: TypeAlias = onp.ArrayND[npc.complexfloating]

array_f8_1d: onp.Array[tuple[Literal[256]], np.float64]
array_c16_1d: onp.Array[tuple[Literal[256]], np.complex128]

spectrogram_mode_real: Literal["psd", "magnitude", "angle", "phase"]

# test spectrogram function overloads
assert_type(spectrogram(array_f8_1d), tuple[_DoubleND, _DoubleND, _FloatND])
assert_type(spectrogram(array_f8_1d, mode=spectrogram_mode_real), tuple[_DoubleND, _DoubleND, _FloatND])
assert_type(spectrogram(array_f8_1d, mode="complex"), tuple[_DoubleND, _DoubleND, _ComplexND])
assert_type(
    spectrogram(array_f8_1d, 1.0, ("tukey", 2.5), None, None, None, "constant", True, "density", -1, "complex"),
    tuple[_DoubleND, _DoubleND, _ComplexND],
)

# test isft function overloads
assert_type(istft(array_c16_1d), tuple[_DoubleND, _FloatND])
assert_type(istft(array_c16_1d, input_onesided=True), tuple[_DoubleND, _FloatND])
assert_type(istft(array_c16_1d, 1.0, "hann", 256, 128, 256, False), tuple[_DoubleND, _ComplexND])
assert_type(
    istft(array_c16_1d, input_onesided=False, fs=1.0, window="hann", nperseg=256, noverlap=128, nfft=256),
    tuple[_DoubleND, _ComplexND],
)
assert_type(
    istft(
        array_c16_1d,
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
    tuple[_DoubleND, _FloatND],
)
assert_type(
    istft(
        array_c16_1d,
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
    tuple[_DoubleND, _ComplexND],
)
