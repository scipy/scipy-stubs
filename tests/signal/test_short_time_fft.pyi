# type-tests for `signal/_short_time_fft.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import ShortTimeFFT, closest_STFT_dual_window

###

f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
c128_1d: onp.Array1D[np.complex128]

###
# ShortTimeFFT.from_window / from_dual / from_win_equals_dual

stft_f64 = ShortTimeFFT.from_window("hann", 1.0, 64, 32)
assert_type(stft_f64, ShortTimeFFT[np.float64])

stft_c128 = ShortTimeFFT(c128_1d, hop=32, fs=1.0)
assert_type(stft_c128, ShortTimeFFT[np.complex128])

###
# properties

assert_type(stft_f64.win, onp.Array1D[np.float64])
assert_type(stft_f64.dual_win, onp.Array1D[np.float64])
assert_type(stft_f64.hop, int)
assert_type(stft_f64.f, onp.Array1D[np.float64])
assert_type(stft_f64.invertible, bool)
assert_type(stft_f64.f_pts, int)
assert_type(stft_f64.m_num, int)
assert_type(stft_f64.delta_t, float)
assert_type(stft_f64.delta_f, float)

###
# t / k_max / p_range / upper_border_begin / lower_border_end

assert_type(stft_f64.t(128), onp.Array1D[np.float64])
assert_type(stft_f64.k_max(128), int)
assert_type(stft_f64.p_num(128), int)
assert_type(stft_f64.p_range(128), tuple[int, int])
assert_type(stft_f64.upper_border_begin(128), tuple[int, int])
assert_type(stft_f64.lower_border_end, tuple[int, int])

###
# stft

assert_type(stft_f64.stft(f64_1d), onp.Array2D[np.complex128])
assert_type(stft_f64.stft(f64_2d), onp.Array3D[np.complex128])

###
# stft_detrend

assert_type(stft_f64.stft_detrend(f64_1d, "constant"), onp.Array2D[np.complex128])
assert_type(stft_f64.stft_detrend(f64_2d, None), onp.Array3D[np.complex128])

###
# spectrogram

assert_type(stft_f64.spectrogram(f64_1d), onp.Array2D[np.float64])
assert_type(stft_f64.spectrogram(f64_1d, f64_1d), onp.Array2D[np.complex128])
assert_type(stft_f64.spectrogram(f64_2d), onp.Array3D[np.float64])
assert_type(stft_f64.spectrogram(f64_2d, f64_2d), onp.Array3D[np.complex128])

###
# istft

assert_type(stft_f64.istft(f64_2d), onp.ArrayND[np.complex128])

###
# extent

assert_type(stft_f64.extent(128), tuple[float, float, float, float])

###
# closest_STFT_dual_window

assert_type(closest_STFT_dual_window(f64_1d, 32, scaled=True), tuple[onp.Array1D[np.float64], np.float64])
assert_type(closest_STFT_dual_window(f64_1d, 32, scaled=False), tuple[onp.Array1D[np.float64], float])
