from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy import fft

###

_f64: np.float64 = ...
_f16_1d: onp.Array1D[np.float16] = ...
_f64_1d: onp.Array1D[np.float64] = ...
_f32_2d: onp.Array2D[np.float32] = ...
_f80_3d: onp.Array3D[np.float128] = ...

###

# fht
assert_type(fft.fht(_f64_1d, _f64, 1), onp.Array1D[np.float64])
assert_type(fft.fht(_f32_2d, _f64, 1), onp.Array2D[np.float32])
assert_type(fft.fht(_f80_3d, _f64, 1), onp.Array3D[np.float128])
assert_type(fft.fht([0.4], _f64, 1), onp.Array1D[np.float64])
assert_type(fft.fht([[0.1]], _f64, 1), onp.Array2D[np.float64])
assert_type(fft.fht([[[0.1]]], _f64, 1), onp.Array3D[np.float64])
assert_type(fft.fht(_f16_1d, _f64, 1), onp.ArrayND[np.float64 | Any, tuple[int] | tuple[Any, ...]])

# ifht
assert_type(fft.ifht(_f64_1d, _f64, 1), onp.Array1D[np.float64])
assert_type(fft.ifht(_f32_2d, _f64, 1), onp.Array2D[np.float32])
assert_type(fft.ifht(_f80_3d, _f64, 1), onp.Array3D[np.float128])
assert_type(fft.ifht([0.4], _f64, 1), onp.Array1D[np.float64])
assert_type(fft.ifht([[0.1]], _f64, 1), onp.Array2D[np.float64])
assert_type(fft.ifht([[[0.1]]], _f64, 1), onp.Array3D[np.float64])
assert_type(fft.ifht(_f16_1d, _f64, 1), onp.ArrayND[np.float64 | Any, tuple[int] | tuple[Any, ...]])

# fftoffset
assert_type(fft.fhtoffset(0.1, 2.0, 0.5, 0.0), np.float64)
