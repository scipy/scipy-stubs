from typing import assert_type

import numpy as np

from scipy.fft import fht, fhtoffset, ifht

f64_0d: np.float64 = ...
f64_1d: np.ndarray[tuple[int], np.dtype[np.float64]] = ...
f32_2d: np.ndarray[tuple[int, int], np.dtype[np.float32]] = ...
f80_3d: np.ndarray[tuple[int, int, int], np.dtype[np.longdouble]] = ...

###

# fftoffset
assert_type(fhtoffset(0.1, 2.0, 0.5, 0.0), np.float64)

# fht
assert_type(fht(f64_1d, f64_0d, 1), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(fht(f32_2d, f64_0d, 1), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(fht(f80_3d, f64_0d, 1), np.ndarray[tuple[int, int, int], np.dtype[np.longdouble]])
assert_type(fht([0.4], f64_0d, 1), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(fht([[0.1]], f64_0d, 1), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(fht([[[0.1]]], f64_0d, 1), np.ndarray[tuple[int, int, int], np.dtype[np.float64]])

# ifht
assert_type(ifht(f64_1d, f64_0d, 1), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(ifht(f32_2d, f64_0d, 1), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(ifht(f80_3d, f64_0d, 1), np.ndarray[tuple[int, int, int], np.dtype[np.longdouble]])
assert_type(ifht([0.4], f64_0d, 1), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(ifht([[0.1]], f64_0d, 1), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(ifht([[[0.1]]], f64_0d, 1), np.ndarray[tuple[int, int, int], np.dtype[np.float64]])
