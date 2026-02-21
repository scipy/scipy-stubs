import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.fft import (
    fft,
    fft2,
    fftn,
    hfft,
    hfft2,
    hfftn,
    ifft,
    ifft2,
    ifftn,
    ihfft,
    ihfft2,
    ihfftn,
    irfft,
    irfft2,
    irfftn,
    rfft,
    rfft2,
    rfftn,
)

###

int_1d: list[int]
int_2d: list[list[int]]
int_3d: list[list[list[int]]]

float_1d: list[float]
float_2d: list[list[float]]
float_3d: list[list[list[float]]]

complex_1d: list[complex]
complex_2d: list[list[complex]]
complex_3d: list[list[list[complex]]]

i16_1d: onp.Array1D[np.int16]
i16_2d: onp.Array2D[np.int16]
i16_3d: onp.Array3D[np.int16]

f32_1d: onp.Array1D[np.float32]
f32_2d: onp.Array2D[np.float32]
f32_3d: onp.Array3D[np.float32]

f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]

c64_1d: onp.Array1D[np.complex64]
c64_2d: onp.Array2D[np.complex64]
c64_3d: onp.Array3D[np.complex64]

c128_1d: onp.Array1D[np.complex128]
c128_2d: onp.Array2D[np.complex128]
c128_3d: onp.Array3D[np.complex128]

# NOTE: These extended precision types may not exist at runtime, but are used
# here to work around `[c]longdouble` issues on `numpy<2.2`

f80_1d: onp.Array1D[np.float128]
f80_2d: onp.Array2D[np.float128]
f80_3d: onp.Array3D[np.float128]

c160_1d: onp.Array1D[np.complex256]
c160_2d: onp.Array2D[np.complex256]
c160_3d: onp.Array3D[np.complex256]

###

###

# fft (same as ifft)
assert_subtype[onp.Array1D[np.complex128]](fft(int_1d))
assert_subtype[onp.Array1D[np.complex128]](fft(float_1d))
assert_subtype[onp.Array1D[np.complex128]](fft(complex_1d))
assert_subtype[onp.Array1D[np.complex128]](fft(i16_1d))
assert_subtype[onp.Array1D[np.complex64]](fft(f32_1d))
assert_subtype[onp.Array1D[np.complex128]](fft(f64_1d))
assert_subtype[onp.Array1D[np.clongdouble]](fft(f80_1d))
assert_subtype[onp.Array1D[np.complex64]](fft(c64_1d))
assert_subtype[onp.Array1D[np.complex128]](fft(c128_1d))
assert_subtype[onp.Array1D[np.clongdouble]](fft(c160_1d))
assert_subtype[onp.Array2D[np.complex128]](fft(int_2d))
assert_subtype[onp.Array2D[np.complex128]](fft(float_2d))
assert_subtype[onp.Array2D[np.complex128]](fft(complex_2d))
assert_subtype[onp.Array2D[np.complex128]](fft(i16_2d))
assert_subtype[onp.Array2D[np.complex64]](fft(f32_2d))
assert_subtype[onp.Array2D[np.complex128]](fft(f64_2d))
assert_subtype[onp.Array2D[np.clongdouble]](fft(f80_2d))
assert_subtype[onp.Array2D[np.complex64]](fft(c64_2d))
assert_subtype[onp.Array2D[np.complex128]](fft(c128_2d))
assert_subtype[onp.Array2D[np.clongdouble]](fft(c160_2d))

# ifft (same as fft)
assert_subtype[onp.Array1D[np.complex128]](ifft(int_1d))
assert_subtype[onp.Array1D[np.complex128]](ifft(float_1d))
assert_subtype[onp.Array1D[np.complex128]](ifft(complex_1d))
assert_subtype[onp.Array1D[np.complex128]](ifft(i16_1d))
assert_subtype[onp.Array1D[np.complex64]](ifft(f32_1d))
assert_subtype[onp.Array1D[np.complex128]](ifft(f64_1d))
assert_subtype[onp.Array1D[np.clongdouble]](ifft(f80_1d))
assert_subtype[onp.Array1D[np.complex64]](ifft(c64_1d))
assert_subtype[onp.Array1D[np.complex128]](ifft(c128_1d))
assert_subtype[onp.Array1D[np.clongdouble]](ifft(c160_1d))
assert_subtype[onp.Array2D[np.complex128]](ifft(int_2d))
assert_subtype[onp.Array2D[np.complex128]](ifft(float_2d))
assert_subtype[onp.Array2D[np.complex128]](ifft(complex_2d))
assert_subtype[onp.Array2D[np.complex128]](ifft(i16_2d))
assert_subtype[onp.Array2D[np.complex64]](ifft(f32_2d))
assert_subtype[onp.Array2D[np.complex128]](ifft(f64_2d))
assert_subtype[onp.Array2D[np.clongdouble]](ifft(f80_2d))
assert_subtype[onp.Array2D[np.complex64]](ifft(c64_2d))
assert_subtype[onp.Array2D[np.complex128]](ifft(c128_2d))
assert_subtype[onp.Array2D[np.clongdouble]](ifft(c160_2d))

# rfft (same as ihfft)
assert_subtype[onp.Array1D[np.complex128]](rfft(int_1d))
assert_subtype[onp.Array1D[np.complex128]](rfft(float_1d))
rfft(complex_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array1D[np.complex128]](rfft(i16_1d))
assert_subtype[onp.Array1D[np.complex64]](rfft(f32_1d))
assert_subtype[onp.Array1D[np.complex128]](rfft(f64_1d))
assert_subtype[onp.Array1D[np.clongdouble]](rfft(f80_1d))
rfft(c64_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfft(c128_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfft(c160_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array2D[np.complex128]](rfft(int_2d))
assert_subtype[onp.Array2D[np.complex128]](rfft(float_2d))
rfft(complex_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array2D[np.complex128]](rfft(i16_2d))
assert_subtype[onp.Array2D[np.complex64]](rfft(f32_2d))
assert_subtype[onp.Array2D[np.complex128]](rfft(f64_2d))
assert_subtype[onp.Array2D[np.clongdouble]](rfft(f80_2d))
rfft(c64_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfft(c128_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfft(c160_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]

# irfft (same as hfft)
assert_subtype[onp.Array1D[np.float64]](irfft(int_1d))
assert_subtype[onp.Array1D[np.float64]](irfft(float_1d))
assert_subtype[onp.Array1D[np.float64]](irfft(complex_1d))
assert_subtype[onp.Array1D[np.float64]](irfft(i16_1d))
assert_subtype[onp.Array1D[np.float32]](irfft(f32_1d))
assert_subtype[onp.Array1D[np.float64]](irfft(f64_1d))
assert_subtype[onp.Array1D[np.longdouble]](irfft(f80_1d))
assert_subtype[onp.Array1D[np.float32]](irfft(c64_1d))
assert_subtype[onp.Array1D[np.float64]](irfft(c128_1d))
assert_subtype[onp.Array1D[np.longdouble]](irfft(c160_1d))
assert_subtype[onp.Array2D[np.float64]](irfft(int_2d))
assert_subtype[onp.Array2D[np.float64]](irfft(float_2d))
assert_subtype[onp.Array2D[np.float64]](irfft(complex_2d))
assert_subtype[onp.Array2D[np.float64]](irfft(i16_2d))
assert_subtype[onp.Array2D[np.float32]](irfft(f32_2d))
assert_subtype[onp.Array2D[np.float64]](irfft(f64_2d))
assert_subtype[onp.Array2D[np.longdouble]](irfft(f80_2d))
assert_subtype[onp.Array2D[np.float32]](irfft(c64_2d))
assert_subtype[onp.Array2D[np.float64]](irfft(c128_2d))
assert_subtype[onp.Array2D[np.longdouble]](irfft(c160_2d))

# hfft (same as irfft)
assert_subtype[onp.Array1D[np.float64]](hfft(int_1d))
assert_subtype[onp.Array1D[np.float64]](hfft(float_1d))
assert_subtype[onp.Array1D[np.float64]](hfft(complex_1d))
assert_subtype[onp.Array1D[np.float64]](hfft(i16_1d))
assert_subtype[onp.Array1D[np.float32]](hfft(f32_1d))
assert_subtype[onp.Array1D[np.float64]](hfft(f64_1d))
assert_subtype[onp.Array1D[np.longdouble]](hfft(f80_1d))
assert_subtype[onp.Array1D[np.float32]](hfft(c64_1d))
assert_subtype[onp.Array1D[np.float64]](hfft(c128_1d))
assert_subtype[onp.Array1D[np.longdouble]](hfft(c160_1d))
assert_subtype[onp.Array2D[np.float64]](hfft(int_2d))
assert_subtype[onp.Array2D[np.float64]](hfft(float_2d))
assert_subtype[onp.Array2D[np.float64]](hfft(complex_2d))
assert_subtype[onp.Array2D[np.float64]](hfft(i16_2d))
assert_subtype[onp.Array2D[np.float32]](hfft(f32_2d))
assert_subtype[onp.Array2D[np.float64]](hfft(f64_2d))
assert_subtype[onp.Array2D[np.longdouble]](hfft(f80_2d))
assert_subtype[onp.Array2D[np.float32]](hfft(c64_2d))
assert_subtype[onp.Array2D[np.float64]](hfft(c128_2d))
assert_subtype[onp.Array2D[np.longdouble]](hfft(c160_2d))

# ihfft (same as rfft)
assert_subtype[onp.Array1D[np.complex128]](ihfft(int_1d))
assert_subtype[onp.Array1D[np.complex128]](ihfft(float_1d))
ihfft(complex_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array1D[np.complex128]](ihfft(i16_1d))
assert_subtype[onp.Array1D[np.complex64]](ihfft(f32_1d))
assert_subtype[onp.Array1D[np.complex128]](ihfft(f64_1d))
assert_subtype[onp.Array1D[np.clongdouble]](ihfft(f80_1d))
ihfft(c64_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfft(c128_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfft(c160_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array2D[np.complex128]](ihfft(int_2d))
assert_subtype[onp.Array2D[np.complex128]](ihfft(float_2d))
ihfft(complex_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array2D[np.complex128]](ihfft(i16_2d))
assert_subtype[onp.Array2D[np.complex64]](ihfft(f32_2d))
assert_subtype[onp.Array2D[np.complex128]](ihfft(f64_2d))
assert_subtype[onp.Array2D[np.clongdouble]](ihfft(f80_2d))
ihfft(c64_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfft(c128_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfft(c160_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]

# fft2 (same as ifft2)
assert_subtype[onp.Array2D[np.complex128]](fft2(int_2d))
assert_subtype[onp.Array2D[np.complex128]](fft2(float_2d))
assert_subtype[onp.Array2D[np.complex128]](fft2(complex_2d))
assert_subtype[onp.Array2D[np.complex128]](fft2(i16_2d))
assert_subtype[onp.Array2D[np.complex64]](fft2(f32_2d))
assert_subtype[onp.Array2D[np.complex128]](fft2(f64_2d))
assert_subtype[onp.Array2D[np.clongdouble]](fft2(f80_2d))
assert_subtype[onp.Array2D[np.complex64]](fft2(c64_2d))
assert_subtype[onp.Array2D[np.complex128]](fft2(c128_2d))
assert_subtype[onp.Array2D[np.clongdouble]](fft2(c160_2d))

# ifft2 (same as fft2)
assert_subtype[onp.Array2D[np.complex128]](ifft2(int_2d))
assert_subtype[onp.Array2D[np.complex128]](ifft2(float_2d))
assert_subtype[onp.Array2D[np.complex128]](ifft2(complex_2d))
assert_subtype[onp.Array2D[np.complex128]](ifft2(i16_2d))
assert_subtype[onp.Array2D[np.complex64]](ifft2(f32_2d))
assert_subtype[onp.Array2D[np.complex128]](ifft2(f64_2d))
assert_subtype[onp.Array2D[np.clongdouble]](ifft2(f80_2d))
assert_subtype[onp.Array2D[np.complex64]](ifft2(c64_2d))
assert_subtype[onp.Array2D[np.complex128]](ifft2(c128_2d))
assert_subtype[onp.Array2D[np.clongdouble]](ifft2(c160_2d))

# rfft2 (same as ihfft2)
assert_subtype[onp.Array2D[np.complex128]](rfft2(int_2d))
assert_subtype[onp.Array2D[np.complex128]](rfft2(float_2d))
rfft2(complex_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array2D[np.complex128]](rfft2(i16_2d))
assert_subtype[onp.Array2D[np.complex64]](rfft2(f32_2d))
assert_subtype[onp.Array2D[np.complex128]](rfft2(f64_2d))
assert_subtype[onp.Array2D[np.clongdouble]](rfft2(f80_2d))
rfft2(c64_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfft2(c128_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfft2(c160_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]

# irfft2 (same as hfft2)
assert_subtype[onp.Array2D[np.float64]](irfft2(int_2d))
assert_subtype[onp.Array2D[np.float64]](irfft2(float_2d))
assert_subtype[onp.Array2D[np.float64]](irfft2(complex_2d))
assert_subtype[onp.Array2D[np.float64]](irfft2(i16_2d))
assert_subtype[onp.Array2D[np.float32]](irfft2(f32_2d))
assert_subtype[onp.Array2D[np.float64]](irfft2(f64_2d))
assert_subtype[onp.Array2D[np.longdouble]](irfft2(f80_2d))
assert_subtype[onp.Array2D[np.float32]](irfft2(c64_2d))
assert_subtype[onp.Array2D[np.float64]](irfft2(c128_2d))
assert_subtype[onp.Array2D[np.longdouble]](irfft2(c160_2d))

# hfft2 (same as irfft2)
assert_subtype[onp.Array2D[np.float64]](hfft2(int_2d))
assert_subtype[onp.Array2D[np.float64]](hfft2(float_2d))
assert_subtype[onp.Array2D[np.float64]](hfft2(complex_2d))
assert_subtype[onp.Array2D[np.float64]](hfft2(i16_2d))
assert_subtype[onp.Array2D[np.float32]](hfft2(f32_2d))
assert_subtype[onp.Array2D[np.float64]](hfft2(f64_2d))
assert_subtype[onp.Array2D[np.longdouble]](hfft2(f80_2d))
assert_subtype[onp.Array2D[np.float32]](hfft2(c64_2d))
assert_subtype[onp.Array2D[np.float64]](hfft2(c128_2d))
assert_subtype[onp.Array2D[np.longdouble]](hfft2(c160_2d))

# ihfft2 (same as rfft2)
assert_subtype[onp.Array2D[np.complex128]](ihfft2(int_2d))
assert_subtype[onp.Array2D[np.complex128]](ihfft2(float_2d))
ihfft2(complex_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array2D[np.complex128]](ihfft2(i16_2d))
assert_subtype[onp.Array2D[np.complex64]](ihfft2(f32_2d))
assert_subtype[onp.Array2D[np.complex128]](ihfft2(f64_2d))
assert_subtype[onp.Array2D[np.clongdouble]](ihfft2(f80_2d))
ihfft2(c64_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfft2(c128_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfft2(c160_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]

# fftn (same as ifftn)
assert_subtype[onp.Array1D[np.complex128]](fftn(int_1d))
assert_subtype[onp.Array1D[np.complex128]](fftn(float_1d))
assert_subtype[onp.Array1D[np.complex128]](fftn(complex_1d))
assert_subtype[onp.Array1D[np.complex128]](fftn(i16_1d))
assert_subtype[onp.Array1D[np.complex64]](fftn(f32_1d))
assert_subtype[onp.Array1D[np.complex128]](fftn(f64_1d))
assert_subtype[onp.Array1D[np.clongdouble]](fftn(f80_1d))
assert_subtype[onp.Array1D[np.complex64]](fftn(c64_1d))
assert_subtype[onp.Array1D[np.complex128]](fftn(c128_1d))
assert_subtype[onp.Array1D[np.clongdouble]](fftn(c160_1d))
assert_subtype[onp.Array2D[np.complex128]](fftn(int_2d))
assert_subtype[onp.Array2D[np.complex128]](fftn(float_2d))
assert_subtype[onp.Array2D[np.complex128]](fftn(complex_2d))
assert_subtype[onp.Array2D[np.complex128]](fftn(i16_2d))
assert_subtype[onp.Array2D[np.complex64]](fftn(f32_2d))
assert_subtype[onp.Array2D[np.complex128]](fftn(f64_2d))
assert_subtype[onp.Array2D[np.clongdouble]](fftn(f80_2d))
assert_subtype[onp.Array2D[np.complex64]](fftn(c64_2d))
assert_subtype[onp.Array2D[np.complex128]](fftn(c128_2d))
assert_subtype[onp.Array2D[np.clongdouble]](fftn(c160_2d))

# ifftn (same as fftn)
assert_subtype[onp.Array1D[np.complex128]](ifftn(int_1d))
assert_subtype[onp.Array1D[np.complex128]](ifftn(float_1d))
assert_subtype[onp.Array1D[np.complex128]](ifftn(complex_1d))
assert_subtype[onp.Array1D[np.complex128]](ifftn(i16_1d))
assert_subtype[onp.Array1D[np.complex64]](ifftn(f32_1d))
assert_subtype[onp.Array1D[np.complex128]](ifftn(f64_1d))
assert_subtype[onp.Array1D[np.clongdouble]](ifftn(f80_1d))
assert_subtype[onp.Array1D[np.complex64]](ifftn(c64_1d))
assert_subtype[onp.Array1D[np.complex128]](ifftn(c128_1d))
assert_subtype[onp.Array1D[np.clongdouble]](ifftn(c160_1d))
assert_subtype[onp.Array2D[np.complex128]](ifftn(int_2d))
assert_subtype[onp.Array2D[np.complex128]](ifftn(float_2d))
assert_subtype[onp.Array2D[np.complex128]](ifftn(complex_2d))
assert_subtype[onp.Array2D[np.complex128]](ifftn(i16_2d))
assert_subtype[onp.Array2D[np.complex64]](ifftn(f32_2d))
assert_subtype[onp.Array2D[np.complex128]](ifftn(f64_2d))
assert_subtype[onp.Array2D[np.clongdouble]](ifftn(f80_2d))
assert_subtype[onp.Array2D[np.complex64]](ifftn(c64_2d))
assert_subtype[onp.Array2D[np.complex128]](ifftn(c128_2d))
assert_subtype[onp.Array2D[np.clongdouble]](ifftn(c160_2d))

# rfftn (same as ihfftn)
assert_subtype[onp.Array1D[np.complex128]](rfftn(int_1d))
assert_subtype[onp.Array1D[np.complex128]](rfftn(float_1d))
rfftn(complex_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array1D[np.complex128]](rfftn(i16_1d))
assert_subtype[onp.Array1D[np.complex64]](rfftn(f32_1d))
assert_subtype[onp.Array1D[np.complex128]](rfftn(f64_1d))
assert_subtype[onp.Array1D[np.clongdouble]](rfftn(f80_1d))
rfftn(c64_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfftn(c128_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfftn(c160_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array2D[np.complex128]](rfftn(int_2d))
assert_subtype[onp.Array2D[np.complex128]](rfftn(float_2d))
rfftn(complex_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array2D[np.complex128]](rfftn(i16_2d))
assert_subtype[onp.Array2D[np.complex64]](rfftn(f32_2d))
assert_subtype[onp.Array2D[np.complex128]](rfftn(f64_2d))
assert_subtype[onp.Array2D[np.clongdouble]](rfftn(f80_2d))
rfftn(c64_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfftn(c128_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfftn(c160_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]

# irfftn (same as hfftn)
assert_subtype[onp.Array1D[np.float64]](irfftn(int_1d))
assert_subtype[onp.Array1D[np.float64]](irfftn(float_1d))
assert_subtype[onp.Array1D[np.float64]](irfftn(complex_1d))
assert_subtype[onp.Array1D[np.float64]](irfftn(i16_1d))
assert_subtype[onp.Array1D[np.float32]](irfftn(f32_1d))
assert_subtype[onp.Array1D[np.float64]](irfftn(f64_1d))
assert_subtype[onp.Array1D[np.longdouble]](irfftn(f80_1d))
assert_subtype[onp.Array1D[np.float32]](irfftn(c64_1d))
assert_subtype[onp.Array1D[np.float64]](irfftn(c128_1d))
assert_subtype[onp.Array1D[np.longdouble]](irfftn(c160_1d))
assert_subtype[onp.Array2D[np.float64]](irfftn(int_2d))
assert_subtype[onp.Array2D[np.float64]](irfftn(float_2d))
assert_subtype[onp.Array2D[np.float64]](irfftn(complex_2d))
assert_subtype[onp.Array2D[np.float64]](irfftn(i16_2d))
assert_subtype[onp.Array2D[np.float32]](irfftn(f32_2d))
assert_subtype[onp.Array2D[np.float64]](irfftn(f64_2d))
assert_subtype[onp.Array2D[np.longdouble]](irfftn(f80_2d))
assert_subtype[onp.Array2D[np.float32]](irfftn(c64_2d))
assert_subtype[onp.Array2D[np.float64]](irfftn(c128_2d))
assert_subtype[onp.Array2D[np.longdouble]](irfftn(c160_2d))

# hfftn (same as irfftn)
assert_subtype[onp.Array1D[np.float64]](hfftn(int_1d))
assert_subtype[onp.Array1D[np.float64]](hfftn(float_1d))
assert_subtype[onp.Array1D[np.float64]](hfftn(complex_1d))
assert_subtype[onp.Array1D[np.float64]](hfftn(i16_1d))
assert_subtype[onp.Array1D[np.float32]](hfftn(f32_1d))
assert_subtype[onp.Array1D[np.float64]](hfftn(f64_1d))
assert_subtype[onp.Array1D[np.longdouble]](hfftn(f80_1d))
assert_subtype[onp.Array1D[np.float32]](hfftn(c64_1d))
assert_subtype[onp.Array1D[np.float64]](hfftn(c128_1d))
assert_subtype[onp.Array1D[np.longdouble]](hfftn(c160_1d))
assert_subtype[onp.Array2D[np.float64]](hfftn(int_2d))
assert_subtype[onp.Array2D[np.float64]](hfftn(float_2d))
assert_subtype[onp.Array2D[np.float64]](hfftn(complex_2d))
assert_subtype[onp.Array2D[np.float64]](hfftn(i16_2d))
assert_subtype[onp.Array2D[np.float32]](hfftn(f32_2d))
assert_subtype[onp.Array2D[np.float64]](hfftn(f64_2d))
assert_subtype[onp.Array2D[np.longdouble]](hfftn(f80_2d))
assert_subtype[onp.Array2D[np.float32]](hfftn(c64_2d))
assert_subtype[onp.Array2D[np.float64]](hfftn(c128_2d))
assert_subtype[onp.Array2D[np.longdouble]](hfftn(c160_2d))

# ihfftn (same as rfftn)
assert_subtype[onp.Array1D[np.complex128]](ihfftn(int_1d))
assert_subtype[onp.Array1D[np.complex128]](ihfftn(float_1d))
ihfftn(complex_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array1D[np.complex128]](ihfftn(i16_1d))
assert_subtype[onp.Array1D[np.complex64]](ihfftn(f32_1d))
assert_subtype[onp.Array1D[np.complex128]](ihfftn(f64_1d))
assert_subtype[onp.Array1D[np.clongdouble]](ihfftn(f80_1d))
ihfftn(c64_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfftn(c128_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfftn(c160_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array2D[np.complex128]](ihfftn(int_2d))
assert_subtype[onp.Array2D[np.complex128]](ihfftn(float_2d))
ihfftn(complex_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
assert_subtype[onp.Array2D[np.complex128]](ihfftn(i16_2d))
assert_subtype[onp.Array2D[np.complex64]](ihfftn(f32_2d))
assert_subtype[onp.Array2D[np.complex128]](ihfftn(f64_2d))
assert_subtype[onp.Array2D[np.clongdouble]](ihfftn(f80_2d))
ihfftn(c64_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfftn(c128_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfftn(c160_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
