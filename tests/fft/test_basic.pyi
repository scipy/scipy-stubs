import numpy as np
import optype.numpy as onp

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

# NOTE: `typing.assert_type` will work on numpy 2.1+ because of differences in shape-typing,
# so we instead use these helper functions to assert the expected types.

def _assert_1d_f32(x: onp.Array1D[np.float32], /) -> None: ...
def _assert_1d_f64(x: onp.Array1D[np.float64], /) -> None: ...
def _assert_1d_f80(x: onp.Array1D[np.longdouble], /) -> None: ...
def _assert_1d_c64(x: onp.Array1D[np.complex64], /) -> None: ...
def _assert_1d_c128(x: onp.Array1D[np.complex128], /) -> None: ...
def _assert_1d_c160(x: onp.Array1D[np.clongdouble], /) -> None: ...
def _assert_2d_c64(x: onp.Array2D[np.complex64], /) -> None: ...
def _assert_2d_c128(x: onp.Array2D[np.complex128], /) -> None: ...
def _assert_2d_c160(x: onp.Array2D[np.clongdouble], /) -> None: ...
def _assert_2d_f32(x: onp.Array2D[np.float32], /) -> None: ...
def _assert_2d_f64(x: onp.Array2D[np.float64], /) -> None: ...
def _assert_2d_f80(x: onp.Array2D[np.longdouble], /) -> None: ...

###

# fft (same as ifft)
_assert_1d_c128(fft(int_1d))
_assert_1d_c128(fft(float_1d))
_assert_1d_c128(fft(complex_1d))
_assert_1d_c128(fft(i16_1d))
_assert_1d_c64(fft(f32_1d))
_assert_1d_c128(fft(f64_1d))
_assert_1d_c160(fft(f80_1d))
_assert_1d_c64(fft(c64_1d))
_assert_1d_c128(fft(c128_1d))
_assert_1d_c160(fft(c160_1d))
_assert_2d_c128(fft(int_2d))
_assert_2d_c128(fft(float_2d))
_assert_2d_c128(fft(complex_2d))
_assert_2d_c128(fft(i16_2d))
_assert_2d_c64(fft(f32_2d))
_assert_2d_c128(fft(f64_2d))
_assert_2d_c160(fft(f80_2d))
_assert_2d_c64(fft(c64_2d))
_assert_2d_c128(fft(c128_2d))
_assert_2d_c160(fft(c160_2d))

# ifft (same as fft)
_assert_1d_c128(ifft(int_1d))
_assert_1d_c128(ifft(float_1d))
_assert_1d_c128(ifft(complex_1d))
_assert_1d_c128(ifft(i16_1d))
_assert_1d_c64(ifft(f32_1d))
_assert_1d_c128(ifft(f64_1d))
_assert_1d_c160(ifft(f80_1d))
_assert_1d_c64(ifft(c64_1d))
_assert_1d_c128(ifft(c128_1d))
_assert_1d_c160(ifft(c160_1d))
_assert_2d_c128(ifft(int_2d))
_assert_2d_c128(ifft(float_2d))
_assert_2d_c128(ifft(complex_2d))
_assert_2d_c128(ifft(i16_2d))
_assert_2d_c64(ifft(f32_2d))
_assert_2d_c128(ifft(f64_2d))
_assert_2d_c160(ifft(f80_2d))
_assert_2d_c64(ifft(c64_2d))
_assert_2d_c128(ifft(c128_2d))
_assert_2d_c160(ifft(c160_2d))

# rfft (same as ihfft)
_assert_1d_c128(rfft(int_1d))
_assert_1d_c128(rfft(float_1d))
rfft(complex_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
_assert_1d_c128(rfft(i16_1d))
_assert_1d_c64(rfft(f32_1d))
_assert_1d_c128(rfft(f64_1d))
_assert_1d_c160(rfft(f80_1d))
rfft(c64_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfft(c128_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfft(c160_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
_assert_2d_c128(rfft(int_2d))
_assert_2d_c128(rfft(float_2d))
rfft(complex_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
_assert_2d_c128(rfft(i16_2d))
_assert_2d_c64(rfft(f32_2d))
_assert_2d_c128(rfft(f64_2d))
_assert_2d_c160(rfft(f80_2d))
rfft(c64_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfft(c128_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfft(c160_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]

# irfft (same as hfft)
_assert_1d_f64(irfft(int_1d))
_assert_1d_f64(irfft(float_1d))
_assert_1d_f64(irfft(complex_1d))
_assert_1d_f64(irfft(i16_1d))
_assert_1d_f32(irfft(f32_1d))
_assert_1d_f64(irfft(f64_1d))
_assert_1d_f80(irfft(f80_1d))
_assert_1d_f32(irfft(c64_1d))
_assert_1d_f64(irfft(c128_1d))
_assert_1d_f80(irfft(c160_1d))
_assert_2d_f64(irfft(int_2d))
_assert_2d_f64(irfft(float_2d))
_assert_2d_f64(irfft(complex_2d))
_assert_2d_f64(irfft(i16_2d))
_assert_2d_f32(irfft(f32_2d))
_assert_2d_f64(irfft(f64_2d))
_assert_2d_f80(irfft(f80_2d))
_assert_2d_f32(irfft(c64_2d))
_assert_2d_f64(irfft(c128_2d))
_assert_2d_f80(irfft(c160_2d))

# hfft (same as irfft)
_assert_1d_f64(hfft(int_1d))
_assert_1d_f64(hfft(float_1d))
_assert_1d_f64(hfft(complex_1d))
_assert_1d_f64(hfft(i16_1d))
_assert_1d_f32(hfft(f32_1d))
_assert_1d_f64(hfft(f64_1d))
_assert_1d_f80(hfft(f80_1d))
_assert_1d_f32(hfft(c64_1d))
_assert_1d_f64(hfft(c128_1d))
_assert_1d_f80(hfft(c160_1d))
_assert_2d_f64(hfft(int_2d))
_assert_2d_f64(hfft(float_2d))
_assert_2d_f64(hfft(complex_2d))
_assert_2d_f64(hfft(i16_2d))
_assert_2d_f32(hfft(f32_2d))
_assert_2d_f64(hfft(f64_2d))
_assert_2d_f80(hfft(f80_2d))
_assert_2d_f32(hfft(c64_2d))
_assert_2d_f64(hfft(c128_2d))
_assert_2d_f80(hfft(c160_2d))

# ihfft (same as rfft)
_assert_1d_c128(ihfft(int_1d))
_assert_1d_c128(ihfft(float_1d))
ihfft(complex_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
_assert_1d_c128(ihfft(i16_1d))
_assert_1d_c64(ihfft(f32_1d))
_assert_1d_c128(ihfft(f64_1d))
_assert_1d_c160(ihfft(f80_1d))
ihfft(c64_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfft(c128_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfft(c160_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
_assert_2d_c128(ihfft(int_2d))
_assert_2d_c128(ihfft(float_2d))
ihfft(complex_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
_assert_2d_c128(ihfft(i16_2d))
_assert_2d_c64(ihfft(f32_2d))
_assert_2d_c128(ihfft(f64_2d))
_assert_2d_c160(ihfft(f80_2d))
ihfft(c64_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfft(c128_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfft(c160_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]

# fft2 (same as ifft2)
_assert_2d_c128(fft2(int_2d))
_assert_2d_c128(fft2(float_2d))
_assert_2d_c128(fft2(complex_2d))
_assert_2d_c128(fft2(i16_2d))
_assert_2d_c64(fft2(f32_2d))
_assert_2d_c128(fft2(f64_2d))
_assert_2d_c160(fft2(f80_2d))
_assert_2d_c64(fft2(c64_2d))
_assert_2d_c128(fft2(c128_2d))
_assert_2d_c160(fft2(c160_2d))

# ifft2 (same as fft2)
_assert_2d_c128(ifft2(int_2d))
_assert_2d_c128(ifft2(float_2d))
_assert_2d_c128(ifft2(complex_2d))
_assert_2d_c128(ifft2(i16_2d))
_assert_2d_c64(ifft2(f32_2d))
_assert_2d_c128(ifft2(f64_2d))
_assert_2d_c160(ifft2(f80_2d))
_assert_2d_c64(ifft2(c64_2d))
_assert_2d_c128(ifft2(c128_2d))
_assert_2d_c160(ifft2(c160_2d))

# rfft2 (same as ihfft2)
_assert_2d_c128(rfft2(int_2d))
_assert_2d_c128(rfft2(float_2d))
rfft2(complex_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
_assert_2d_c128(rfft2(i16_2d))
_assert_2d_c64(rfft2(f32_2d))
_assert_2d_c128(rfft2(f64_2d))
_assert_2d_c160(rfft2(f80_2d))
rfft2(c64_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfft2(c128_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfft2(c160_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]

# irfft2 (same as hfft2)
_assert_2d_f64(irfft2(int_2d))
_assert_2d_f64(irfft2(float_2d))
_assert_2d_f64(irfft2(complex_2d))
_assert_2d_f64(irfft2(i16_2d))
_assert_2d_f32(irfft2(f32_2d))
_assert_2d_f64(irfft2(f64_2d))
_assert_2d_f80(irfft2(f80_2d))
_assert_2d_f32(irfft2(c64_2d))
_assert_2d_f64(irfft2(c128_2d))
_assert_2d_f80(irfft2(c160_2d))

# hfft2 (same as irfft2)
_assert_2d_f64(hfft2(int_2d))
_assert_2d_f64(hfft2(float_2d))
_assert_2d_f64(hfft2(complex_2d))
_assert_2d_f64(hfft2(i16_2d))
_assert_2d_f32(hfft2(f32_2d))
_assert_2d_f64(hfft2(f64_2d))
_assert_2d_f80(hfft2(f80_2d))
_assert_2d_f32(hfft2(c64_2d))
_assert_2d_f64(hfft2(c128_2d))
_assert_2d_f80(hfft2(c160_2d))

# ihfft2 (same as rfft2)
_assert_2d_c128(ihfft2(int_2d))
_assert_2d_c128(ihfft2(float_2d))
ihfft2(complex_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
_assert_2d_c128(ihfft2(i16_2d))
_assert_2d_c64(ihfft2(f32_2d))
_assert_2d_c128(ihfft2(f64_2d))
_assert_2d_c160(ihfft2(f80_2d))
ihfft2(c64_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfft2(c128_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
ihfft2(c160_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]

# fftn (same as ifftn)
_assert_1d_c128(fftn(int_1d))
_assert_1d_c128(fftn(float_1d))
_assert_1d_c128(fftn(complex_1d))
_assert_1d_c128(fftn(i16_1d))
_assert_1d_c64(fftn(f32_1d))
_assert_1d_c128(fftn(f64_1d))
_assert_1d_c160(fftn(f80_1d))
_assert_1d_c64(fftn(c64_1d))
_assert_1d_c128(fftn(c128_1d))
_assert_1d_c160(fftn(c160_1d))
_assert_2d_c128(fftn(int_2d))
_assert_2d_c128(fftn(float_2d))
_assert_2d_c128(fftn(complex_2d))
_assert_2d_c128(fftn(i16_2d))
_assert_2d_c64(fftn(f32_2d))
_assert_2d_c128(fftn(f64_2d))
_assert_2d_c160(fftn(f80_2d))
_assert_2d_c64(fftn(c64_2d))
_assert_2d_c128(fftn(c128_2d))
_assert_2d_c160(fftn(c160_2d))

# ifftn (same as fftn)
_assert_1d_c128(ifftn(int_1d))
_assert_1d_c128(ifftn(float_1d))
_assert_1d_c128(ifftn(complex_1d))
_assert_1d_c128(ifftn(i16_1d))
_assert_1d_c64(ifftn(f32_1d))
_assert_1d_c128(ifftn(f64_1d))
_assert_1d_c160(ifftn(f80_1d))
_assert_1d_c64(ifftn(c64_1d))
_assert_1d_c128(ifftn(c128_1d))
_assert_1d_c160(ifftn(c160_1d))
_assert_2d_c128(ifftn(int_2d))
_assert_2d_c128(ifftn(float_2d))
_assert_2d_c128(ifftn(complex_2d))
_assert_2d_c128(ifftn(i16_2d))
_assert_2d_c64(ifftn(f32_2d))
_assert_2d_c128(ifftn(f64_2d))
_assert_2d_c160(ifftn(f80_2d))
_assert_2d_c64(ifftn(c64_2d))
_assert_2d_c128(ifftn(c128_2d))
_assert_2d_c160(ifftn(c160_2d))

# rfftn (same as ihfftn)
_assert_1d_c128(rfftn(int_1d))
_assert_1d_c128(rfftn(float_1d))
rfftn(complex_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
_assert_1d_c128(rfftn(i16_1d))
_assert_1d_c64(rfftn(f32_1d))
_assert_1d_c128(rfftn(f64_1d))
_assert_1d_c160(rfftn(f80_1d))
rfftn(c64_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfftn(c128_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfftn(c160_1d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
_assert_2d_c128(rfftn(int_2d))
_assert_2d_c128(rfftn(float_2d))
rfftn(complex_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
_assert_2d_c128(rfftn(i16_2d))
_assert_2d_c64(rfftn(f32_2d))
_assert_2d_c128(rfftn(f64_2d))
_assert_2d_c160(rfftn(f80_2d))
rfftn(c64_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfftn(c128_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]
rfftn(c160_2d)  # type:ignore[arg-type] # pyright:ignore[reportArgumentType, reportCallIssue] # pyrefly:ignore[no-matching-overload]

# irfftn (same as hfftn)
_assert_1d_f64(irfftn(int_1d))
_assert_1d_f64(irfftn(float_1d))
_assert_1d_f64(irfftn(complex_1d))
_assert_1d_f64(irfftn(i16_1d))
_assert_1d_f32(irfftn(f32_1d))
_assert_1d_f64(irfftn(f64_1d))
_assert_1d_f80(irfftn(f80_1d))
_assert_1d_f32(irfftn(c64_1d))
_assert_1d_f64(irfftn(c128_1d))
_assert_1d_f80(irfftn(c160_1d))
_assert_2d_f64(irfftn(int_2d))
_assert_2d_f64(irfftn(float_2d))
_assert_2d_f64(irfftn(complex_2d))
_assert_2d_f64(irfftn(i16_2d))
_assert_2d_f32(irfftn(f32_2d))
_assert_2d_f64(irfftn(f64_2d))
_assert_2d_f80(irfftn(f80_2d))
_assert_2d_f32(irfftn(c64_2d))
_assert_2d_f64(irfftn(c128_2d))
_assert_2d_f80(irfftn(c160_2d))

# hfftn (same as irfftn)
_assert_1d_f64(hfftn(int_1d))
_assert_1d_f64(hfftn(float_1d))
_assert_1d_f64(hfftn(complex_1d))
_assert_1d_f64(hfftn(i16_1d))
_assert_1d_f32(hfftn(f32_1d))
_assert_1d_f64(hfftn(f64_1d))
_assert_1d_f80(hfftn(f80_1d))
_assert_1d_f32(hfftn(c64_1d))
_assert_1d_f64(hfftn(c128_1d))
_assert_1d_f80(hfftn(c160_1d))
_assert_2d_f64(hfftn(int_2d))
_assert_2d_f64(hfftn(float_2d))
_assert_2d_f64(hfftn(complex_2d))
_assert_2d_f64(hfftn(i16_2d))
_assert_2d_f32(hfftn(f32_2d))
_assert_2d_f64(hfftn(f64_2d))
_assert_2d_f80(hfftn(f80_2d))
_assert_2d_f32(hfftn(c64_2d))
_assert_2d_f64(hfftn(c128_2d))
_assert_2d_f80(hfftn(c160_2d))
