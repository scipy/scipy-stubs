import numpy as np
import optype.numpy as onp

from scipy.fft import fft, fft2, hfft, hfft2, ifft, ifft2, ihfft, irfft, irfft2, rfft, rfft2

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
