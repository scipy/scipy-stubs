from typing import Any, SupportsIndex, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy._typing import AnyShape

__all__ = ["fft", "fft2", "fftn", "ifft", "ifft2", "ifftn", "irfft", "rfft"]

###

type _AsF64ND = onp.ToArrayND[float, np.float64 | npc.integer | np.bool]
type _AsC128ND = onp.ToArrayND[complex, np.complex128 | np.float64 | npc.integer | np.bool]

type _Axis = SupportsIndex
type _Axes = tuple[SupportsIndex, SupportsIndex]

###

# NOTE: keep in sync with `ifft`
@overload  # +c128
def fft(
    x: _AsC128ND, n: onp.ToJustInt | None = None, axis: _Axis = -1, overwrite_x: bool = False
) -> onp.ArrayND[np.complex128]: ...
@overload  # fallback
def fft(
    x: onp.ToComplexND, n: onp.ToJustInt | None = None, axis: _Axis = -1, overwrite_x: bool = False
) -> onp.ArrayND[np.complex128 | Any]: ...

# NOTE: keep in sync with `fft`
@overload  # +c128
def ifft(
    x: _AsC128ND, n: onp.ToJustInt | None = None, axis: _Axis = -1, overwrite_x: bool = False
) -> onp.ArrayND[np.complex128]: ...
@overload  # fallback
def ifft(
    x: onp.ToComplexND, n: onp.ToJustInt | None = None, axis: _Axis = -1, overwrite_x: bool = False
) -> onp.ArrayND[np.complex128 | Any]: ...

# NOTE: keep in sync with `irfft`
@overload  # +f64
def rfft(x: _AsF64ND, n: onp.ToJustInt | None = None, axis: _Axis = -1, overwrite_x: bool = False) -> onp.ArrayND[np.float64]: ...
@overload  # fallback
def rfft(
    x: onp.ToFloatND, n: onp.ToJustInt | None = None, axis: _Axis = -1, overwrite_x: bool = False
) -> onp.ArrayND[np.float64 | Any]: ...

# NOTE: keep in sync with `rfft`
@overload  # +f64
def irfft(
    x: _AsF64ND, n: onp.ToJustInt | None = None, axis: _Axis = -1, overwrite_x: bool = False
) -> onp.ArrayND[np.float64]: ...
@overload  # fallback
def irfft(
    x: onp.ToFloatND, n: onp.ToJustInt | None = None, axis: _Axis = -1, overwrite_x: bool = False
) -> onp.ArrayND[np.float64 | Any]: ...

# NOTE: keep in sync with `ifftn`
@overload  # +c128
def fftn(
    x: _AsC128ND, shape: AnyShape | None = None, axes: AnyShape | None = None, overwrite_x: bool = False
) -> onp.ArrayND[np.complex128]: ...
@overload  # fallback
def fftn(
    x: onp.ToComplexND, shape: AnyShape | None = None, axes: AnyShape | None = None, overwrite_x: bool = False
) -> onp.ArrayND[np.complex128 | Any]: ...

# NOTE: keep in sync with `fftn`
@overload  # +c128
def ifftn(
    x: _AsC128ND, shape: AnyShape | None = None, axes: AnyShape | None = None, overwrite_x: bool = False
) -> onp.ArrayND[np.complex128]: ...
@overload  # fallback
def ifftn(
    x: onp.ToComplexND, shape: AnyShape | None = None, axes: AnyShape | None = None, overwrite_x: bool = False
) -> onp.ArrayND[np.complex128 | Any]: ...

# NOTE: keep in sync with `ifft2`
@overload  # +c128
def fft2(
    x: _AsC128ND, shape: AnyShape | None = None, axes: _Axes = (-2, -1), overwrite_x: bool = False
) -> onp.ArrayND[np.complex128]: ...
@overload  # fallback
def fft2(
    x: onp.ToComplexND, shape: AnyShape | None = None, axes: _Axes = (-2, -1), overwrite_x: bool = False
) -> onp.ArrayND[np.complex128 | Any]: ...

# NOTE: keep in sync with `fft2`
@overload  # +c128
def ifft2(
    x: _AsC128ND, shape: AnyShape | None = None, axes: _Axes = (-2, -1), overwrite_x: bool = False
) -> onp.ArrayND[np.complex128]: ...
@overload  # fallback
def ifft2(
    x: onp.ToComplexND, shape: AnyShape | None = None, axes: _Axes = (-2, -1), overwrite_x: bool = False
) -> onp.ArrayND[np.complex128 | Any]: ...
