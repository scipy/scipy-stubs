from collections.abc import Callable
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import deprecated

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from .windows._windows import _ToWindow

__all__ = ["check_COLA", "check_NOLA", "coherence", "csd", "istft", "lombscargle", "periodogram", "spectrogram", "stft", "welch"]

_Float1D: TypeAlias = onp.Array1D[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float64]
_FloatingND: TypeAlias = onp.ArrayND[np.float32 | np.float64 | np.longdouble]
_CFloatingND: TypeAlias = onp.ArrayND[npc.complexfloating]

_Detrend: TypeAlias = Literal["literal", "constant", False] | Callable[[onp.ArrayND], onp.ArrayND]
_Scaling: TypeAlias = Literal["density", "spectrum"]
_LegacyScaling: TypeAlias = Literal["psd", "spectrum"]
_Average: TypeAlias = Literal["mean", "median"]
_Boundary: TypeAlias = Literal["even", "odd", "constant", "zeros"] | None
_Normalize: TypeAlias = Literal["power", "normalize", "amplitude"] | bool

###

@overload
def lombscargle(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    freqs: onp.ToFloat1D,
    *,
    precenter: op.JustObject = ...,
    normalize: _Normalize = False,
    weights: onp.ToFloat1D | None = None,
    floating_mean: bool = False,
) -> _Float1D: ...
@overload
@deprecated(
    "The `precenter` argument is deprecated and will be removed in SciPy 1.19.0. "
    "The functionality can be substituted by passing `y - y.mean()` to `y`."
)
def lombscargle(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    freqs: onp.ToFloat1D,
    *,
    precenter: bool,
    normalize: _Normalize = False,
    weights: onp.ToFloat1D | None = None,
    floating_mean: bool = False,
) -> _Float1D: ...

#
@overload  # f64
def periodogram(
    x: onp.ToIntND | onp.ToJustFloat64_ND | onp.ToJustComplex128_ND,
    fs: float = 1.0,
    window: _ToWindow | None = "boxcar",
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
) -> tuple[_Float1D, onp.ArrayND[np.float64]]: ...
@overload  # f32
def periodogram(
    x: onp.ToJustFloat16_ND | onp.ToJustFloat32_ND | onp.ToJustComplex64_ND,
    fs: float = 1.0,
    window: _ToWindow | None = "boxcar",
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
) -> tuple[_Float1D, onp.ArrayND[np.float32]]: ...
@overload  # f80
def periodogram(
    x: onp.ToJustLongDoubleND | onp.ToJustCLongDoubleND,
    fs: float = 1.0,
    window: _ToWindow | None = "boxcar",
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
) -> tuple[_Float1D, onp.ArrayND[np.longdouble]]: ...

#
@overload  # f64
def welch(
    x: onp.ToIntND | onp.ToJustFloat64_ND | onp.ToJustComplex128_ND,
    fs: float = 1.0,
    window: _ToWindow = "hann_periodic",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    average: _Average = "mean",
) -> tuple[_Float1D, onp.ArrayND[np.float64]]: ...
@overload  # f64
def welch(
    x: onp.ToJustFloat16_ND | onp.ToJustFloat32_ND | onp.ToJustComplex64_ND,
    fs: float = 1.0,
    window: _ToWindow = "hann_periodic",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    average: _Average = "mean",
) -> tuple[_Float1D, onp.ArrayND[np.float32]]: ...
@overload  # f64
def welch(
    x: onp.ToJustLongDoubleND | onp.ToJustCLongDoubleND,
    fs: float = 1.0,
    window: _ToWindow = "hann_periodic",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    average: _Average = "mean",
) -> tuple[_Float1D, onp.ArrayND[np.longdouble]]: ...

# NOTE: We assume that `x is not y` always holds here.
# See https://github.com/scipy/scipy/issues/24285 for details.
@overload  # c128
def csd(
    x: onp.ToIntND | onp.ToJustFloat64_ND | onp.ToJustComplex128_ND,
    y: onp.ToIntND | onp.ToJustFloat64_ND | onp.ToJustComplex128_ND,
    fs: float = 1.0,
    window: _ToWindow = "hann_periodic",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    average: _Average = "mean",
) -> tuple[_Float1D, onp.ArrayND[np.complex128]]: ...
@overload  # c64
def csd(
    x: onp.ToJustFloat16_ND | onp.ToJustFloat32_ND | onp.ToJustComplex64_ND,
    y: onp.ToJustFloat16_ND | onp.ToJustFloat32_ND | onp.ToJustComplex64_ND,
    fs: float = 1.0,
    window: _ToWindow = "hann_periodic",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    average: _Average = "mean",
) -> tuple[_Float1D, onp.ArrayND[np.complex64]]: ...
@overload  # c160
def csd(
    x: onp.ToJustLongDoubleND | onp.ToJustCLongDoubleND,
    y: onp.ToJustLongDoubleND | onp.ToJustCLongDoubleND,
    fs: float = 1.0,
    window: _ToWindow = "hann_periodic",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    average: _Average = "mean",
) -> tuple[_Float1D, onp.ArrayND[np.clongdouble]]: ...
@overload  # fallback
def csd(
    x: onp.ToComplexND,
    y: onp.ToComplexND,
    fs: float = 1.0,
    window: _ToWindow = "hann_periodic",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    average: _Average = "mean",
) -> tuple[_Float1D, _CFloatingND]: ...

# NOTE: Even though it is theoretically possible to pass `mode` as positional argument, it's unlikely to be done in practice,
# and would significantly complicate the overloads. Thus, we only support passing `mode` as keyword argument here (if "complex").
@overload  # f64, mode != "complex"
def spectrogram(
    x: onp.ToIntND | onp.ToJustFloat64_ND | onp.ToJustComplex128_ND,
    fs: float = 1.0,
    window: _ToWindow = ("tukey_periodic", 0.25),
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    mode: Literal["psd", "magnitude", "angle", "phase"] = "psd",
) -> tuple[_Float1D, _Float1D, onp.ArrayND[np.float64]]: ...
@overload  # c128, mode == "complex"
def spectrogram(
    x: onp.ToIntND | onp.ToJustFloat64_ND | onp.ToJustComplex128_ND,
    fs: float = 1.0,
    window: _ToWindow = ("tukey_periodic", 0.25),
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    *,
    mode: Literal["complex"],
) -> tuple[_Float1D, _Float1D, onp.ArrayND[np.complex128]]: ...
@overload  # f32, mode != "complex"
def spectrogram(
    x: onp.ToJustFloat16_ND | onp.ToJustFloat32_ND | onp.ToJustComplex64_ND,
    fs: float = 1.0,
    window: _ToWindow = ("tukey_periodic", 0.25),
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    mode: Literal["psd", "magnitude", "angle", "phase"] = "psd",
) -> tuple[_Float1D, _Float1D, onp.ArrayND[np.float32]]: ...
@overload  # c64, mode == "complex"
def spectrogram(
    x: onp.ToJustFloat16_ND | onp.ToJustFloat32_ND | onp.ToJustComplex64_ND,
    fs: float = 1.0,
    window: _ToWindow = ("tukey_periodic", 0.25),
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    *,
    mode: Literal["complex"],
) -> tuple[_Float1D, _Float1D, onp.ArrayND[np.complex64]]: ...
@overload  # f80, mode != "complex"
def spectrogram(
    x: onp.ToJustLongDoubleND | onp.ToJustCLongDoubleND,
    fs: float = 1.0,
    window: _ToWindow = ("tukey_periodic", 0.25),
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    mode: Literal["psd", "magnitude", "angle", "phase"] = "psd",
) -> tuple[_Float1D, _Float1D, onp.ArrayND[np.longdouble]]: ...
@overload  # c80, mode == "complex"
def spectrogram(
    x: onp.ToJustLongDoubleND | onp.ToJustCLongDoubleND,
    fs: float = 1.0,
    window: _ToWindow = ("tukey_periodic", 0.25),
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    *,
    mode: Literal["complex"],
) -> tuple[_Float1D, _Float1D, onp.ArrayND[np.clongdouble]]: ...
@overload  # fallback
def spectrogram(
    x: onp.ToComplexND,
    fs: float = 1.0,
    window: _ToWindow = ("tukey_periodic", 0.25),
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: _Detrend = "constant",
    return_onesided: bool = True,
    scaling: _Scaling = "density",
    axis: int = -1,
    mode: str = "psd",
) -> tuple[_Float1D, _Float1D, onp.ArrayND[Any]]: ...

#
def check_COLA(window: _ToWindow, nperseg: onp.ToInt, noverlap: onp.ToInt, tol: onp.ToFloat = 1e-10) -> np.bool_: ...
def check_NOLA(window: _ToWindow, nperseg: onp.ToInt, noverlap: onp.ToInt, tol: onp.ToFloat = 1e-10) -> np.bool_: ...

#
def stft(
    x: onp.ToComplexND,
    fs: onp.ToFloat = 1.0,
    window: _ToWindow = "hann_periodic",
    nperseg: onp.ToInt = 256,
    noverlap: onp.ToInt | None = None,
    nfft: onp.ToInt | None = None,
    detrend: _Detrend = False,
    return_onesided: op.CanBool = True,
    boundary: _Boundary = "zeros",
    padded: op.CanBool = True,
    axis: op.CanIndex = -1,
    scaling: _LegacyScaling = "spectrum",
) -> tuple[_FloatND, _FloatND, _CFloatingND]: ...

#
@overload  # input_onesided is `True`
def istft(
    Zxx: onp.ToComplexND,
    fs: onp.ToFloat = 1.0,
    window: _ToWindow = "hann_periodic",
    nperseg: onp.ToInt | None = None,
    noverlap: onp.ToInt | None = None,
    nfft: onp.ToInt | None = None,
    input_onesided: onp.ToTrue = True,
    boundary: op.CanBool = True,
    time_axis: op.CanIndex = -1,
    freq_axis: op.CanIndex = -2,
    scaling: _LegacyScaling = "spectrum",
) -> tuple[_FloatND, _FloatingND]: ...
@overload  # input_onesided is `False` (positional)
def istft(
    Zxx: onp.ToComplexND,
    fs: onp.ToFloat,
    window: _ToWindow,
    nperseg: onp.ToInt | None,
    noverlap: onp.ToInt | None,
    nfft: onp.ToInt | None,
    input_onesided: onp.ToFalse,
    boundary: op.CanBool = True,
    time_axis: op.CanIndex = -1,
    freq_axis: op.CanIndex = -2,
    scaling: _LegacyScaling = "spectrum",
) -> tuple[_FloatND, _CFloatingND]: ...
@overload  # input_onesided is `False` (keyword)
def istft(
    Zxx: onp.ToComplexND,
    fs: onp.ToFloat = 1.0,
    window: _ToWindow = "hann_periodic",
    nperseg: onp.ToInt | None = None,
    noverlap: onp.ToInt | None = None,
    nfft: onp.ToInt | None = None,
    *,
    input_onesided: onp.ToFalse,
    boundary: op.CanBool = True,
    time_axis: op.CanIndex = -1,
    freq_axis: op.CanIndex = -2,
    scaling: _LegacyScaling = "spectrum",
) -> tuple[_FloatND, _CFloatingND]: ...

#
def coherence(
    x: onp.ToComplexND,
    y: onp.ToComplexND,
    fs: onp.ToFloat = 1.0,
    window: _ToWindow = "hann_periodic",
    nperseg: onp.ToInt | None = None,
    noverlap: onp.ToInt | None = None,
    nfft: onp.ToInt | None = None,
    detrend: _Detrend = "constant",
    axis: op.CanIndex = -1,
) -> tuple[_FloatND, _FloatingND]: ...
