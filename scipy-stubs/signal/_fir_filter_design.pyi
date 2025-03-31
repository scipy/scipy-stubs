from typing import Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc
from .windows._windows import _ToWindow

__all__ = ["firls", "firwin", "firwin2", "kaiser_atten", "kaiser_beta", "kaiserord", "minimum_phase", "remez"]

###

_InexactT_py = TypeVar("_InexactT_py", bound=complex | float | int | bool)
_InexactT_np = TypeVar("_InexactT_np", bound=npc.inexact)

_IIRFilterType: TypeAlias = Literal["bandpass", "lowpass", "highpass", "bandstop"]
_RemezFilterType: TypeAlias = Literal["bandpass", "differentiator", "hilbert"]
_LinearPhaseFIRMethod: TypeAlias = Literal["homomorphic", "hilbert"]

###

#
def kaiser_beta(a: onp.ToFloat) -> float | int | bool: ...
def kaiserord(ripple: onp.ToFloat, width: onp.ToFloat) -> tuple[int | bool, float | int | bool]: ...

#
@overload
def kaiser_atten(numtaps: _InexactT_py, width: _InexactT_py) -> _InexactT_py: ...
@overload
def kaiser_atten(numtaps: _InexactT_np | float | int | bool, width: _InexactT_np) -> _InexactT_np: ...
@overload
def kaiser_atten(numtaps: npc.integer, width: npc.integer | float | int | bool) -> np.float64: ...
@overload
def kaiser_atten(numtaps: npc.integer | float | int | bool, width: npc.integer) -> np.float64: ...

#
@overload
def firwin(
    numtaps: onp.ToJustInt,
    cutoff: onp.ToFloat64 | onp.ToFloat64_1D,
    *,
    width: onp.ToFloat64 | None = None,
    window: _ToWindow = "hamming",
    pass_zero: _IIRFilterType | bool = True,
    scale: op.CanBool = True,
    fs: onp.ToFloat64 | None = None,
) -> onp.Array1D[np.float64]: ...
@overload
def firwin(
    numtaps: onp.ToJustInt,
    cutoff: onp.ToFloat | onp.ToFloat1D,
    *,
    width: onp.ToFloat | None = None,
    window: _ToWindow = "hamming",
    pass_zero: _IIRFilterType | bool = True,
    scale: op.CanBool = True,
    fs: onp.ToFloat | None = None,
) -> onp.Array1D[np.float64 | np.longdouble]: ...

#
def firwin2(
    numtaps: onp.ToJustInt,
    freq: onp.ToFloat1D,
    gain: onp.ToFloat1D,
    *,
    nfreqs: onp.ToJustInt | None = None,
    window: _ToWindow = "hamming",
    antisymmetric: op.CanBool = False,
    fs: onp.ToFloat | None = None,
) -> onp.Array1D[np.float64]: ...

#
@overload
def firls(
    numtaps: onp.ToJustInt,
    bands: onp.ToFloat1D,
    desired: onp.ToFloat1D,
    *,
    weight: onp.ToFloat1D | None = None,
    fs: onp.ToFloat | None = None,
) -> onp.Array1D[np.float64]: ...
@overload
def firls(
    numtaps: onp.ToJustInt,
    bands: onp.ToFloat2D,
    desired: onp.ToFloat2D,
    *,
    weight: onp.ToFloat1D | None = None,
    fs: onp.ToFloat | None = None,
) -> onp.Array1D[np.float64]: ...

#
def remez(
    numtaps: onp.ToJustInt,
    bands: onp.ToFloat1D,
    desired: onp.ToFloat1D,
    *,
    weight: onp.ToFloat1D | None = None,
    type: _RemezFilterType = "bandpass",
    maxiter: onp.ToJustInt = 25,
    grid_density: onp.ToJustInt = 16,
    fs: onp.ToFloat | None = None,
) -> onp.Array1D[np.float64]: ...

#
def minimum_phase(
    h: onp.ToFloat1D,
    method: _LinearPhaseFIRMethod = "homomorphic",
    n_fft: onp.ToJustInt | None = None,
    *,
    half: op.CanBool = True,
) -> onp.Array1D[np.float64]: ...
