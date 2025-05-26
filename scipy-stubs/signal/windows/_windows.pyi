from types import ModuleType
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeAliasType

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc
from _typeshed import Incomplete
from scipy._typing import Falsy, Truthy

__all__ = [
    "barthann",
    "bartlett",
    "blackman",
    "blackmanharris",
    "bohman",
    "boxcar",
    "chebwin",
    "cosine",
    "dpss",
    "exponential",
    "flattop",
    "gaussian",
    "general_cosine",
    "general_gaussian",
    "general_hamming",
    "get_window",
    "hamming",
    "hann",
    "kaiser",
    "kaiser_bessel_derived",
    "lanczos",
    "nuttall",
    "parzen",
    "taylor",
    "triang",
    "tukey",
]

###

_Xp = TypeAliasType("_Xp", ModuleType)
_Device = TypeAliasType("_Device", Incomplete)

_Float64_1D: TypeAlias = onp.Array1D[np.float64]
_Float64_2D: TypeAlias = onp.Array2D[np.float64]

_ToBool: TypeAlias = bool | np.bool_[Any]
_ToInt: TypeAlias = int | np.int16 | np.int32 | np.int64
_ToFloat: TypeAlias = float | npc.floating | npc.integer

_Norm: TypeAlias = Literal[2, "approximate", "subsample"]

_NameWindow0: TypeAlias = Literal[
    "barthann", "brthan", "bth",
    "bartlett", "bart", "brt",
    "blackman", "black", "blk",
    "blackmanharris", "blackharr", "bkh",
    "bohman", "bman", "bmn",
    "boxcar", "box", "ones", "rect", "rectangular",
    "cosine", "halfcosine",
    "exponential", "poisson",
    "flattop", "flat", "flt",
    "hamming", "hamm", "ham",
    "hann", "han",
    "lanczos", "sinc",
    "nuttall", "nutl", "nut",
    "parzen", "parz", "par",
    "taylor", "taylorwin",
    "triangle", "triang", "tri",
    "tukey", "tuk",
]  # fmt: skip
_NameWindow1: TypeAlias = Literal[
    "chebwin", "cheb",
    "dpss",
    "exponential", "poisson",
    "gaussian", "gauss", "gss",
    "general hamming", "general_hamming",
    "kaiser", "ksr",
    "kaiser bessel derived", "kbd",
    "tukey", "tuk",
]  # fmt: skip
_NameWindow2: TypeAlias = Literal[
    "exponential", "poisson",
    "general gaussian", "general_gaussian", "general gauss", "general_gauss", "ggs",
]  # fmt: skip
_NameTaylor: TypeAlias = Literal["taylor", "taylorwin"]
_NameGenCos: TypeAlias = Literal["general cosine", "general_cosine"]
_NameDPSS: TypeAlias = Literal["dpss"]

_ToWindow0: TypeAlias = _NameWindow0 | tuple[_NameWindow0]
_ToWindow1: TypeAlias = _ToFloat | tuple[_NameWindow1, _ToFloat]
_ToWindow2: TypeAlias = tuple[_NameWindow2, _ToFloat, _ToFloat]
_ToGenCos: TypeAlias = tuple[_NameGenCos, onp.ToFloat1D]
_ToTaylor: TypeAlias = tuple[_NameTaylor, _ToInt] | tuple[_NameTaylor, _ToInt, _ToInt] | tuple[_NameTaylor, _ToInt, _ToInt, bool]
_ToDPSS: TypeAlias = tuple[_NameDPSS, _ToFloat, op.CanIndex]

_ToWindow = TypeAliasType("_ToWindow", _ToWindow0 | _ToWindow1 | _ToWindow2 | _ToGenCos | _ToTaylor | _ToDPSS)

###

def get_window(
    window: _ToWindow, Nx: _ToInt, fftbins: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None
) -> _Float64_1D: ...

#
def barthann(M: _ToInt, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...
def bartlett(M: _ToInt, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...
def blackman(M: _ToInt, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...
def blackmanharris(M: _ToInt, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...
def bohman(M: _ToInt, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...
def boxcar(M: _ToInt, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...
def cosine(M: _ToInt, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...
def flattop(M: _ToInt, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...
def hamming(M: _ToInt, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...
def hann(M: _ToInt, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...
def lanczos(M: _ToInt, *, sym: _ToBool = True, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...
def nuttall(M: _ToInt, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...
def parzen(M: _ToInt, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...
def triang(M: _ToInt, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None) -> _Float64_1D: ...

#
def chebwin(
    M: _ToInt, at: _ToFloat, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None
) -> _Float64_1D: ...
def gaussian(
    M: _ToInt, std: _ToFloat, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None
) -> _Float64_1D: ...
def general_hamming(
    M: _ToInt, alpha: _ToFloat, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None
) -> _Float64_1D: ...
def kaiser(
    M: _ToInt, beta: _ToFloat, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None
) -> _Float64_1D: ...
def kaiser_bessel_derived(
    M: _ToInt, beta: _ToFloat, *, sym: _ToBool = True, xp: _Xp | None = None, device: _Device | None = None
) -> _Float64_1D: ...
def tukey(
    M: _ToInt, alpha: _ToFloat = 0.5, sym: _ToBool = True, *, xp: _Xp | None = None, device: _Device | None = None
) -> _Float64_1D: ...

#
def general_cosine(M: _ToInt, a: onp.ToFloat1D, sym: _ToBool = True) -> _Float64_1D: ...

#
def exponential(
    M: _ToInt,
    center: _ToFloat | None = None,
    tau: _ToFloat = 1.0,
    sym: _ToBool = True,
    *,
    xp: _Xp | None = None,
    device: _Device | None = None,
) -> _Float64_1D: ...
def general_gaussian(
    M: _ToInt,
    p: _ToFloat,
    sig: _ToFloat,
    sym: _ToBool = True,
    *,
    xp: _Xp | None = None,
    device: _Device | None = None,
) -> _Float64_1D: ...

#
def taylor(
    M: _ToInt,
    nbar: onp.ToInt = 4,
    sll: onp.ToInt = 30,
    norm: _ToBool = True,
    sym: _ToBool = True,
    *,
    xp: _Xp | None = None,
    device: _Device | None = None,
) -> _Float64_1D: ...

#
@overload  # Kmax=None, return_ratios=False
def dpss(
    M: _ToInt,
    NW: _ToFloat,
    Kmax: None = None,
    sym: _ToBool = True,
    norm: _Norm | None = None,
    return_ratios: Falsy = False,
    *,
    xp: _Xp | None = None,
    device: _Device | None = None,
) -> _Float64_1D: ...
@overload  # Kmax=None, return_ratios=True, /
def dpss(
    M: _ToInt,
    NW: _ToFloat,
    Kmax: None,
    sym: _ToBool,
    norm: _Norm | None,
    return_ratios: Truthy,
    *,
    xp: _Xp | None = None,
    device: _Device | None = None,
) -> tuple[_Float64_1D, np.float64]: ...
@overload  # Kmax=None, *, return_ratios=True
def dpss(
    M: _ToInt,
    NW: _ToFloat,
    Kmax: None = None,
    sym: _ToBool = True,
    norm: _Norm | None = None,
    *,
    return_ratios: Truthy,
    xp: _Xp | None = None,
    device: _Device | None = None,
) -> tuple[_Float64_1D, np.float64]: ...
@overload  # Kmax, return_ratios=False
def dpss(
    M: _ToInt,
    NW: _ToFloat,
    Kmax: op.CanIndex,
    sym: _ToBool = True,
    norm: _Norm | None = None,
    return_ratios: Falsy = False,
    *,
    xp: _Xp | None = None,
    device: _Device | None = None,
) -> _Float64_2D: ...
@overload  # Kmax, return_ratios=True, /
def dpss(
    M: _ToInt,
    NW: _ToFloat,
    Kmax: op.CanIndex,
    sym: _ToBool,
    norm: _Norm | None,
    return_ratios: Truthy,
    *,
    xp: _Xp | None = None,
    device: _Device | None = None,
) -> tuple[_Float64_2D, _Float64_1D]: ...
@overload  # Kmax, *, return_ratios=True
def dpss(
    M: _ToInt,
    NW: _ToFloat,
    Kmax: op.CanIndex,
    sym: _ToBool = True,
    norm: _Norm | None = None,
    *,
    return_ratios: Truthy,
    xp: _Xp | None = None,
    device: _Device | None = None,
) -> tuple[_Float64_2D, _Float64_1D]: ...
