from _typeshed import Incomplete
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Literal as L, SupportsIndex, TypeAlias, overload
from typing_extensions import TypeAliasType, TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = [
    "BadCoefficients",
    "band_stop_obj",
    "bessel",
    "besselap",
    "bilinear",
    "bilinear_zpk",
    "buttap",
    "butter",
    "buttord",
    "cheb1ap",
    "cheb1ord",
    "cheb2ap",
    "cheb2ord",
    "cheby1",
    "cheby2",
    "ellip",
    "ellipap",
    "ellipord",
    "findfreqs",
    "freqs",
    "freqs_zpk",
    "freqz",
    "freqz_sos",
    "freqz_zpk",
    "gammatone",
    "group_delay",
    "iircomb",
    "iirdesign",
    "iirfilter",
    "iirnotch",
    "iirpeak",
    "lp2bp",
    "lp2bp_zpk",
    "lp2bs",
    "lp2bs_zpk",
    "lp2hp",
    "lp2hp_zpk",
    "lp2lp",
    "lp2lp_zpk",
    "normalize",
    "sos2tf",
    "sos2zpk",
    "sosfreqz",
    "tf2sos",
    "tf2zpk",
    "zpk2sos",
    "zpk2tf",
]

_Floating: TypeAlias = npc.floating
_CFloating: TypeAlias = npc.complexfloating

_Floating1D: TypeAlias = onp.Array1D[npc.floating]
_FloatingND: TypeAlias = onp.ArrayND[npc.floating]
_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float64]
_Complex1D: TypeAlias = onp.Array1D[np.complex128]
_ComplexND: TypeAlias = onp.ArrayND[np.complex128]

_Order: TypeAlias = L[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

_SCT_z = TypeVar("_SCT_z", bound=np.generic)
_SCT_p = TypeVar("_SCT_p", bound=np.generic, default=np.complex128)
_SCT_k = TypeVar("_SCT_k", bound=np.generic | float, default=np.float64)
_ZPK = TypeAliasType("_ZPK", tuple[onp.Array1D[_SCT_z], onp.Array1D[_SCT_p], _SCT_k], type_params=(_SCT_z, _SCT_p, _SCT_k))

_SCT_ba = TypeVar("_SCT_ba", bound=npc.floating, default=np.float64)
_Ba1D = TypeAliasType("_Ba1D", tuple[onp.Array1D[_SCT_ba], onp.Array1D[_SCT_ba]], type_params=(_SCT_ba,))
_Ba2D = TypeAliasType("_Ba2D", tuple[onp.Array2D[_SCT_ba], onp.Array1D[_SCT_ba]], type_params=(_SCT_ba,))

# excludes `float16` and `longdouble`
_ToFloat: TypeAlias = float | np.float32 | np.float64 | npc.integer
_ToFloat1D: TypeAlias = Sequence[_ToFloat] | onp.CanArrayND[np.float32 | np.float64 | npc.integer]
_ToFloat2D: TypeAlias = Sequence[_ToFloat1D] | onp.CanArrayND[np.float32 | np.float64 | npc.integer]

_FType0: TypeAlias = L["butter", "cheby1", "cheby2", "ellip"]
_FType: TypeAlias = _FType0 | L["bessel"]
_BType0: TypeAlias = L["bandpass", "lowpass", "highpass", "bandstop"]
_BType: TypeAlias = _BType0 | L["band", "pass", "bp", "bands", "stop", "bs", "low", "lp", "l", "high", "hp", "h"]
_Pairing: TypeAlias = L["nearest", "keep_odd", "minimal"]
_Norm: TypeAlias = L["phase", "delay", "mag"]

_WorNReal: TypeAlias = SupportsIndex | onp.ToFloat1D | None

###

class BadCoefficients(UserWarning): ...

#
def findfreqs(num: onp.ToComplex1D, den: onp.ToComplex1D, N: SupportsIndex, kind: L["ba", "zp"] = "ba") -> _Float1D: ...

#
@overload  # worN: real
def freqs(
    b: onp.ToComplex1D, a: onp.ToComplex1D, worN: _WorNReal = 200, plot: Callable[[_Float1D, _Complex1D], object] | None = None
) -> tuple[_Float1D, _Complex1D]: ...
@overload  # worN: complex
def freqs(
    b: onp.ToComplex1D,
    a: onp.ToComplex1D,
    worN: onp.ToJustComplex1D,
    plot: Callable[[_Float1D, _Complex1D], object] | Callable[[_Complex1D, _Complex1D], object] | None = None,
) -> tuple[_Complex1D, _Complex1D]: ...

#
@overload  # worN: real
def freqs_zpk(
    z: onp.ToComplex1D, p: onp.ToComplex1D, k: onp.ToComplex1D, worN: _WorNReal = 200
) -> tuple[_Float1D, _Complex1D]: ...
@overload  # worN: complex
def freqs_zpk(
    z: onp.ToComplex1D, p: onp.ToComplex1D, k: onp.ToComplex1D, worN: onp.ToJustComplex1D
) -> tuple[_Complex1D, _Complex1D]: ...

#
@overload  # worN: real
def freqz(
    b: onp.ToComplex | onp.ToComplexND,
    a: onp.ToComplex | onp.ToComplexND = 1,
    worN: _WorNReal = 512,
    whole: bool = False,
    plot: Callable[[_FloatND, _ComplexND], object] | None = None,
    fs: float = 6.283185307179586,
    include_nyquist: bool = False,
) -> tuple[_FloatND, _ComplexND]: ...
@overload  # worN: complex  (keyword)
def freqz(
    b: onp.ToComplex | onp.ToComplexND,
    a: onp.ToComplex | onp.ToComplexND = 1,
    *,
    worN: onp.ToJustComplex1D,
    whole: bool = False,
    plot: Callable[[_FloatND, _ComplexND], object] | None = None,
    fs: float = 6.283185307179586,
    include_nyquist: bool = False,
) -> tuple[_ComplexND, _ComplexND]: ...

#
@overload  # worN: real
def freqz_zpk(
    z: onp.ToComplex1D,
    p: onp.ToComplex1D,
    k: onp.ToComplex1D,
    worN: _WorNReal = 512,
    whole: bool = False,
    fs: float = 6.283185307179586,
) -> tuple[_FloatND, _ComplexND]: ...
@overload  # worN: complex
def freqz_zpk(
    z: onp.ToComplex1D,
    p: onp.ToComplex1D,
    k: onp.ToComplex1D,
    worN: onp.ToJustComplex1D,
    whole: bool = False,
    fs: float = 6.283185307179586,
) -> tuple[_ComplexND, _ComplexND]: ...

#
@overload  # w: real
def group_delay(
    system: tuple[onp.ToComplex1D, onp.ToComplex1D], w: _WorNReal = 512, whole: bool = False, fs: float = 6.283185307179586
) -> _Ba1D: ...
@overload  # w: complex
def group_delay(
    system: tuple[onp.ToComplex1D, onp.ToComplex1D], w: onp.ToJustComplex1D, whole: bool = False, fs: float = 6.283185307179586
) -> tuple[_Complex1D, _Float1D]: ...

#
@overload  # worN: real
def freqz_sos(
    sos: onp.ToFloat2D, worN: _WorNReal = 512, whole: bool = False, fs: float = 6.283185307179586
) -> tuple[_Float1D, _Complex1D]: ...
@overload  # worN: real
def freqz_sos(
    sos: onp.ToFloat2D, worN: onp.ToJustComplex1D, whole: bool = False, fs: float = 6.283185307179586
) -> tuple[_Complex1D, _Complex1D]: ...

sosfreqz = freqz_sos

#
def tf2zpk(b: _ToFloat1D, a: _ToFloat1D) -> _ZPK[np.float64 | np.complex128] | _ZPK[np.complex64, np.complex64, np.float32]: ...
def tf2sos(b: _ToFloat1D, a: _ToFloat1D, pairing: _Pairing | None = None, *, analog: bool = False) -> _Float2D: ...

#
def zpk2tf(z: onp.ToFloat1D, p: onp.ToFloat1D, k: float) -> _Ba1D: ...
def zpk2sos(
    z: onp.ToFloat1D, p: onp.ToFloat1D, k: float, pairing: _Pairing | None = None, *, analog: bool = False
) -> _Float2D: ...

# TODO: better overloads
@overload
def normalize(b: onp.ToFloatStrict1D, a: onp.ToFloat1D) -> _Ba1D[_Floating]: ...
@overload
def normalize(b: onp.ToFloatStrict2D, a: onp.ToFloat1D) -> _Ba2D[_Floating]: ...
@overload
def normalize(b: onp.ToFloat1D | onp.ToFloat2D, a: onp.ToFloat1D) -> _Ba1D[_Floating] | _Ba2D[_Floating]: ...

# TODO: overloads
def sos2tf(sos: onp.ToFloat2D) -> tuple[_Floating1D, _Floating1D]: ...

# TODO: overloads
def sos2zpk(sos: _ToFloat2D) -> _ZPK[np.complex128, np.complex128, np.float32 | np.float64]: ...

#  # TODO: better overloads
@overload
def lp2lp(b: onp.ToFloatStrict1D, a: onp.ToFloat1D, wo: float = 1.0) -> _Ba1D | _Ba1D[np.longdouble]: ...
@overload
def lp2lp(b: onp.ToFloatStrict2D, a: onp.ToFloat1D, wo: float = 1.0) -> _Ba2D | _Ba2D[np.longdouble]: ...
@overload
def lp2lp(
    b: onp.ToFloat1D | onp.ToFloat2D, a: onp.ToFloat1D, wo: float = 1.0
) -> _Ba1D | _Ba1D[np.longdouble] | _Ba2D | _Ba2D[np.longdouble]: ...

# TODO: overloads
def lp2hp(
    b: onp.ToFloat1D, a: onp.ToFloat1D, wo: float = 1.0
) -> _Ba1D | _Ba1D[np.float16] | _Ba1D[np.float32] | _Ba1D[np.longdouble]: ...

# TODO: overloads
def lp2bp(
    b: onp.ToFloat1D, a: onp.ToFloat1D, wo: float = 1.0, bw: float = 1.0
) -> _Ba1D | _Ba1D[np.float32] | _Ba1D[np.longdouble]: ...

# TODO: overloads
def lp2bs(
    b: onp.ToFloat1D, a: onp.ToFloat1D, wo: float = 1.0, bw: float = 1.0
) -> _Ba1D | _Ba1D[np.float32] | _Ba1D[np.longdouble]: ...

# TODO: overloads
def lp2lp_zpk(z: onp.ToComplex1D, p: onp.ToComplex1D, k: float, wo: float = 1.0) -> _ZPK[npc.inexact, _CFloating, _Floating]: ...
def lp2hp_zpk(z: onp.ToComplex1D, p: onp.ToComplex1D, k: float, wo: float = 1.0) -> _ZPK[npc.inexact, _CFloating, _Floating]: ...

# TODO: overloads
def lp2bp_zpk(
    z: onp.ToComplex1D, p: onp.ToComplex1D, k: float, wo: float = 1.0, bw: float = 1.0
) -> _ZPK[npc.inexact, _CFloating, _Floating]: ...

# TODO: overloads
def lp2bs_zpk(
    z: onp.ToComplex1D, p: onp.ToComplex1D, k: float, wo: float = 1.0, bw: float = 1.0
) -> _ZPK[npc.inexact, _CFloating, _Floating]: ...

#
def bilinear(b: onp.ToFloat1D, a: onp.ToFloat1D, fs: float = 1.0) -> _Ba1D: ...

# TODO: better overloads
@overload
def bilinear_zpk(
    z: onp.ToFloat1D, p: onp.ToComplex1D, k: float, fs: float
) -> _ZPK[np.float64, _CFloating] | _ZPK[np.longdouble, _CFloating, np.longdouble]: ...
@overload
def bilinear_zpk(
    z: onp.ToComplex1D, p: onp.ToComplex1D, k: float, fs: float
) -> _ZPK[np.float64 | np.complex128, _CFloating] | _ZPK[np.longdouble | np.clongdouble, _CFloating, np.longdouble]: ...

#
@overload  # output="ba" (default)
def iirdesign(
    wp: float | onp.ToFloat1D,
    ws: float | onp.ToFloat1D,
    gpass: float,
    gstop: float,
    analog: bool = False,
    ftype: _FType0 = "ellip",
    output: L["ba"] = "ba",
    fs: float | None = None,
) -> _Ba1D: ...
@overload  # output="zpk" (positional)
def iirdesign(
    wp: float | onp.ToFloat1D,
    ws: float | onp.ToFloat1D,
    gpass: float,
    gstop: float,
    analog: bool,
    ftype: _FType0,
    output: L["zpk"],
    fs: float | None = None,
) -> _ZPK[np.complex128]: ...
@overload  # output="zpk" (keyword)
def iirdesign(
    wp: float | onp.ToFloat1D,
    ws: float | onp.ToFloat1D,
    gpass: float,
    gstop: float,
    analog: bool = False,
    ftype: _FType0 = "ellip",
    *,
    output: L["zpk"],
    fs: float | None = None,
) -> _ZPK[np.complex128]: ...
@overload  # output="sos" (positional)
def iirdesign(
    wp: float | onp.ToFloat1D,
    ws: float | onp.ToFloat1D,
    gpass: float,
    gstop: float,
    analog: bool,
    ftype: _FType0,
    output: L["sos"],
    fs: float | None = None,
) -> _Float2D: ...
@overload  # output="sos" (keyword)
def iirdesign(
    wp: float | onp.ToFloat1D,
    ws: float | onp.ToFloat1D,
    gpass: float,
    gstop: float,
    analog: bool = False,
    ftype: _FType0 = "ellip",
    *,
    output: L["sos"],
    fs: float | None = None,
) -> _Float2D: ...

#
@overload  # output="ba" (default)
def iirfilter(
    N: int,
    Wn: float | onp.ToFloat1D,
    rp: float | None = None,
    rs: float | None = None,
    btype: _BType = "band",
    analog: bool = False,
    ftype: _FType = "butter",
    output: L["ba"] = "ba",
    fs: float | None = None,
) -> _Ba1D: ...
@overload  # output="zpk" (positional)
def iirfilter(
    N: int,
    Wn: float | onp.ToFloat1D,
    rp: float | None,
    rs: float | None,
    btype: _BType,
    analog: bool,
    ftype: _FType,
    output: L["zpk"],
    fs: float | None = None,
) -> _ZPK[np.complex128]: ...
@overload  # output="zpk" (keyword)
def iirfilter(
    N: int,
    Wn: float | onp.ToFloat1D,
    rp: float | None = None,
    rs: float | None = None,
    btype: _BType = "band",
    analog: bool = False,
    ftype: _FType = "butter",
    *,
    output: L["zpk"],
    fs: float | None = None,
) -> _ZPK[np.complex128]: ...
@overload  # output="sos" (positional)
def iirfilter(
    N: int,
    Wn: float | onp.ToFloat1D,
    rp: float | None,
    rs: float | None,
    btype: _BType,
    analog: bool,
    ftype: _FType,
    output: L["sos"],
    fs: float | None = None,
) -> _Float2D: ...
@overload  # output="sos" (keyword)
def iirfilter(
    N: int,
    Wn: float | onp.ToFloat1D,
    rp: float | None = None,
    rs: float | None = None,
    btype: _BType = "band",
    analog: bool = False,
    ftype: _FType = "butter",
    *,
    output: L["sos"],
    fs: float | None = None,
) -> _Float2D: ...

#
@overload  # output="ba" (default)
def butter(
    N: int,
    Wn: float | onp.ToFloat1D,
    btype: _BType = "low",
    analog: bool = False,
    output: L["ba"] = "ba",
    fs: float | None = None,
) -> _Ba1D: ...
@overload  # output="zpk" (keyword)
def butter(
    N: int, Wn: float | onp.ToFloat1D, btype: _BType = "low", analog: bool = False, *, output: L["zpk"], fs: float | None = None
) -> _ZPK[np.float64, np.complex128, float]: ...
@overload  # output="sos" (keyword)
def butter(
    N: int, Wn: float | onp.ToFloat1D, btype: _BType = "low", analog: bool = False, *, output: L["sos"], fs: float | None = None
) -> _Float2D: ...

#
@overload  # output="ba" (default)
def cheby1(
    N: int,
    rp: float,
    Wn: float | onp.ToFloat1D,
    btype: _BType = "low",
    analog: bool = False,
    output: L["ba"] = "ba",
    fs: float | None = None,
) -> _Ba1D: ...
@overload  # output="zpk" (positional)
def cheby1(
    N: int, rp: float, Wn: float | onp.ToFloat1D, btype: _BType, analog: bool, output: L["zpk"], fs: float | None = None
) -> _ZPK[np.complex128]: ...
@overload  # output="zpk" (keyword)
def cheby1(
    N: int,
    rp: float,
    Wn: float | onp.ToFloat1D,
    btype: _BType = "low",
    analog: bool = False,
    *,
    output: L["zpk"],
    fs: float | None = None,
) -> _ZPK[np.complex128]: ...
@overload  # output="sos" (positional)
def cheby1(
    N: int, rp: float, Wn: float | onp.ToFloat1D, btype: _BType, analog: bool, output: L["sos"], fs: float | None = None
) -> _Float2D: ...
@overload  # output="sos" (keyword)
def cheby1(
    N: int,
    rp: float,
    Wn: float | onp.ToFloat1D,
    btype: _BType = "low",
    analog: bool = False,
    *,
    output: L["sos"],
    fs: float | None = None,
) -> _Float2D: ...

#
@overload  # output="ba" (default)
def cheby2(
    N: int,
    rs: float,
    Wn: float | onp.ToFloat1D,
    btype: _BType = "low",
    analog: bool = False,
    output: L["ba"] = "ba",
    fs: float | None = None,
) -> _Ba1D: ...
@overload  # output="zpk" (positional)
def cheby2(
    N: int, rs: float, Wn: float | onp.ToFloat1D, btype: _BType, analog: bool, output: L["zpk"], fs: float | None = None
) -> _ZPK[np.complex128]: ...
@overload  # output="zpk" (keyword)
def cheby2(
    N: int,
    rs: float,
    Wn: float | onp.ToFloat1D,
    btype: _BType = "low",
    analog: bool = False,
    *,
    output: L["zpk"],
    fs: float | None = None,
) -> _ZPK[np.complex128]: ...
@overload  # output="sos" (positional)
def cheby2(
    N: int, rs: float, Wn: float | onp.ToFloat1D, btype: _BType, analog: bool, output: L["sos"], fs: float | None = None
) -> _Float2D: ...
@overload  # output="sos" (keyword)
def cheby2(
    N: int,
    rs: float,
    Wn: float | onp.ToFloat1D,
    btype: _BType = "low",
    analog: bool = False,
    *,
    output: L["sos"],
    fs: float | None = None,
) -> _Float2D: ...

#
@overload  # output="ba" (default)
def ellip(
    N: int,
    rp: float,
    rs: float,
    Wn: float | onp.ToFloat1D,
    btype: _BType = "low",
    analog: bool = False,
    output: L["ba"] = "ba",
    fs: float | None = None,
) -> _Ba1D: ...
@overload  # output="zpk" (postitional)
def ellip(
    N: int,
    rp: float,
    rs: float,
    Wn: float | onp.ToFloat1D,
    btype: _BType,
    analog: bool,
    output: L["zpk"],
    fs: float | None = None,
) -> _ZPK[np.complex128]: ...
@overload  # output="zpk" (keyword)
def ellip(
    N: int,
    rp: float,
    rs: float,
    Wn: float | onp.ToFloat1D,
    btype: _BType = "low",
    analog: bool = False,
    *,
    output: L["zpk"],
    fs: float | None = None,
) -> _ZPK[np.complex128]: ...
@overload  # output="sos" (postitional)
def ellip(
    N: int,
    rp: float,
    rs: float,
    Wn: float | onp.ToFloat1D,
    btype: _BType,
    analog: bool,
    output: L["sos"],
    fs: float | None = None,
) -> _Float2D: ...
@overload  # output="sos" (keyword)
def ellip(
    N: int,
    rp: float,
    rs: float,
    Wn: float | onp.ToFloat1D,
    btype: _BType = "low",
    analog: bool = False,
    *,
    output: L["sos"],
    fs: float | None = None,
) -> _Float2D: ...

#
@overload  # output="ba" (default)
def bessel(
    N: int,
    Wn: float | onp.ToFloat1D,
    btype: _BType = "low",
    analog: bool = False,
    output: L["ba"] = "ba",
    norm: _Norm = "phase",
    fs: float | None = None,
) -> _Ba1D: ...
@overload  # output="zpk" (postitional)
def bessel(
    N: int,
    Wn: float | onp.ToFloat1D,
    btype: _BType,
    analog: bool,
    output: L["zpk"],
    norm: _Norm = "phase",
    fs: float | None = None,
) -> _ZPK[np.float64]: ...
@overload  # output="zpk" (keyword)
def bessel(
    N: int,
    Wn: float | onp.ToFloat1D,
    btype: _BType = "low",
    analog: bool = False,
    *,
    output: L["zpk"],
    norm: _Norm = "phase",
    fs: float | None = None,
) -> _ZPK[np.float64]: ...
@overload  # output="sos" (postitional)
def bessel(
    N: int,
    Wn: float | onp.ToFloat1D,
    btype: _BType,
    analog: bool,
    output: L["sos"],
    norm: _Norm = "phase",
    fs: float | None = None,
) -> _Float2D: ...
@overload  # output="sos" (keyword)
def bessel(
    N: int,
    Wn: float | onp.ToFloat1D,
    btype: _BType = "low",
    analog: bool = False,
    *,
    output: L["sos"],
    norm: _Norm = "phase",
    fs: float | None = None,
) -> _Float2D: ...

# TODO: overloads
def band_stop_obj(
    wp: float,
    ind: L[0, 1] | npc.integer,  # bool doesn't work
    passb: onp.ArrayND[_Floating | npc.integer],  # 1-d
    stopb: onp.ArrayND[_Floating | npc.integer],  # 1-d
    gpass: float,
    gstop: float,
    type: L["butter", "cheby", "ellip"],
) -> np.float64 | np.longdouble: ...

# TODO: better overloads
@overload
def buttord(
    wp: float, ws: float | onp.ToFloatND, gpass: float, gstop: float, analog: bool = False, fs: float | None = None
) -> tuple[int, np.float64 | np.longdouble]: ...
@overload
def buttord(
    wp: onp.ToFloatND, ws: float | onp.ToFloatND, gpass: float, gstop: float, analog: bool = False, fs: float | None = None
) -> tuple[int, onp.Array1D[np.float64 | np.longdouble]]: ...

# TODO: better overloads
@overload
def cheb1ord(
    wp: float, ws: float | onp.ToFloatND, gpass: float, gstop: float, analog: bool = False, fs: float | None = None
) -> tuple[int, _Floating]: ...
@overload
def cheb1ord(
    wp: onp.ToFloatND, ws: float | onp.ToFloatND, gpass: float, gstop: float, analog: bool = False, fs: float | None = None
) -> tuple[int, _FloatingND]: ...

# TODO: better overloads
@overload
def cheb2ord(
    wp: float, ws: float | onp.ToFloatND, gpass: float, gstop: float, analog: bool = False, fs: float | None = None
) -> tuple[int, _Floating]: ...
@overload
def cheb2ord(
    wp: onp.ToFloatND, ws: float | onp.ToFloatND, gpass: float, gstop: float, analog: bool = False, fs: float | None = None
) -> tuple[int, _FloatND]: ...  # only nd-output is cast to float64

# TODO: better overloads
@overload
def ellipord(
    wp: float, ws: float | onp.ToFloatND, gpass: float, gstop: float, analog: bool = False, fs: float | None = None
) -> tuple[int, _Floating]: ...
@overload
def ellipord(
    wp: onp.ToFloatND, ws: float | onp.ToFloatND, gpass: float, gstop: float, analog: bool = False, fs: float | None = None
) -> tuple[int, _FloatingND]: ...

#
@overload
def buttap(N: int, *, xp: None = None, device: None = None) -> tuple[_Float1D, _Complex1D, L[1]]: ...
@overload
def buttap(N: int, *, xp: ModuleType, device: object = None) -> tuple[Incomplete, Incomplete, L[1]]: ...

#
@overload
def cheb1ap(N: int, rp: float, *, xp: None = None, device: None = None) -> tuple[_Float1D, _Complex1D, float]: ...
@overload
def cheb1ap(N: int, rp: float, *, xp: ModuleType, device: object = None) -> tuple[Incomplete, Incomplete, float]: ...

#
@overload
def cheb2ap(N: int, rs: float, *, xp: None = None, device: None = None) -> tuple[_Complex1D, _Complex1D, float]: ...
@overload
def cheb2ap(N: int, rs: float, *, xp: ModuleType, device: object = None) -> tuple[Incomplete, Incomplete, float]: ...

#
@overload
def ellipap(N: int, rp: float, rs: float, *, xp: None = None, device: None = None) -> tuple[_Complex1D, _Complex1D, float]: ...
@overload
def ellipap(N: int, rp: float, rs: float, *, xp: ModuleType, device: object = None) -> tuple[Incomplete, Incomplete, float]: ...

#
@overload
def besselap(N: int, norm: _Norm = "phase", *, xp: None = None, device: None = None) -> tuple[_Float1D, _Complex1D, float]: ...
@overload
def besselap(N: int, norm: _Norm = "phase", *, xp: ModuleType, device: object = None) -> tuple[Incomplete, Incomplete, float]: ...

#
@overload
def iirnotch(w0: float, Q: float, fs: float = 2.0, *, xp: None = None, device: None = None) -> _Ba1D: ...
@overload
def iirnotch(w0: float, Q: float, fs: float = 2.0, *, xp: ModuleType, device: object = None) -> tuple[Incomplete, Incomplete]: ...

#
@overload
def iirpeak(w0: float, Q: float, fs: float = 2.0, *, xp: None = None, device: None = None) -> _Ba1D: ...
@overload
def iirpeak(w0: float, Q: float, fs: float = 2.0, *, xp: ModuleType, device: object = None) -> tuple[Incomplete, Incomplete]: ...

#
@overload
def iircomb(
    w0: float,
    Q: float,
    ftype: L["notch", "peak"] = "notch",
    fs: float = 2.0,
    *,
    pass_zero: bool = False,
    xp: None = None,
    device: None = None,
) -> _Ba1D: ...
@overload
def iircomb(
    w0: float,
    Q: float,
    ftype: L["notch", "peak"] = "notch",
    fs: float = 2.0,
    *,
    pass_zero: bool = False,
    xp: ModuleType,
    device: object = None,
) -> tuple[Incomplete, Incomplete]: ...

#
@overload
def gammatone(
    freq: float,
    ftype: L["fir", "iir"],
    order: _Order | None = None,
    numtaps: int | None = None,
    fs: float | None = None,
    *,
    xp: None = None,
    device: None = None,
) -> _Ba1D: ...
@overload
def gammatone(
    freq: float,
    ftype: L["fir", "iir"],
    order: _Order | None = None,
    numtaps: int | None = None,
    fs: float | None = None,
    *,
    xp: ModuleType,
    device: object = None,
) -> tuple[Incomplete, Incomplete]: ...

# ???
def maxflat() -> None: ...
def yulewalk() -> None: ...
