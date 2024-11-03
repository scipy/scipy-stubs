from collections.abc import Sequence
from typing import Literal, TypeAlias, overload
from typing_extensions import Unpack

import numpy as np
import optype as op
from scipy._typing import AnyInt, AnyReal

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

_Array_f8_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
_Array_f8_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]

_Norm: TypeAlias = Literal[2, "approximate", "subsample"]
_WindowLength: TypeAlias = int | np.int16 | np.int32 | np.int64
_Window: TypeAlias = Literal[
    "barthann",
    "brthan",
    "bth",
    "bartlett",
    "bart",
    "brt",
    "blackman",
    "black",
    "blk",
    "blackmanharris",
    "blackharr",
    "bkh",
    "bohman",
    "bman",
    "bmn",
    "boxcar",
    "box",
    "ones",
    "rect",
    "rectangular",
    "cosine",
    "halfcosine",
    "exponential",
    "poisson",
    "flattop",
    "flat",
    "flt",
    "hamming",
    "hamm",
    "ham",
    "hann",
    "han",
    "lanczos",
    "sinc",
    "nuttall",
    "nutl",
    "nut",
    "parzen",
    "parz",
    "par",
    "taylor",
    "taylorwin",
    "triangle",
    "triang",
    "tri",
    "tukey",
    "tuk",
]
_WindowNeedsParams: TypeAlias = Literal[
    "chebwin",
    "cheb",
    "dpss",
    "gaussian",
    "gauss",
    "gss",
    "general cosine",
    "general_cosine",
    "general gaussian",
    "general_gaussian",
    "general gauss",
    "general_gauss",
    "ggs",
    "general hamming",
    "general_hamming",
    "kaiser",
    "ksr",
    "kaiser bessel derived",
    "kbd",
]

def general_cosine(M: _WindowLength, a: Sequence[AnyReal], sym: op.CanBool = True) -> _Array_f8_1d: ...
def boxcar(M: _WindowLength, sym: op.CanBool = True) -> _Array_f8_1d: ...
def triang(M: _WindowLength, sym: op.CanBool = True) -> _Array_f8_1d: ...
def parzen(M: _WindowLength, sym: op.CanBool = True) -> _Array_f8_1d: ...
def bohman(M: _WindowLength, sym: op.CanBool = True) -> _Array_f8_1d: ...
def blackman(M: _WindowLength, sym: op.CanBool = True) -> _Array_f8_1d: ...
def nuttall(M: _WindowLength, sym: op.CanBool = True) -> _Array_f8_1d: ...
def blackmanharris(M: _WindowLength, sym: op.CanBool = True) -> _Array_f8_1d: ...
def flattop(M: _WindowLength, sym: op.CanBool = True) -> _Array_f8_1d: ...
def bartlett(M: _WindowLength, sym: op.CanBool = True) -> _Array_f8_1d: ...
def hann(M: _WindowLength, sym: op.CanBool = True) -> _Array_f8_1d: ...
def tukey(M: _WindowLength, alpha: AnyReal = 0.5, sym: op.CanBool = True) -> _Array_f8_1d: ...
def barthann(M: _WindowLength, sym: op.CanBool = True) -> _Array_f8_1d: ...
def general_hamming(M: _WindowLength, alpha: AnyReal, sym: op.CanBool = True) -> _Array_f8_1d: ...
def hamming(M: _WindowLength, sym: op.CanBool = True) -> _Array_f8_1d: ...
def kaiser(M: _WindowLength, beta: AnyReal, sym: op.CanBool = True) -> _Array_f8_1d: ...
def kaiser_bessel_derived(M: _WindowLength, beta: AnyReal, *, sym: op.CanBool = True) -> _Array_f8_1d: ...
def gaussian(M: _WindowLength, std: AnyReal, sym: op.CanBool = True) -> _Array_f8_1d: ...
def general_gaussian(M: _WindowLength, p: AnyReal, sig: AnyReal, sym: op.CanBool = True) -> _Array_f8_1d: ...
def chebwin(M: _WindowLength, at: AnyReal, sym: op.CanBool = True) -> _Array_f8_1d: ...
def cosine(M: _WindowLength, sym: op.CanBool = True) -> _Array_f8_1d: ...
def exponential(M: _WindowLength, center: AnyReal | None = None, tau: AnyReal = 1.0, sym: op.CanBool = True) -> _Array_f8_1d: ...
def taylor(
    M: _WindowLength, nbar: AnyInt = 4, sll: AnyInt = 30, norm: op.CanBool = True, sym: op.CanBool = True
) -> _Array_f8_1d: ...
def lanczos(M: _WindowLength, *, sym: op.CanBool = True) -> _Array_f8_1d: ...

# Overloads where `return_ratios` is `False`.
@overload
def dpss(
    M: _WindowLength,
    NW: AnyReal,
    Kmax: op.CanIndex,
    sym: op.CanBool = True,
    norm: _Norm | None = None,
    return_ratios: Literal[False] = False,
) -> _Array_f8_2d: ...
@overload
def dpss(
    M: _WindowLength,
    NW: AnyReal,
    Kmax: None = None,
    sym: op.CanBool = True,
    norm: _Norm | None = None,
    return_ratios: Literal[False] = False,
) -> _Array_f8_1d: ...

# Overloads where `return_ratios` is `True`.
# `return_ratios` as a positional argument
@overload
def dpss(
    M: _WindowLength,
    NW: AnyReal,
    Kmax: op.CanIndex,
    sym: op.CanBool,
    norm: _Norm | None,
    return_ratios: Literal[True],
) -> tuple[_Array_f8_2d, _Array_f8_1d]: ...

# `return_ratios` as a keyword argument
@overload
def dpss(
    M: _WindowLength,
    NW: AnyReal,
    Kmax: op.CanIndex,
    sym: op.CanBool = True,
    norm: _Norm | None = None,
    *,
    return_ratios: Literal[True],
) -> tuple[_Array_f8_2d, _Array_f8_1d]: ...

# `return_ratios` as a positional argument
@overload
def dpss(
    M: _WindowLength,
    NW: AnyReal,
    Kmax: None,
    sym: op.CanBool,
    norm: _Norm | None,
    return_ratios: Literal[True],
) -> tuple[_Array_f8_1d, np.float64]: ...

# `return_ratios` as a keyword argument
@overload
def dpss(
    M: _WindowLength,
    NW: AnyReal,
    Kmax: None = None,
    sym: op.CanBool = True,
    norm: _Norm | None = None,
    *,
    return_ratios: Literal[True],
) -> tuple[_Array_f8_1d, np.float64]: ...

#
def get_window(
    window: _Window | AnyReal | tuple[_Window | _WindowNeedsParams, Unpack[tuple[object, ...]]],
    Nx: _WindowLength,
    fftbins: op.CanBool = True,
) -> _Array_f8_1d: ...
