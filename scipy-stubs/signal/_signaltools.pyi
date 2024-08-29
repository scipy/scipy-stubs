from typing import Literal

import numpy as np
from numpy._typing import ArrayLike

from scipy._typing import Untyped

from scipy import linalg as linalg, ndimage as ndimage
from scipy.spatial import cKDTree as cKDTree
from scipy.special import lambertw as lambertw
from ._arraytools import (
    axis_reverse as axis_reverse,
    axis_slice as axis_slice,
    const_ext as const_ext,
    even_ext as even_ext,
    odd_ext as odd_ext,
)
from ._filter_design import cheby1 as cheby1, zpk2sos as zpk2sos
from ._fir_filter_design import firwin as firwin
from ._ltisys import dlti as dlti
from ._upfirdn import upfirdn as upfirdn
from .windows import get_window as get_window

def correlate(in1, in2, mode: str = "full", method: str = "auto") -> Untyped: ...
def correlation_lags(in1_len, in2_len, mode: str = "full") -> Untyped: ...
def fftconvolve(in1, in2, mode: str = "full", axes: Untyped | None = None) -> Untyped: ...
def oaconvolve(in1, in2, mode: str = "full", axes: Untyped | None = None) -> Untyped: ...
def choose_conv_method(in1, in2, mode: str = "full", measure: bool = False) -> Untyped: ...
def convolve(in1, in2, mode: str = "full", method: str = "auto") -> Untyped: ...
def order_filter(a, domain, rank) -> Untyped: ...
def medfilt(volume, kernel_size: Untyped | None = None) -> Untyped: ...
def wiener(im, mysize: Untyped | None = None, noise: Untyped | None = None) -> Untyped: ...
def convolve2d(in1, in2, mode: str = "full", boundary: str = "fill", fillvalue: int = 0) -> Untyped: ...
def correlate2d(in1, in2, mode: str = "full", boundary: str = "fill", fillvalue: int = 0) -> Untyped: ...
def medfilt2d(input, kernel_size: int = 3) -> Untyped: ...
def lfilter(b, a, x, axis: int = -1, zi: Untyped | None = None) -> Untyped: ...
def lfiltic(b, a, y, x: Untyped | None = None) -> Untyped: ...
def deconvolve(signal, divisor) -> Untyped: ...
def hilbert(x, N: Untyped | None = None, axis: int = -1) -> Untyped: ...
def hilbert2(x, N: Untyped | None = None) -> Untyped: ...
def unique_roots(p, tol: float = 0.001, rtype: str = "min") -> Untyped: ...
def invres(r, p, k, tol: float = 0.001, rtype: str = "avg") -> Untyped: ...
def residue(b, a, tol: float = 0.001, rtype: str = "avg") -> Untyped: ...
def residuez(b, a, tol: float = 0.001, rtype: str = "avg") -> Untyped: ...
def invresz(r, p, k, tol: float = 0.001, rtype: str = "avg") -> Untyped: ...
def resample(x, num, t: Untyped | None = None, axis: int = 0, window: Untyped | None = None, domain: str = "time") -> Untyped: ...
def resample_poly(
    x, up, down, axis: int = 0, window=("kaiser", 5.0), padtype: str = "constant", cval: Untyped | None = None
) -> Untyped: ...
def vectorstrength(events, period) -> Untyped: ...
def detrend(
    data: np.ndarray,
    axis: int = -1,
    type: Literal["linear", "constant"] = "linear",
    bp: ArrayLike | int = 0,
    overwrite_data: bool = False,
) -> np.ndarray: ...
def lfilter_zi(b, a) -> Untyped: ...
def sosfilt_zi(sos) -> Untyped: ...
def filtfilt(
    b,
    a,
    x,
    axis: int = -1,
    padtype: str = "odd",
    padlen: Untyped | None = None,
    method: str = "pad",
    irlen: Untyped | None = None,
) -> Untyped: ...
def sosfilt(sos, x, axis: int = -1, zi: Untyped | None = None) -> Untyped: ...
def sosfiltfilt(sos, x, axis: int = -1, padtype: str = "odd", padlen: Untyped | None = None) -> Untyped: ...
def decimate(x, q, n: Untyped | None = None, ftype: str = "iir", axis: int = -1, zero_phase: bool = True) -> Untyped: ...
