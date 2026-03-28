from collections.abc import Callable
from types import ModuleType
from typing import Any, TypeAlias, overload
from typing_extensions import TypeAliasType, TypeVar

import numpy as np

###

_: TypeAlias = object  # ignored parameters

_WindowSpec: TypeAlias = str | tuple[object, ...] | Callable[..., object]

_T = TypeVar("_T")
_TupleT = TypeVar("_TupleT", bound=tuple[object, ...])
_ModuleT = TypeVar("_ModuleT", bound=ModuleType)

# these pretend that we can a module (`numpy` or `array_api_compat.np_compat`) in Python's type system
_NumpyModule: TypeAlias = ModuleType

# this pretends that we can express `array_namespace` in Python's type system
_ArrayT = TypeVar("_ArrayT")
_ArrayNamespace = TypeAliasType("_ArrayNamespace", ModuleType, type_params=(_ArrayT,))

###

@overload
def _skip_if_lti(arg: _TupleT) -> _TupleT: ...
@overload
def _skip_if_lti(arg: _T) -> tuple[_T] | Any: ...

#
@overload
def _skip_if_str_or_tuple(window: _WindowSpec) -> None: ...
@overload
def _skip_if_str_or_tuple(window: _T) -> _T: ...

#
@overload
def _skip_if_poly1d(arg: np.poly1d) -> None: ...
@overload
def _skip_if_poly1d(arg: _T) -> _T: ...

###

def abcd_normalize_signature(
    A: _ArrayT | None = None, B: _ArrayT | None = None, C: _ArrayT | None = None, D: _ArrayT | None = None
) -> _ArrayNamespace[_ArrayT]: ...

#
def argrelextrema_signature(data: _ArrayT, *args: _, **kwargs: _) -> _ArrayNamespace[_ArrayT]: ...

argrelmax_signature = argrelextrema_signature
argrelmin_signature = argrelextrema_signature

def band_stop_obj_signature(
    wp: _, ind: _, passb: _ArrayT, stopb: _ArrayT, gpass: _, gstop: _, type: _
) -> _ArrayNamespace[_ArrayT]: ...

#
def bessel_signature(N: _, Wn: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

butter_signature = bessel_signature

def cheby2_signature(N: _, rs: _, Wn: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...
def cheby1_signature(N: _, rp: _, Wn: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...
def ellip_signature(N: _, rp: _, rs: _, Wn: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

###

@overload
def besselap_signature(N: _, norm: _ = "phase", *, xp: None = None, device: None = None) -> _NumpyModule: ...
@overload
def besselap_signature(N: _, norm: _ = "phase", *, xp: _ModuleT, device: _ = None) -> _ModuleT: ...

#
@overload
def buttap_signature(N: _, *, xp: None = None, device: None = None) -> _NumpyModule: ...
@overload
def buttap_signature(N: _, *, xp: _ModuleT, device: _ = None) -> _ModuleT: ...

#
@overload
def cheb1ap_signature(N: _, rp: _, *, xp: None = None, device: None = None) -> _NumpyModule: ...
@overload
def cheb1ap_signature(N: _, rp: _, *, xp: _ModuleT, device: _ = None) -> _ModuleT: ...

#
@overload
def cheb2ap_signature(N: _, rs: _, *, xp: None = None, device: None = None) -> _NumpyModule: ...
@overload
def cheb2ap_signature(N: _, rs: _, *, xp: _ModuleT, device: _ = None) -> _ModuleT: ...

#
@overload
def ellipap_signature(N: _, rp: _, rs: _, *, xp: None = None, device: None = None) -> _NumpyModule: ...
@overload
def ellipap_signature(N: _, rp: _, rs: _, *, xp: _ModuleT, device: _ = None) -> _ModuleT: ...

#
def correlation_lags_signature(in1_len: _, in2_len: _, mode: _ = "full") -> _NumpyModule: ...

#
def czt_points_signature(m: _, w: _ = None, a: _ = 1 + 0j) -> _NumpyModule: ...

#
@overload
def gammatone_signature(
    freq: _, ftype: _, order: _ = None, numtaps: _ = None, fs: _ = None, *, xp: None = None, device: None = None
) -> _NumpyModule: ...
@overload
def gammatone_signature(
    freq: _, ftype: _, order: _ = None, numtaps: _ = None, fs: _ = None, *, xp: _ModuleT, device: _ = None
) -> _ModuleT: ...

#
@overload
def iircomb_signature(
    w0: _, Q: _, ftype: _ = "notch", fs: _ = 2.0, *, pass_zero: _ = False, xp: None = None, device: None = None
) -> _NumpyModule: ...
@overload
def iircomb_signature(
    w0: _, Q: _, ftype: _ = "notch", fs: _ = 2.0, *, pass_zero: _ = False, xp: _ModuleT, device: _ = None
) -> _ModuleT: ...

#
@overload
def iirnotch_signature(w0: _, Q: _, fs: _ = 2.0, *, xp: None = None, device: None = None) -> _NumpyModule: ...
@overload
def iirnotch_signature(w0: _, Q: _, fs: _ = 2.0, *, xp: _ModuleT, device: _ = None) -> _ModuleT: ...

iirpeak_signature = iirnotch_signature

@overload
def savgol_coeffs_signature(
    window_length: _,
    polyorder: _,
    deriv: _ = 0,
    delta: _ = 1.0,
    pos: _ = None,
    use: _ = "conv",
    *,
    xp: None = None,
    device: None = None,
) -> _NumpyModule: ...
@overload
def savgol_coeffs_signature(
    window_length: _,
    polyorder: _,
    deriv: _ = 0,
    delta: _ = 1.0,
    pos: _ = None,
    use: _ = "conv",
    *,
    xp: _ModuleT,
    device: _ = None,
) -> _ModuleT: ...

#
def unit_impulse_signature(shape: _, idx: _ = None, dtype: _ = float) -> _NumpyModule: ...  # noqa: PYI011

#
def buttord_signature(
    wp: _ArrayT, ws: _ArrayT, gpass: _, gstop: _, analog: _ = False, fs: _ = None
) -> _ArrayNamespace[_ArrayT]: ...

cheb1ord_signature = buttord_signature
cheb2ord_signature = buttord_signature
ellipord_signature = buttord_signature

#
def kaiser_atten_signature(numtaps: _, width: _) -> _NumpyModule: ...
def kaiser_beta_signature(a: _) -> _NumpyModule: ...
def kaiserord_signature(ripple: _, width: _) -> _NumpyModule: ...

#
@overload
def get_window_signature(window: _, Nx: _, fftbins: _ = True, *, xp: None = None, device: None = None) -> _NumpyModule: ...
@overload
def get_window_signature(window: _, Nx: _, fftbins: _ = True, *, xp: _ModuleT, device: _ = None) -> _ModuleT: ...

#
def bode_signature(system: _ArrayT | tuple[_ArrayT, ...], w: _ArrayT | None = None, n: _ = 100) -> _ArrayNamespace[_ArrayT]: ...

dbode_signature = bode_signature

def freqresp_signature(
    system: _ArrayT | tuple[_ArrayT, ...], w: _ArrayT | None = None, n: _ = 10_000
) -> _ArrayNamespace[_ArrayT]: ...

dfreqresp_signature = freqresp_signature

def impulse_signature(
    system: _ArrayT | tuple[_ArrayT, ...], X0: _ArrayT | None = None, T: _ArrayT | None = None, N: _ = None
) -> _ArrayNamespace[_ArrayT]: ...

#
def dimpulse_signature(
    system: _ArrayT | tuple[_ArrayT, ...], x0: _ArrayT | None = None, t: _ArrayT | None = None, n: _ = None
) -> _ArrayNamespace[_ArrayT]: ...

#
def lsim_signature(
    system: _ArrayT | tuple[_ArrayT, ...], U: _ArrayT, T: _ArrayT, X0: _ArrayT | None = None, interp: _ = True
) -> _ArrayNamespace[_ArrayT]: ...

#
def dlsim_signature(
    system: _ArrayT | tuple[_ArrayT, ...], u: _ArrayT, t: _ArrayT | None = None, x0: _ArrayT | None = None
) -> _ArrayNamespace[_ArrayT]: ...

#
def step_signature(
    system: _ArrayT | tuple[_ArrayT, ...], X0: _ArrayT | None = None, T: _ArrayT | None = None, N: _ = None
) -> _ArrayNamespace[_ArrayT]: ...

#
def dstep_signature(
    system: _ArrayT | tuple[_ArrayT, ...], x0: _ArrayT | None = None, t: _ArrayT | None = None, n: _ = None
) -> _ArrayNamespace[_ArrayT]: ...

#
def cont2discrete_signature(
    system: _ArrayT | tuple[_ArrayT, ...], dt: _, method: _ = "zoh", alpha: _ = None
) -> _ArrayNamespace[_ArrayT]: ...

#
def bilinear_signature(b: _ArrayT, a: _ArrayT, fs: _ = 1.0) -> _ArrayNamespace[_ArrayT]: ...
def bilinear_zpk_signature(z: _ArrayT, p: _ArrayT, k: _, fs: _) -> _ArrayNamespace[_ArrayT]: ...

#
def chirp_signature(t: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...
def choose_conv_method_signature(in1: _ArrayT, in2: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...
def convolve_signature(in1: _ArrayT, in2: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

fftconvolve_signature = convolve_signature
oaconvolve_signature = convolve_signature
correlate_signature = convolve_signature
correlate_signature = convolve_signature
convolve2d_signature = convolve_signature
correlate2d_signature = convolve_signature

def coherence_signature(
    x: _ArrayT, y: _ArrayT, fs: _ = 1.0, window: _ArrayT | _WindowSpec = "hann_periodic", *args: _, **kwds: _
) -> _ArrayNamespace[_ArrayT]: ...

#
def csd_signature(
    x: _ArrayT, y: _ArrayT, fs: _ = 1.0, window: _ArrayT | _WindowSpec = "hann_periodic", *args: _, **kwds: _
) -> _ArrayNamespace[_ArrayT]: ...

#
def periodogram_signature(
    x: _ArrayT, fs: _ = 1.0, window: _ArrayT | _WindowSpec = "boxcar", *args: _, **kwds: _
) -> _ArrayNamespace[_ArrayT]: ...

#
def welch_signature(
    x: _ArrayT, fs: _ = 1.0, window: _ArrayT | _WindowSpec = "hann_periodic", *args: _, **kwds: _
) -> _ArrayNamespace[_ArrayT]: ...

#
def spectrogram_signature(
    x: _ArrayT, fs: _ = 1.0, window: _ArrayT | _WindowSpec = ("tukey_periodic", 0.25), *args: _, **kwds: _
) -> _ArrayNamespace[_ArrayT]: ...

#
def stft_signature(
    x: _ArrayT, fs: _ = 1.0, window: _ArrayT | _WindowSpec = "hann_periodic", *args: _, **kwds: _
) -> _ArrayNamespace[_ArrayT]: ...

#
def istft_signature(
    Zxx: _ArrayT, fs: _ = 1.0, window: _ArrayT | _WindowSpec = "hann_periodic", *args: _, **kwds: _
) -> _ArrayNamespace[_ArrayT]: ...

#
def resample_signature(
    x: _ArrayT, num: _, t: _ArrayT | None = None, axis: _ = 0, window: _ArrayT | _WindowSpec | None = None, domain: _ = "time"
) -> _ArrayNamespace[_ArrayT]: ...

#
def resample_poly_signature(
    x: _ArrayT, up: _, down: _, axis: _ = 0, window: _ArrayT | _WindowSpec = ("kaiser", 5.0), *args: _, **kwds: _
) -> _ArrayNamespace[_ArrayT]: ...

#
def check_COLA_signature(window: _ArrayT | _WindowSpec, nperseg: _, noverlap: _, tol: _ = 1e-10) -> _ArrayNamespace[_ArrayT]: ...

#
def check_NOLA_signature(window: _ArrayT | _WindowSpec, nperseg: _, noverlap: _, tol: _ = 1e-10) -> _ArrayNamespace[_ArrayT]: ...

#
def czt_signature(x: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

decimate_signature = czt_signature
gauss_spline_signature = czt_signature

def deconvolve_signature(signal: _ArrayT, divisor: _ArrayT) -> _ArrayNamespace[_ArrayT]: ...

#
def detrend_signature(
    data: _ArrayT, axis: _ = 1, type: _ = "linear", bp: _ArrayT | _ = 0, *args: _, **kwds: _
) -> _ArrayNamespace[_ArrayT]: ...

#
def filtfilt_signature(b: _ArrayT, a: _ArrayT, x: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

#
def lfilter_signature(
    b: _ArrayT, a: _ArrayT, x: _ArrayT, axis: _ = -1, zi: _ArrayT | None = None
) -> _ArrayNamespace[_ArrayT]: ...

#
def envelope_signature(z: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

#
def find_peaks_signature(
    x: _,
    height: _ = None,
    threshold: _ = None,
    distance: _ = None,
    prominence: _ = None,
    width: _ = None,
    wlen: _ = None,
    rel_height: _ = 0.5,
    plateau_size: _ = None,
) -> _NumpyModule: ...  # np_compat

#
def find_peaks_cwt_signature(
    vector: _ArrayT, widths: _ArrayT, wavelet: _ = None, max_distances: _ArrayT | None = None, *args: _, **kwds: _
) -> _ArrayNamespace[_ArrayT]: ...

#
def findfreqs_signature(num: _ArrayT, den: _ArrayT, N: _, kind: _ = "ba") -> _ArrayNamespace[_ArrayT]: ...

#
def firls_signature(
    numtaps: _, bands: _ArrayT, desired: _ArrayT, *, weight: _ArrayT | None = None, fs: _ = None
) -> _ArrayNamespace[_ArrayT]: ...

#
@overload
def firwin_signature(numtaps: _, cutoff: float, *args: _, **kwds: _) -> _NumpyModule: ...  # np_compat
@overload
def firwin_signature(numtaps: _, cutoff: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

#
def firwin2_signature(numtaps: _, freq: _ArrayT, gain: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

#
def freqs_zpk_signature(
    z: _ArrayT, p: _ArrayT, k: _, worN: _ArrayT | int = 200, *args: _, **kwds: _
) -> _ArrayNamespace[_ArrayT]: ...

freqz_zpk_signature = freqs_zpk_signature

def freqs_signature(b: _ArrayT, a: _ArrayT, worN: _ArrayT | int = 200, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...
def freqz_signature(
    b: _ArrayT, a: _ArrayT | int = 1, worN: _ArrayT | int = 512, *args: _, **kwds: _
) -> _ArrayNamespace[_ArrayT]: ...
def freqz_sos_signature(sos: _ArrayT, worN: _ArrayT | int = 512, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

sosfreqz_signature = freqz_sos_signature

@overload
def gausspulse_signature(t: str, *args: _, **kwds: _) -> _NumpyModule: ...
@overload
def gausspulse_signature(t: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

#
def group_delay_signature(
    system: tuple[_ArrayT, _ArrayT], w: _ArrayT | _ = 512, whole: _ = False, fs: _ = 6.283185307179586
) -> _ArrayNamespace[_ArrayT]: ...

#
def hilbert_signature(x: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

hilbert2_signature = hilbert_signature

def iirdesign_signature(wp: _ArrayT, ws: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...
def iirfilter_signature(N: _, Wn: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...
def invres_signature(r: _ArrayT, p: _ArrayT, k: _ArrayT, tol: _ = 0.001, rtype: _ = "avg") -> _ArrayNamespace[_ArrayT]: ...

invresz_signature = invres_signature

def lfilter_zi_signature(b: _ArrayT, a: _ArrayT) -> _ArrayNamespace[_ArrayT]: ...
def sosfilt_zi_signature(sos: _ArrayT) -> _ArrayNamespace[_ArrayT]: ...

#
def remez_signature(
    numtaps: _, bands: _ArrayT, desired: _ArrayT, *, weight: _ArrayT | None = None, **kwds: _
) -> _ArrayNamespace[_ArrayT]: ...

#
def lfiltic_signature(b: _ArrayT, a: _ArrayT, y: _ArrayT, x: _ArrayT | None = None) -> _ArrayNamespace[_ArrayT]: ...

#
def lombscargle_signature(
    x: _ArrayT,
    y: _ArrayT,
    freqs: _ArrayT,
    precenter: _ = False,
    normalize: _ = False,
    *,
    weights: _ArrayT | None = None,
    floating_mean: _ = False,
) -> _ArrayNamespace[_ArrayT]: ...

#
def lp2bp_signature(b: _ArrayT, a: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

lp2bs_signature = lp2bp_signature
lp2hp_signature = lp2bp_signature
lp2lp_signature = lp2bp_signature

tf2zpk_signature = lp2bp_signature
tf2sos_signature = lp2bp_signature

normalize_signature = lp2bp_signature
residue_signature = lp2bp_signature
residuez_signature = residue_signature

def lp2bp_zpk_signature(z: _ArrayT, p: _ArrayT, k: _, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

lp2bs_zpk_signature = lp2bp_zpk_signature
lp2hp_zpk_signature = lp2bs_zpk_signature
lp2lp_zpk_signature = lp2bs_zpk_signature

def zpk2sos_signature(z: _ArrayT, p: _ArrayT, k: _, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

zpk2ss_signature = zpk2sos_signature
zpk2tf_signature = zpk2sos_signature

def max_len_seq_signature(
    nbits: _, state: _ArrayT | None = None, length: _ = None, taps: _ArrayT | None = None
) -> _ArrayNamespace[_ArrayT]: ...

#
def medfilt_signature(volume: _ArrayT, kernel_size: _ = None) -> _ArrayNamespace[_ArrayT]: ...
def medfilt2d_signature(input: _ArrayT, kernel_size: _ = 3) -> _ArrayNamespace[_ArrayT]: ...

#
def minimum_phase_signature(h: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

#
def order_filter_signature(a: _ArrayT, domain: _ArrayT, rank: _) -> _ArrayNamespace[_ArrayT]: ...

#
def peak_prominences_signature(x: _ArrayT, peaks: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

peak_widths_signature = peak_prominences_signature

def place_poles_signature(
    A: _ArrayT, B: _ArrayT, poles: _ArrayT, method: _ = "YT", rtol: _ = 0.001, maxiter: _ = 30
) -> _ArrayNamespace[_ArrayT]: ...

#
def savgol_filter_signature(x: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...
def sawtooth_signature(t: _ArrayT, width: _ = 1) -> _ArrayNamespace[_ArrayT]: ...
def sepfir2d_signature(input: _ArrayT, hrow: _ArrayT, hcol: _ArrayT) -> _ArrayNamespace[_ArrayT]: ...
def sos2tf_signature(sos: _ArrayT) -> _ArrayNamespace[_ArrayT]: ...

sos2zpk_signature = sos2tf_signature

def sosfilt_signature(sos: _ArrayT, x: _ArrayT, axis: _ = -1, zi: _ArrayT | None = None) -> _ArrayNamespace[_ArrayT]: ...
def sosfiltfilt_signature(sos: _ArrayT, x: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...
def spline_filter_signature(Iin: _ArrayT, lmbda: _ = 5.0) -> _ArrayNamespace[_ArrayT]: ...
def square_signature(t: _ArrayT, duty: _ = 0.5) -> _ArrayNamespace[_ArrayT]: ...
def ss2tf_signature(A: _ArrayT, B: _ArrayT, C: _ArrayT, D: _ArrayT, input: _ = 0) -> _ArrayNamespace[_ArrayT]: ...

ss2zpk_signature = ss2tf_signature

def sweep_poly_signature(t: _ArrayT, poly: _ArrayT | np.poly1d, phi: _ = 0) -> _ArrayNamespace[_ArrayT]: ...

#
def symiirorder1_signature(signal: _ArrayT, c0: _, z1: _, precision: _ = -1.0) -> _ArrayNamespace[_ArrayT]: ...
def symiirorder2_signature(input: _ArrayT, r: _, omega: _, precision: _ = -1.0) -> _ArrayNamespace[_ArrayT]: ...
def cspline1d_signature(signal: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

qspline1d_signature = cspline1d_signature
cspline2d_signature = cspline1d_signature
qspline2d_signature = qspline1d_signature

def cspline1d_eval_signature(cj: _ArrayT, newx: _ArrayT, *args: _, **kwds: _) -> _ArrayNamespace[_ArrayT]: ...

qspline1d_eval_signature = cspline1d_eval_signature

def tf2ss_signature(num: _ArrayT, den: _ArrayT) -> _ArrayNamespace[_ArrayT]: ...
def unique_roots_signature(p: _ArrayT, tol: _ = 0.001, rtype: _ = "min") -> _ArrayNamespace[_ArrayT]: ...
def upfirdn_signature(
    h: _ArrayT, x: _ArrayT, up: _ = 1, down: _ = 1, axis: _ = -1, mode: _ = "constant", cval: _ = 0
) -> _ArrayNamespace[_ArrayT]: ...
def vectorstrength_signature(events: _ArrayT, period: _ArrayT) -> _ArrayNamespace[_ArrayT]: ...
def wiener_signature(im: _ArrayT, mysize: _ = None, noise: _ = None) -> _ArrayNamespace[_ArrayT]: ...
def zoom_fft_signature(
    x: _ArrayT, fn: _ArrayT, m: _ = None, *, fs: _ = 2, endpoint: _ = False, axis: _ = -1
) -> _ArrayNamespace[_ArrayT]: ...
