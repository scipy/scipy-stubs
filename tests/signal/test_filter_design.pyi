# type-tests for `signal/_max_len_seq.pyi`

from typing import Any, Literal, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.signal import (
    BadCoefficients,
    band_stop_obj,
    bessel,
    besselap,
    bilinear,
    bilinear_zpk,
    buttap,
    butter,
    buttord,
    cheb1ap,
    cheb1ord,
    cheb2ap,
    cheb2ord,
    cheby1,
    cheby2,
    ellip,
    ellipap,
    ellipord,
    findfreqs,
    freqs,
    freqs_zpk,
    freqz,
    freqz_sos,
    freqz_zpk,
    gammatone,
    group_delay,
    iircomb,
    iirdesign,
    iirfilter,
    iirnotch,
    iirpeak,
    lp2bp,
    lp2bp_zpk,
    lp2bs,
    lp2bs_zpk,
    lp2hp,
    lp2hp_zpk,
    lp2lp,
    lp2lp_zpk,
    normalize,
    sos2tf,
    sos2zpk,
    sosfreqz,
    tf2sos,
    tf2zpk,
    zpk2sos,
    zpk2tf,
)

###

_i64_1d: onp.Array1D[np.int64]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f80_1d: onp.Array1D[npc.floating80]
_c64_1d: onp.Array1D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]
_c160_1d: onp.Array1D[npc.complexfloating160]

_i64_2d: onp.Array2D[np.int64]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]

###

# BadCoefficients
assert_type(BadCoefficients(), BadCoefficients)

_bad_coefficients_cls: type[UserWarning] = BadCoefficients
_bad_coefficients: UserWarning = BadCoefficients()

# findfreqs
assert_type(findfreqs(_f64_1d, _f64_1d, 10), onp.Array1D[np.float64])

# freqs
assert_type(freqs(_f64_1d, _f64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(freqs(_f64_1d, _f64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(freqs(_f64_1d, _f64_1d, _f64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(freqs(_f64_1d, _f64_1d, _c128_1d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])

# freqs_zpk
assert_type(freqs_zpk(_f64_1d, _f64_1d, _f64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(freqs_zpk(_f64_1d, _f64_1d, _f64_1d, _f64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(freqs_zpk(_f64_1d, _f64_1d, _f64_1d, _c128_1d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])

# freqz
assert_type(freqz(_f64_1d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.complex128]])
assert_type(freqz(_f64_1d, _f64_1d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.complex128]])
assert_type(freqz(_f64_1d, _f64_1d, _f64_1d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.complex128]])
assert_type(freqz(_f64_1d, _f64_1d, _c128_1d), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])
assert_type(freqz(_f64_1d, worN=_c128_1d), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])

# freqz_zpk
assert_type(freqz_zpk(_f64_1d, _f64_1d, _f64_1d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.complex128]])
assert_type(freqz_zpk(_f64_1d, _f64_1d, _f64_1d, _f64_1d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.complex128]])
assert_type(freqz_zpk(_f64_1d, _f64_1d, _f64_1d, _c128_1d), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])

# group_delay
assert_type(group_delay((_f64_1d, _f64_1d)), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(group_delay((_f64_1d, _f64_1d), _f64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(group_delay((_f64_1d, _f64_1d), _c128_1d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.float64]])

# freqz_sos
assert_type(freqz_sos(_f64_2d), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(freqz_sos(_f64_2d, _f64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(freqz_sos(_f64_2d, _c128_1d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])

# sosfreqz
assert_type(sosfreqz(_f64_2d), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(sosfreqz(_f64_2d, _f64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(sosfreqz(_f64_2d, _c128_1d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])

# tf2zpk
assert_type(tf2zpk(_i64_1d, _i64_1d), tuple[onp.Array1D[np.float64 | np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(tf2zpk(_f32_1d, _f32_1d), tuple[onp.Array1D[np.float32 | np.complex64], onp.Array1D[np.complex64], np.float32])
assert_type(tf2zpk(_f32_1d, _c64_1d), tuple[onp.Array1D[np.complex64], onp.Array1D[np.complex64], np.float32])
assert_type(tf2zpk(_c64_1d, _f32_1d), tuple[onp.Array1D[np.complex64], onp.Array1D[np.complex64], np.float32])
assert_type(tf2zpk(_c64_1d, _c64_1d), tuple[onp.Array1D[np.complex64], onp.Array1D[np.complex64], np.float32])
assert_type(tf2zpk(_f32_1d, _f64_1d), tuple[onp.Array1D[np.float64 | np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(tf2zpk(_f64_1d, _f32_1d), tuple[onp.Array1D[np.float64 | np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(tf2zpk(_f64_1d, _f64_1d), tuple[onp.Array1D[np.float64 | np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(tf2zpk(_f64_1d, _c128_1d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(tf2zpk(_c128_1d, _f64_1d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(tf2zpk(_c128_1d, _c128_1d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])

# tf2sos
assert_type(tf2sos(_f64_1d, _f64_1d), onp.Array2D[np.float64])

# zpk2tf
assert_type(zpk2tf(_f64_1d, _f64_1d, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])

# zpk2sos
assert_type(zpk2sos(_f64_1d, _f64_1d, 1.0), onp.Array2D[np.float64])

# normalize
assert_type(normalize(_i64_1d, _i64_1d), tuple[onp.ArrayND[np.float64], onp.Array1D[np.float64]])
assert_type(normalize(_f32_1d, _f32_1d), tuple[onp.ArrayND[np.float32], onp.Array1D[np.float32]])
assert_type(normalize(_c64_1d, _c64_1d), tuple[onp.ArrayND[np.complex64], onp.Array1D[np.complex64]])
assert_type(normalize(_f64_1d, _f64_1d), tuple[onp.ArrayND[np.float64], onp.Array1D[np.float64]])
assert_type(normalize(_f64_1d, _c128_1d), tuple[onp.ArrayND[np.complex128], onp.Array1D[np.complex128]])
assert_type(normalize(_c128_1d, _f64_1d), tuple[onp.ArrayND[np.complex128], onp.Array1D[np.complex128]])
assert_type(normalize(_c128_1d, _c128_1d), tuple[onp.ArrayND[np.complex128], onp.Array1D[np.complex128]])

# sos2tf
assert_type(sos2tf(_i64_2d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(sos2tf(_f32_2d), tuple[onp.Array1D[np.float32], onp.Array1D[np.float32]])
assert_type(sos2tf(_f64_2d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(sos2tf(_c64_2d), tuple[onp.Array1D[np.complex64], onp.Array1D[np.complex64]])
assert_type(sos2tf(_c128_2d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])

# sos2zpk
assert_type(sos2zpk(_i64_2d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(sos2zpk(_f32_2d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float32])
assert_type(sos2zpk(_f64_2d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(sos2zpk(_c64_2d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.complex64])
assert_type(sos2zpk(_c128_2d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.complex128])

# lp2lp
assert_type(lp2lp(_i64_1d, _i64_1d), tuple[onp.ArrayND[np.float64], onp.Array1D[np.float64]])
assert_type(lp2lp(_f32_1d, _f32_1d), tuple[onp.ArrayND[np.float64], onp.Array1D[np.float64]])
assert_type(lp2lp(_f64_1d, _f64_1d), tuple[onp.ArrayND[np.float64], onp.Array1D[np.float64]])
assert_type(lp2lp(_f80_1d, _f80_1d), tuple[onp.ArrayND[np.longdouble], onp.Array1D[np.longdouble]])
assert_type(lp2lp(_c64_1d, _c64_1d), tuple[onp.ArrayND[np.complex128], onp.Array1D[np.complex128]])
assert_type(lp2lp(_c128_1d, _c128_1d), tuple[onp.ArrayND[np.complex128], onp.Array1D[np.complex128]])
assert_type(lp2lp(_c160_1d, _c160_1d), tuple[onp.ArrayND[np.clongdouble], onp.Array1D[np.clongdouble]])

# lp2hp
assert_type(lp2hp(_f64_1d, _f64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])

# lp2bp
assert_type(lp2bp(_f64_1d, _f64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])

# lp2bs
assert_type(lp2bs(_f64_1d, _f64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])

# lp2lp_zpk
assert_type(lp2lp_zpk(_f64_1d, _f64_1d, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], float])
assert_type(lp2lp_zpk(_f64_1d, _c128_1d, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128], float])
assert_type(lp2lp_zpk(_c128_1d, _f64_1d, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.float64], float])
assert_type(lp2lp_zpk(_c128_1d, _c128_1d, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], float])

# lp2hp_zpk
assert_type(lp2hp_zpk(_f64_1d, _f64_1d, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(lp2hp_zpk(_f64_1d, _c128_1d, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128], np.float64])
assert_type(lp2hp_zpk(_c128_1d, _f64_1d, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.float64], np.float64])
assert_type(lp2hp_zpk(_c128_1d, _c128_1d, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])

# lp2bp_zpk
assert_type(lp2bp_zpk(_f64_1d, _f64_1d, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], float])
assert_type(lp2bp_zpk(_f64_1d, _c128_1d, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], float])
assert_type(lp2bp_zpk(_c128_1d, _f64_1d, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], float])
assert_type(lp2bp_zpk(_c128_1d, _c128_1d, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], float])

# lp2bs_zpk
assert_type(lp2bs_zpk(_f64_1d, _f64_1d, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(lp2bs_zpk(_f64_1d, _c128_1d, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(lp2bs_zpk(_c128_1d, _f64_1d, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(lp2bs_zpk(_c128_1d, _c128_1d, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])

# bilinear
assert_type(bilinear(_f64_1d, _f64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])

# bilinear_zpk
assert_type(bilinear_zpk(_f64_1d, _f64_1d, 1.0, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], np.float64])
assert_type(bilinear_zpk(_f64_1d, _c128_1d, 1.0, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128], np.float64])
assert_type(bilinear_zpk(_c128_1d, _f64_1d, 1.0, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.float64], np.float64])
assert_type(bilinear_zpk(_c128_1d, _c128_1d, 1.0, 1.0), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])

# iirdesign
assert_type(iirdesign(0.2, 0.3, 1, 40), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(iirdesign(0.2, 0.3, 1, 40, output="ba"), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(iirdesign(0.2, 0.3, 1, 40, output="zpk"), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(iirdesign(0.2, 0.3, 1, 40, output="sos"), onp.Array2D[np.float64])

# iirfilter
assert_type(iirfilter(8, 0.1), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(iirfilter(8, 0.1, output="ba"), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(iirfilter(8, 0.1, output="zpk"), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(iirfilter(8, 0.1, output="sos"), onp.Array2D[np.float64])

# butter
assert_type(butter(8, 0.1), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(butter(8, 0.1, output="ba"), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(butter(8, 0.1, output="zpk"), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128], float])
assert_type(butter(8, 0.1, output="sos"), onp.Array2D[np.float64])

# cheby1
assert_type(cheby1(8, 3, 0.1), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(cheby1(8, 3, 0.1, output="ba"), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(cheby1(8, 3, 0.1, output="zpk"), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(cheby1(8, 3, 0.1, output="sos"), onp.Array2D[np.float64])

# cheby2
assert_type(cheby2(8, 3, 0.1), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(cheby2(8, 3, 0.1, output="ba"), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(cheby2(8, 3, 0.1, output="zpk"), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(cheby2(8, 3, 0.1, output="sos"), onp.Array2D[np.float64])

# ellip
assert_type(ellip(8, 5, 40, 100), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(ellip(8, 5, 40, 100, output="ba"), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(ellip(8, 5, 40, 100, output="zpk"), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], np.float64])
assert_type(ellip(8, 5, 40, 100, output="sos"), onp.Array2D[np.float64])

# bessel
assert_type(bessel(3, 10), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(bessel(3, 10, output="ba"), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(bessel(3, 10, output="zpk"), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128], np.float64])
assert_type(bessel(3, 10, output="sos"), onp.Array2D[np.float64])

# band_stop_obj
assert_type(band_stop_obj(2, 1, _f64_1d, _f64_1d, 3, 30, "butter"), np.float64)
assert_type(band_stop_obj(2, 1, _f80_1d, _f80_1d, 3, 30, "butter"), np.longdouble | Any)

# buttord
assert_type(buttord(0.2, 0.3, 3, 40), tuple[int, np.float64])
assert_type(buttord(0.2, _f64_1d, 3, 40), tuple[int, np.float64])
assert_type(buttord(0.2, _f80_1d, 3, 40), tuple[int, np.longdouble])
assert_type(buttord(_f64_1d, 0.3, 3, 40), tuple[int, onp.Array1D[np.float64]])

# cheb1ord
assert_type(cheb1ord(0.2, 0.3, 3, 40), tuple[int, np.float64])
assert_type(cheb1ord(0.2, _f64_1d, 3, 40), tuple[int, np.float64])
assert_type(cheb1ord(0.2, _f80_1d, 3, 40), tuple[int, np.longdouble])
assert_type(cheb1ord(_f64_1d, 0.3, 3, 40), tuple[int, onp.Array1D[np.float64]])

# cheb2ord
assert_type(cheb2ord(0.2, 0.3, 3, 40), tuple[int, np.float64])
assert_type(cheb2ord(0.2, _f64_1d, 3, 40), tuple[int, np.float64])
assert_type(cheb2ord(0.2, _f80_1d, 3, 40), tuple[int, np.longdouble])
assert_type(cheb2ord(_f64_1d, 0.3, 3, 40), tuple[int, onp.Array1D[np.float64]])

# ellipord
assert_type(ellipord(0.2, 0.3, 3, 40), tuple[int, np.float64])
assert_type(ellipord(0.2, _f64_1d, 3, 40), tuple[int, np.float64])
assert_type(ellipord(_f64_1d, 0.3, 3, 40), tuple[int, onp.Array1D[np.float64]])

# buttap
assert_type(buttap(4), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128], Literal[1]])
assert_type(buttap(4, xp=np), tuple[Any, Any, Literal[1]])

# cheb1ap
assert_type(cheb1ap(4, 0.1), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128], float])
assert_type(cheb1ap(4, 0.1, xp=np), tuple[Any, Any, float])

# cheb2ap
assert_type(cheb2ap(4, 0.1), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], float])
assert_type(cheb2ap(4, 0.1, xp=np), tuple[Any, Any, float])

# ellipap
assert_type(ellipap(4, 0.1, 0.2), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], float])
assert_type(ellipap(4, 0.1, 0.2, xp=np), tuple[Any, Any, float])

# besselap
assert_type(besselap(4), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128], float])
assert_type(besselap(4, xp=np), tuple[Any, Any, float])

# iirnotch
assert_type(iirnotch(60.0, 30.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(iirnotch(60.0, 30.0, xp=np), tuple[Any, Any])

# iirpeak
assert_type(iirpeak(60.0, 30.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(iirpeak(60.0, 30.0, xp=np), tuple[Any, Any])

# iircomb
assert_type(iircomb(60.0, 30.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(iircomb(60.0, 30.0, xp=np), tuple[Any, Any])

# gammatone
assert_type(gammatone(1000.0, "iir"), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(gammatone(1000.0, "fir"), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(gammatone(1000.0, "iir", xp=np), tuple[Any, Any])
assert_type(gammatone(1000.0, "fir", xp=np), tuple[Any, Any])
