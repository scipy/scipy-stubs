# type-tests for `stats/_stats_py.pyi`
# (functions not already covered by other test files)

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.stats import (
    alexandergovern,
    brunnermunzel,
    chisquare,
    combine_pvalues,
    cumfreq,
    describe,
    energy_distance,
    expectile,
    f_oneway,
    fisher_exact,
    friedmanchisquare,
    gmean,
    gstd,
    hmean,
    iqr,
    jarque_bera,
    kruskal,
    ks_1samp,
    ks_2samp,
    kstest,
    kurtosistest,
    median_abs_deviation,
    normaltest,
    obrientransform,
    percentileofscore,
    pmean,
    pointbiserialr,
    power_divergence,
    quantile_test,
    ranksums,
    relfreq,
    scoreatpercentile,
    siegelslopes,
    sigmaclip,
    skewtest,
    theilslopes,
    tiecorrect,
    tmax,
    tmean,
    tmin,
    trim1,
    trim_mean,
    trimboth,
    tsem,
    tstd,
    ttest_1samp,
    ttest_ind_from_stats,
    tvar,
    wasserstein_distance,
    wasserstein_distance_nd,
    weightedtau,
    zmap,
)
from scipy.stats._stats_mstats_common import SiegelslopesResult, TheilslopesResult
from scipy.stats._stats_py import (
    AlexanderGovernResult,
    BrunnerMunzelResult,
    CumfreqResult,
    DescribeResult,
    F_onewayResult,
    FriedmanchisquareResult,
    KruskalResult,
    KstestResult,
    KurtosistestResult,
    NormaltestResult,
    Power_divergenceResult,
    QuantileTestResult,
    RanksumsResult,
    RelfreqResult,
    SigmaclipResult,
    SignificanceResult,
    SkewtestResult,
    TtestResult,
    Ttest_indResult,
)

###

_bool_1d: onp.Array1D[np.bool_]
_bool_2d: onp.Array2D[np.bool_]
_bool_nd: onp.ArrayND[np.bool_]

_intp_1d: onp.Array1D[np.intp]
_intp_2d: onp.Array2D[np.intp]
_intp_nd: onp.ArrayND[np.intp]

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f32_nd: onp.ArrayND[np.float32]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_c64_1d: onp.Array1D[np.complex64]
_c64_2d: onp.Array2D[np.complex64]
_c64_nd: onp.ArrayND[np.complex64]

_c128_1d: onp.Array1D[np.complex128]
_c128_2d: onp.Array2D[np.complex128]
_c128_nd: onp.ArrayND[np.complex128]

_py_i_1d: list[int]
_py_f_1d: list[float]
_py_c_1d: list[complex]

_py_i_2d: list[list[int]]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]

###

# gmean

assert_type(gmean(_py_i_1d), np.float64)
assert_type(gmean(_py_f_1d), np.float64)
assert_type(gmean(_py_c_1d), np.complex128)
assert_type(gmean(_intp_1d), np.float64)
assert_type(gmean(_f32_1d), np.float32)
assert_type(gmean(_f64_1d), np.float64)
assert_type(gmean(_c64_1d), np.complex64)
assert_type(gmean(_c128_1d), np.complex128)
assert_type(gmean(_py_i_2d), onp.Array1D[np.float64])
assert_type(gmean(_py_f_2d), onp.Array1D[np.float64])
assert_type(gmean(_py_c_2d), onp.Array1D[np.complex128])
assert_type(gmean(_intp_2d), onp.Array1D[np.float64])
assert_type(gmean(_f32_2d), onp.Array1D[np.float32])
assert_type(gmean(_f64_2d), onp.Array1D[np.float64])
assert_type(gmean(_c64_2d), onp.Array1D[np.complex64])
assert_type(gmean(_c128_2d), onp.Array1D[np.complex128])
assert_type(gmean(_intp_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(gmean(_f32_nd), np.float32 | onp.ArrayND[np.float32])  # pyrefly:ignore[assert-type]
assert_type(gmean(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(gmean(_c64_nd), np.complex64 | onp.ArrayND[np.complex64])  # pyrefly:ignore[assert-type]
assert_type(gmean(_c128_nd), np.complex128 | onp.ArrayND[np.complex128])  # pyrefly:ignore[assert-type]

assert_type(gmean(_py_i_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(gmean(_py_f_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(gmean(_py_c_1d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(gmean(_intp_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(gmean(_f32_1d, keepdims=True), onp.Array1D[np.float32])
assert_type(gmean(_f64_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(gmean(_c64_1d, keepdims=True), onp.Array1D[np.complex64])
assert_type(gmean(_c128_1d, keepdims=True), onp.Array1D[np.complex128])
assert_type(gmean(_py_i_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(gmean(_py_f_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(gmean(_py_c_2d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(gmean(_intp_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(gmean(_f32_2d, keepdims=True), onp.Array2D[np.float32])
assert_type(gmean(_f64_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(gmean(_c64_2d, keepdims=True), onp.Array2D[np.complex64])
assert_type(gmean(_c128_2d, keepdims=True), onp.Array2D[np.complex128])
assert_type(gmean(_intp_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(gmean(_f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(gmean(_f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(gmean(_c64_nd, keepdims=True), onp.ArrayND[np.complex64])
assert_type(gmean(_c128_nd, keepdims=True), onp.ArrayND[np.complex128])

assert_type(gmean(_py_i_1d, axis=None), np.float64)
assert_type(gmean(_py_f_1d, axis=None), np.float64)
assert_type(gmean(_py_c_1d, axis=None), np.complex128)
assert_type(gmean(_intp_1d, axis=None), np.float64)
assert_type(gmean(_f32_1d, axis=None), np.float32)
assert_type(gmean(_f64_1d, axis=None), np.float64)
assert_type(gmean(_c64_1d, axis=None), np.complex64)
assert_type(gmean(_c128_1d, axis=None), np.complex128)
assert_type(gmean(_py_i_2d, axis=None), np.float64)
assert_type(gmean(_py_f_2d, axis=None), np.float64)
assert_type(gmean(_py_c_2d, axis=None), np.complex128)
assert_type(gmean(_intp_2d, axis=None), np.float64)
assert_type(gmean(_f32_2d, axis=None), np.float32)
assert_type(gmean(_f64_2d, axis=None), np.float64)
assert_type(gmean(_c64_2d, axis=None), np.complex64)
assert_type(gmean(_c128_2d, axis=None), np.complex128)
assert_type(gmean(_intp_nd, axis=None), np.float64)
assert_type(gmean(_f32_nd, axis=None), np.float32)
assert_type(gmean(_f64_nd, axis=None), np.float64)
assert_type(gmean(_c64_nd, axis=None), np.complex64)
assert_type(gmean(_c128_nd, axis=None), np.complex128)

assert_type(gmean(_py_f_1d, dtype=np.float16), np.float16)
assert_type(gmean(_f64_1d, dtype=np.float16), np.float16)
assert_type(gmean(_py_f_2d, dtype=np.float16), onp.Array1D[np.float16])
assert_type(gmean(_f64_2d, dtype=np.float16), onp.Array1D[np.float16])
assert_type(gmean(_f64_nd, dtype=np.float16), np.float16 | onp.ArrayND[np.float16])  # pyrefly:ignore[assert-type]

assert_type(gmean(_py_f_1d, dtype=np.float16, axis=None), np.float16)
assert_type(gmean(_f64_1d, dtype=np.float16, axis=None), np.float16)
assert_type(gmean(_py_f_2d, dtype=np.float16, axis=None), np.float16)
assert_type(gmean(_f64_2d, dtype=np.float16, axis=None), np.float16)
assert_type(gmean(_f64_nd, dtype=np.float16, axis=None), np.float16)

assert_type(gmean(_py_f_1d, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])
assert_type(gmean(_f64_1d, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])
assert_type(gmean(_py_f_2d, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])
assert_type(gmean(_f64_2d, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])
assert_type(gmean(_f64_nd, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])

# hmean (same as above)

assert_type(hmean(_py_i_1d), np.float64)
assert_type(hmean(_py_f_1d), np.float64)
assert_type(hmean(_py_c_1d), np.complex128)
assert_type(hmean(_intp_1d), np.float64)
assert_type(hmean(_f32_1d), np.float32)
assert_type(hmean(_f64_1d), np.float64)
assert_type(hmean(_c64_1d), np.complex64)
assert_type(hmean(_c128_1d), np.complex128)
assert_type(hmean(_py_i_2d), onp.Array1D[np.float64])
assert_type(hmean(_py_f_2d), onp.Array1D[np.float64])
assert_type(hmean(_py_c_2d), onp.Array1D[np.complex128])
assert_type(hmean(_intp_2d), onp.Array1D[np.float64])
assert_type(hmean(_f32_2d), onp.Array1D[np.float32])
assert_type(hmean(_f64_2d), onp.Array1D[np.float64])
assert_type(hmean(_c64_2d), onp.Array1D[np.complex64])
assert_type(hmean(_c128_2d), onp.Array1D[np.complex128])
assert_type(hmean(_intp_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(hmean(_f32_nd), np.float32 | onp.ArrayND[np.float32])  # pyrefly:ignore[assert-type]
assert_type(hmean(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(hmean(_c64_nd), np.complex64 | onp.ArrayND[np.complex64])  # pyrefly:ignore[assert-type]
assert_type(hmean(_c128_nd), np.complex128 | onp.ArrayND[np.complex128])  # pyrefly:ignore[assert-type]

assert_type(hmean(_py_i_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(hmean(_py_f_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(hmean(_py_c_1d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(hmean(_intp_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(hmean(_f32_1d, keepdims=True), onp.Array1D[np.float32])
assert_type(hmean(_f64_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(hmean(_c64_1d, keepdims=True), onp.Array1D[np.complex64])
assert_type(hmean(_c128_1d, keepdims=True), onp.Array1D[np.complex128])
assert_type(hmean(_py_i_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(hmean(_py_f_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(hmean(_py_c_2d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(hmean(_intp_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(hmean(_f32_2d, keepdims=True), onp.Array2D[np.float32])
assert_type(hmean(_f64_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(hmean(_c64_2d, keepdims=True), onp.Array2D[np.complex64])
assert_type(hmean(_c128_2d, keepdims=True), onp.Array2D[np.complex128])
assert_type(hmean(_intp_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(hmean(_f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(hmean(_f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(hmean(_c64_nd, keepdims=True), onp.ArrayND[np.complex64])
assert_type(hmean(_c128_nd, keepdims=True), onp.ArrayND[np.complex128])

assert_type(hmean(_py_i_1d, axis=None), np.float64)
assert_type(hmean(_py_f_1d, axis=None), np.float64)
assert_type(hmean(_py_c_1d, axis=None), np.complex128)
assert_type(hmean(_intp_1d, axis=None), np.float64)
assert_type(hmean(_f32_1d, axis=None), np.float32)
assert_type(hmean(_f64_1d, axis=None), np.float64)
assert_type(hmean(_c64_1d, axis=None), np.complex64)
assert_type(hmean(_c128_1d, axis=None), np.complex128)
assert_type(hmean(_py_i_2d, axis=None), np.float64)
assert_type(hmean(_py_f_2d, axis=None), np.float64)
assert_type(hmean(_py_c_2d, axis=None), np.complex128)
assert_type(hmean(_intp_2d, axis=None), np.float64)
assert_type(hmean(_f32_2d, axis=None), np.float32)
assert_type(hmean(_f64_2d, axis=None), np.float64)
assert_type(hmean(_c64_2d, axis=None), np.complex64)
assert_type(hmean(_c128_2d, axis=None), np.complex128)
assert_type(hmean(_intp_nd, axis=None), np.float64)
assert_type(hmean(_f32_nd, axis=None), np.float32)
assert_type(hmean(_f64_nd, axis=None), np.float64)
assert_type(hmean(_c64_nd, axis=None), np.complex64)
assert_type(hmean(_c128_nd, axis=None), np.complex128)

assert_type(hmean(_py_f_1d, dtype=np.float16), np.float16)
assert_type(hmean(_f64_1d, dtype=np.float16), np.float16)
assert_type(hmean(_py_f_2d, dtype=np.float16), onp.Array1D[np.float16])
assert_type(hmean(_f64_2d, dtype=np.float16), onp.Array1D[np.float16])
assert_type(hmean(_f64_nd, dtype=np.float16), np.float16 | onp.ArrayND[np.float16])  # pyrefly:ignore[assert-type]

assert_type(hmean(_py_f_1d, dtype=np.float16, axis=None), np.float16)
assert_type(hmean(_f64_1d, dtype=np.float16, axis=None), np.float16)
assert_type(hmean(_py_f_2d, dtype=np.float16, axis=None), np.float16)
assert_type(hmean(_f64_2d, dtype=np.float16, axis=None), np.float16)
assert_type(hmean(_f64_nd, dtype=np.float16, axis=None), np.float16)

assert_type(hmean(_py_f_1d, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])
assert_type(hmean(_f64_1d, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])
assert_type(hmean(_py_f_2d, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])
assert_type(hmean(_f64_2d, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])
assert_type(hmean(_f64_nd, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])

# pmean (same as above)

assert_type(pmean(_py_i_1d, 2), np.float64)
assert_type(pmean(_py_f_1d, 2), np.float64)
assert_type(pmean(_py_c_1d, 2), np.complex128)
assert_type(pmean(_intp_1d, 2), np.float64)
assert_type(pmean(_f32_1d, 2), np.float32)
assert_type(pmean(_f64_1d, 2), np.float64)
assert_type(pmean(_c64_1d, 2), np.complex64)
assert_type(pmean(_c128_1d, 2), np.complex128)
assert_type(pmean(_py_i_2d, 2), onp.Array1D[np.float64])
assert_type(pmean(_py_f_2d, 2), onp.Array1D[np.float64])
assert_type(pmean(_py_c_2d, 2), onp.Array1D[np.complex128])
assert_type(pmean(_intp_2d, 2), onp.Array1D[np.float64])
assert_type(pmean(_f32_2d, 2), onp.Array1D[np.float32])
assert_type(pmean(_f64_2d, 2), onp.Array1D[np.float64])
assert_type(pmean(_c64_2d, 2), onp.Array1D[np.complex64])
assert_type(pmean(_c128_2d, 2), onp.Array1D[np.complex128])
assert_type(pmean(_intp_nd, 2), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(pmean(_f32_nd, 2), np.float32 | onp.ArrayND[np.float32])  # pyrefly:ignore[assert-type]
assert_type(pmean(_f64_nd, 2), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(pmean(_c64_nd, 2), np.complex64 | onp.ArrayND[np.complex64])  # pyrefly:ignore[assert-type]
assert_type(pmean(_c128_nd, 2), np.complex128 | onp.ArrayND[np.complex128])  # pyrefly:ignore[assert-type]

assert_type(pmean(_py_i_1d, 2, keepdims=True), onp.ArrayND[np.float64])
assert_type(pmean(_py_f_1d, 2, keepdims=True), onp.ArrayND[np.float64])
assert_type(pmean(_py_c_1d, 2, keepdims=True), onp.ArrayND[np.complex128])
assert_type(pmean(_intp_1d, 2, keepdims=True), onp.Array1D[np.float64])
assert_type(pmean(_f32_1d, 2, keepdims=True), onp.Array1D[np.float32])
assert_type(pmean(_f64_1d, 2, keepdims=True), onp.Array1D[np.float64])
assert_type(pmean(_c64_1d, 2, keepdims=True), onp.Array1D[np.complex64])
assert_type(pmean(_c128_1d, 2, keepdims=True), onp.Array1D[np.complex128])
assert_type(pmean(_py_i_2d, 2, keepdims=True), onp.ArrayND[np.float64])
assert_type(pmean(_py_f_2d, 2, keepdims=True), onp.ArrayND[np.float64])
assert_type(pmean(_py_c_2d, 2, keepdims=True), onp.ArrayND[np.complex128])
assert_type(pmean(_intp_2d, 2, keepdims=True), onp.Array2D[np.float64])
assert_type(pmean(_f32_2d, 2, keepdims=True), onp.Array2D[np.float32])
assert_type(pmean(_f64_2d, 2, keepdims=True), onp.Array2D[np.float64])
assert_type(pmean(_c64_2d, 2, keepdims=True), onp.Array2D[np.complex64])
assert_type(pmean(_c128_2d, 2, keepdims=True), onp.Array2D[np.complex128])
assert_type(pmean(_intp_nd, 2, keepdims=True), onp.ArrayND[np.float64])
assert_type(pmean(_f32_nd, 2, keepdims=True), onp.ArrayND[np.float32])
assert_type(pmean(_f64_nd, 2, keepdims=True), onp.ArrayND[np.float64])
assert_type(pmean(_c64_nd, 2, keepdims=True), onp.ArrayND[np.complex64])
assert_type(pmean(_c128_nd, 2, keepdims=True), onp.ArrayND[np.complex128])

assert_type(pmean(_py_i_1d, 2, axis=None), np.float64)
assert_type(pmean(_py_f_1d, 2, axis=None), np.float64)
assert_type(pmean(_py_c_1d, 2, axis=None), np.complex128)
assert_type(pmean(_intp_1d, 2, axis=None), np.float64)
assert_type(pmean(_f32_1d, 2, axis=None), np.float32)
assert_type(pmean(_f64_1d, 2, axis=None), np.float64)
assert_type(pmean(_c64_1d, 2, axis=None), np.complex64)
assert_type(pmean(_c128_1d, 2, axis=None), np.complex128)
assert_type(pmean(_py_i_2d, 2, axis=None), np.float64)
assert_type(pmean(_py_f_2d, 2, axis=None), np.float64)
assert_type(pmean(_py_c_2d, 2, axis=None), np.complex128)
assert_type(pmean(_intp_2d, 2, axis=None), np.float64)
assert_type(pmean(_f32_2d, 2, axis=None), np.float32)
assert_type(pmean(_f64_2d, 2, axis=None), np.float64)
assert_type(pmean(_c64_2d, 2, axis=None), np.complex64)
assert_type(pmean(_c128_2d, 2, axis=None), np.complex128)
assert_type(pmean(_intp_nd, 2, axis=None), np.float64)
assert_type(pmean(_f32_nd, 2, axis=None), np.float32)
assert_type(pmean(_f64_nd, 2, axis=None), np.float64)
assert_type(pmean(_c64_nd, 2, axis=None), np.complex64)
assert_type(pmean(_c128_nd, 2, axis=None), np.complex128)

assert_type(pmean(_py_f_1d, 2, dtype=np.float16), np.float16)
assert_type(pmean(_f64_1d, 2, dtype=np.float16), np.float16)
assert_type(pmean(_py_f_2d, 2, dtype=np.float16), onp.Array1D[np.float16])
assert_type(pmean(_f64_2d, 2, dtype=np.float16), onp.Array1D[np.float16])
assert_type(pmean(_f64_nd, 2, dtype=np.float16), np.float16 | onp.ArrayND[np.float16])  # pyrefly:ignore[assert-type]

assert_type(pmean(_py_f_1d, 2, dtype=np.float16, axis=None), np.float16)
assert_type(pmean(_f64_1d, 2, dtype=np.float16, axis=None), np.float16)
assert_type(pmean(_py_f_2d, 2, dtype=np.float16, axis=None), np.float16)
assert_type(pmean(_f64_2d, 2, dtype=np.float16, axis=None), np.float16)
assert_type(pmean(_f64_nd, 2, dtype=np.float16, axis=None), np.float16)

assert_type(pmean(_py_f_1d, 2, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])
assert_type(pmean(_f64_1d, 2, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])
assert_type(pmean(_py_f_2d, 2, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])
assert_type(pmean(_f64_2d, 2, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])
assert_type(pmean(_f64_nd, 2, dtype=np.float16, keepdims=True), onp.ArrayND[np.float16])

# tmean

assert_type(tmean(_py_i_1d), np.float64)
assert_type(tmean(_py_f_1d), np.float64)
assert_type(tmean(_py_c_1d), np.complex128)
assert_type(tmean(_intp_1d), np.float64)
assert_type(tmean(_f32_1d), np.float32)
assert_type(tmean(_f64_1d), np.float64)
assert_type(tmean(_c64_1d), np.complex64)
assert_type(tmean(_c128_1d), np.complex128)
assert_type(tmean(_py_i_2d), onp.Array1D[np.float64])
assert_type(tmean(_py_f_2d), onp.Array1D[np.float64])
assert_type(tmean(_py_c_2d), onp.Array1D[np.complex128])
assert_type(tmean(_intp_2d), onp.Array1D[np.float64])
assert_type(tmean(_f32_2d), onp.Array1D[np.float32])
assert_type(tmean(_f64_2d), onp.Array1D[np.float64])
assert_type(tmean(_c64_2d), onp.Array1D[np.complex64])
assert_type(tmean(_c128_2d), onp.Array1D[np.complex128])
assert_type(tmean(_intp_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(tmean(_f32_nd), np.float32 | onp.ArrayND[np.float32])  # pyrefly:ignore[assert-type]
assert_type(tmean(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(tmean(_c64_nd), np.complex64 | onp.ArrayND[np.complex64])  # pyrefly:ignore[assert-type]
assert_type(tmean(_c128_nd), np.complex128 | onp.ArrayND[np.complex128])  # pyrefly:ignore[assert-type]

assert_type(tmean(_py_i_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmean(_py_f_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmean(_py_c_1d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(tmean(_intp_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(tmean(_f32_1d, keepdims=True), onp.Array1D[np.float32])
assert_type(tmean(_f64_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(tmean(_c64_1d, keepdims=True), onp.Array1D[np.complex64])
assert_type(tmean(_c128_1d, keepdims=True), onp.Array1D[np.complex128])
assert_type(tmean(_py_i_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmean(_py_f_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmean(_py_c_2d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(tmean(_intp_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(tmean(_f32_2d, keepdims=True), onp.Array2D[np.float32])
assert_type(tmean(_f64_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(tmean(_c64_2d, keepdims=True), onp.Array2D[np.complex64])
assert_type(tmean(_c128_2d, keepdims=True), onp.Array2D[np.complex128])
assert_type(tmean(_intp_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmean(_f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(tmean(_f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmean(_c64_nd, keepdims=True), onp.ArrayND[np.complex64])
assert_type(tmean(_c128_nd, keepdims=True), onp.ArrayND[np.complex128])

assert_type(tmean(_py_i_1d, axis=None), np.float64)
assert_type(tmean(_py_f_1d, axis=None), np.float64)
assert_type(tmean(_py_c_1d, axis=None), np.complex128)
assert_type(tmean(_intp_1d, axis=None), np.float64)
assert_type(tmean(_f32_1d, axis=None), np.float32)
assert_type(tmean(_f64_1d, axis=None), np.float64)
assert_type(tmean(_c64_1d, axis=None), np.complex64)
assert_type(tmean(_c128_1d, axis=None), np.complex128)
assert_type(tmean(_py_i_2d, axis=None), np.float64)
assert_type(tmean(_py_f_2d, axis=None), np.float64)
assert_type(tmean(_py_c_2d, axis=None), np.complex128)
assert_type(tmean(_intp_2d, axis=None), np.float64)
assert_type(tmean(_f32_2d, axis=None), np.float32)
assert_type(tmean(_f64_2d, axis=None), np.float64)
assert_type(tmean(_c64_2d, axis=None), np.complex64)
assert_type(tmean(_c128_2d, axis=None), np.complex128)
assert_type(tmean(_intp_nd, axis=None), np.float64)
assert_type(tmean(_f32_nd, axis=None), np.float32)
assert_type(tmean(_f64_nd, axis=None), np.float64)
assert_type(tmean(_c64_nd, axis=None), np.complex64)
assert_type(tmean(_c128_nd, axis=None), np.complex128)

# tvar (same as above)

assert_type(tvar(_py_i_1d), np.float64)
assert_type(tvar(_py_f_1d), np.float64)
assert_type(tvar(_py_c_1d), np.complex128)
assert_type(tvar(_intp_1d), np.float64)
assert_type(tvar(_f32_1d), np.float32)
assert_type(tvar(_f64_1d), np.float64)
assert_type(tvar(_c64_1d), np.complex64)
assert_type(tvar(_c128_1d), np.complex128)
assert_type(tvar(_py_i_2d), onp.Array1D[np.float64])
assert_type(tvar(_py_f_2d), onp.Array1D[np.float64])
assert_type(tvar(_py_c_2d), onp.Array1D[np.complex128])
assert_type(tvar(_intp_2d), onp.Array1D[np.float64])
assert_type(tvar(_f32_2d), onp.Array1D[np.float32])
assert_type(tvar(_f64_2d), onp.Array1D[np.float64])
assert_type(tvar(_c64_2d), onp.Array1D[np.complex64])
assert_type(tvar(_c128_2d), onp.Array1D[np.complex128])
assert_type(tvar(_intp_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(tvar(_f32_nd), np.float32 | onp.ArrayND[np.float32])  # pyrefly:ignore[assert-type]
assert_type(tvar(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(tvar(_c64_nd), np.complex64 | onp.ArrayND[np.complex64])  # pyrefly:ignore[assert-type]
assert_type(tvar(_c128_nd), np.complex128 | onp.ArrayND[np.complex128])  # pyrefly:ignore[assert-type]

assert_type(tvar(_py_i_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tvar(_py_f_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tvar(_py_c_1d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(tvar(_intp_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(tvar(_f32_1d, keepdims=True), onp.Array1D[np.float32])
assert_type(tvar(_f64_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(tvar(_c64_1d, keepdims=True), onp.Array1D[np.complex64])
assert_type(tvar(_c128_1d, keepdims=True), onp.Array1D[np.complex128])
assert_type(tvar(_py_i_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tvar(_py_f_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tvar(_py_c_2d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(tvar(_intp_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(tvar(_f32_2d, keepdims=True), onp.Array2D[np.float32])
assert_type(tvar(_f64_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(tvar(_c64_2d, keepdims=True), onp.Array2D[np.complex64])
assert_type(tvar(_c128_2d, keepdims=True), onp.Array2D[np.complex128])
assert_type(tvar(_intp_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(tvar(_f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(tvar(_f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(tvar(_c64_nd, keepdims=True), onp.ArrayND[np.complex64])
assert_type(tvar(_c128_nd, keepdims=True), onp.ArrayND[np.complex128])

assert_type(tvar(_py_i_1d, axis=None), np.float64)
assert_type(tvar(_py_f_1d, axis=None), np.float64)
assert_type(tvar(_py_c_1d, axis=None), np.complex128)
assert_type(tvar(_intp_1d, axis=None), np.float64)
assert_type(tvar(_f32_1d, axis=None), np.float32)
assert_type(tvar(_f64_1d, axis=None), np.float64)
assert_type(tvar(_c64_1d, axis=None), np.complex64)
assert_type(tvar(_c128_1d, axis=None), np.complex128)
assert_type(tvar(_py_i_2d, axis=None), np.float64)
assert_type(tvar(_py_f_2d, axis=None), np.float64)
assert_type(tvar(_py_c_2d, axis=None), np.complex128)
assert_type(tvar(_intp_2d, axis=None), np.float64)
assert_type(tvar(_f32_2d, axis=None), np.float32)
assert_type(tvar(_f64_2d, axis=None), np.float64)
assert_type(tvar(_c64_2d, axis=None), np.complex64)
assert_type(tvar(_c128_2d, axis=None), np.complex128)
assert_type(tvar(_intp_nd, axis=None), np.float64)
assert_type(tvar(_f32_nd, axis=None), np.float32)
assert_type(tvar(_f64_nd, axis=None), np.float64)
assert_type(tvar(_c64_nd, axis=None), np.complex64)
assert_type(tvar(_c128_nd, axis=None), np.complex128)

# tstd (same as above)

assert_type(tstd(_py_i_1d), np.float64)
assert_type(tstd(_py_f_1d), np.float64)
assert_type(tstd(_py_c_1d), np.complex128)
assert_type(tstd(_intp_1d), np.float64)
assert_type(tstd(_f32_1d), np.float32)
assert_type(tstd(_f64_1d), np.float64)
assert_type(tstd(_c64_1d), np.complex64)
assert_type(tstd(_c128_1d), np.complex128)
assert_type(tstd(_py_i_2d), onp.Array1D[np.float64])
assert_type(tstd(_py_f_2d), onp.Array1D[np.float64])
assert_type(tstd(_py_c_2d), onp.Array1D[np.complex128])
assert_type(tstd(_intp_2d), onp.Array1D[np.float64])
assert_type(tstd(_f32_2d), onp.Array1D[np.float32])
assert_type(tstd(_f64_2d), onp.Array1D[np.float64])
assert_type(tstd(_c64_2d), onp.Array1D[np.complex64])
assert_type(tstd(_c128_2d), onp.Array1D[np.complex128])
assert_type(tstd(_intp_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(tstd(_f32_nd), np.float32 | onp.ArrayND[np.float32])  # pyrefly:ignore[assert-type]
assert_type(tstd(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(tstd(_c64_nd), np.complex64 | onp.ArrayND[np.complex64])  # pyrefly:ignore[assert-type]
assert_type(tstd(_c128_nd), np.complex128 | onp.ArrayND[np.complex128])  # pyrefly:ignore[assert-type]

assert_type(tstd(_py_i_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tstd(_py_f_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tstd(_py_c_1d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(tstd(_intp_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(tstd(_f32_1d, keepdims=True), onp.Array1D[np.float32])
assert_type(tstd(_f64_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(tstd(_c64_1d, keepdims=True), onp.Array1D[np.complex64])
assert_type(tstd(_c128_1d, keepdims=True), onp.Array1D[np.complex128])
assert_type(tstd(_py_i_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tstd(_py_f_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tstd(_py_c_2d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(tstd(_intp_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(tstd(_f32_2d, keepdims=True), onp.Array2D[np.float32])
assert_type(tstd(_f64_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(tstd(_c64_2d, keepdims=True), onp.Array2D[np.complex64])
assert_type(tstd(_c128_2d, keepdims=True), onp.Array2D[np.complex128])
assert_type(tstd(_intp_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(tstd(_f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(tstd(_f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(tstd(_c64_nd, keepdims=True), onp.ArrayND[np.complex64])
assert_type(tstd(_c128_nd, keepdims=True), onp.ArrayND[np.complex128])

assert_type(tstd(_py_i_1d, axis=None), np.float64)
assert_type(tstd(_py_f_1d, axis=None), np.float64)
assert_type(tstd(_py_c_1d, axis=None), np.complex128)
assert_type(tstd(_intp_1d, axis=None), np.float64)
assert_type(tstd(_f32_1d, axis=None), np.float32)
assert_type(tstd(_f64_1d, axis=None), np.float64)
assert_type(tstd(_c64_1d, axis=None), np.complex64)
assert_type(tstd(_c128_1d, axis=None), np.complex128)
assert_type(tstd(_py_i_2d, axis=None), np.float64)
assert_type(tstd(_py_f_2d, axis=None), np.float64)
assert_type(tstd(_py_c_2d, axis=None), np.complex128)
assert_type(tstd(_intp_2d, axis=None), np.float64)
assert_type(tstd(_f32_2d, axis=None), np.float32)
assert_type(tstd(_f64_2d, axis=None), np.float64)
assert_type(tstd(_c64_2d, axis=None), np.complex64)
assert_type(tstd(_c128_2d, axis=None), np.complex128)
assert_type(tstd(_intp_nd, axis=None), np.float64)
assert_type(tstd(_f32_nd, axis=None), np.float32)
assert_type(tstd(_f64_nd, axis=None), np.float64)
assert_type(tstd(_c64_nd, axis=None), np.complex64)
assert_type(tstd(_c128_nd, axis=None), np.complex128)

# tsem (same as above)

assert_type(tsem(_py_i_1d), np.float64)
assert_type(tsem(_py_f_1d), np.float64)
assert_type(tsem(_py_c_1d), np.complex128)
assert_type(tsem(_intp_1d), np.float64)
assert_type(tsem(_f32_1d), np.float32)
assert_type(tsem(_f64_1d), np.float64)
assert_type(tsem(_c64_1d), np.complex64)
assert_type(tsem(_c128_1d), np.complex128)
assert_type(tsem(_py_i_2d), onp.Array1D[np.float64])
assert_type(tsem(_py_f_2d), onp.Array1D[np.float64])
assert_type(tsem(_py_c_2d), onp.Array1D[np.complex128])
assert_type(tsem(_intp_2d), onp.Array1D[np.float64])
assert_type(tsem(_f32_2d), onp.Array1D[np.float32])
assert_type(tsem(_f64_2d), onp.Array1D[np.float64])
assert_type(tsem(_c64_2d), onp.Array1D[np.complex64])
assert_type(tsem(_c128_2d), onp.Array1D[np.complex128])
assert_type(tsem(_intp_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(tsem(_f32_nd), np.float32 | onp.ArrayND[np.float32])  # pyrefly:ignore[assert-type]
assert_type(tsem(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(tsem(_c64_nd), np.complex64 | onp.ArrayND[np.complex64])  # pyrefly:ignore[assert-type]
assert_type(tsem(_c128_nd), np.complex128 | onp.ArrayND[np.complex128])  # pyrefly:ignore[assert-type]

assert_type(tsem(_py_i_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tsem(_py_f_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tsem(_py_c_1d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(tsem(_intp_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(tsem(_f32_1d, keepdims=True), onp.Array1D[np.float32])
assert_type(tsem(_f64_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(tsem(_c64_1d, keepdims=True), onp.Array1D[np.complex64])
assert_type(tsem(_c128_1d, keepdims=True), onp.Array1D[np.complex128])
assert_type(tsem(_py_i_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tsem(_py_f_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tsem(_py_c_2d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(tsem(_intp_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(tsem(_f32_2d, keepdims=True), onp.Array2D[np.float32])
assert_type(tsem(_f64_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(tsem(_c64_2d, keepdims=True), onp.Array2D[np.complex64])
assert_type(tsem(_c128_2d, keepdims=True), onp.Array2D[np.complex128])
assert_type(tsem(_intp_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(tsem(_f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(tsem(_f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(tsem(_c64_nd, keepdims=True), onp.ArrayND[np.complex64])
assert_type(tsem(_c128_nd, keepdims=True), onp.ArrayND[np.complex128])

assert_type(tsem(_py_i_1d, axis=None), np.float64)
assert_type(tsem(_py_f_1d, axis=None), np.float64)
assert_type(tsem(_py_c_1d, axis=None), np.complex128)
assert_type(tsem(_intp_1d, axis=None), np.float64)
assert_type(tsem(_f32_1d, axis=None), np.float32)
assert_type(tsem(_f64_1d, axis=None), np.float64)
assert_type(tsem(_c64_1d, axis=None), np.complex64)
assert_type(tsem(_c128_1d, axis=None), np.complex128)
assert_type(tsem(_py_i_2d, axis=None), np.float64)
assert_type(tsem(_py_f_2d, axis=None), np.float64)
assert_type(tsem(_py_c_2d, axis=None), np.complex128)
assert_type(tsem(_intp_2d, axis=None), np.float64)
assert_type(tsem(_f32_2d, axis=None), np.float32)
assert_type(tsem(_f64_2d, axis=None), np.float64)
assert_type(tsem(_c64_2d, axis=None), np.complex64)
assert_type(tsem(_c128_2d, axis=None), np.complex128)
assert_type(tsem(_intp_nd, axis=None), np.float64)
assert_type(tsem(_f32_nd, axis=None), np.float32)
assert_type(tsem(_f64_nd, axis=None), np.float64)
assert_type(tsem(_c64_nd, axis=None), np.complex64)
assert_type(tsem(_c128_nd, axis=None), np.complex128)

# tmin (same as above)

assert_type(tmin(_py_i_1d), np.float64)
assert_type(tmin(_py_f_1d), np.float64)
assert_type(tmin(_py_c_1d), np.complex128)
assert_type(tmin(_intp_1d), np.float64)
assert_type(tmin(_f32_1d), np.float32)
assert_type(tmin(_f64_1d), np.float64)
assert_type(tmin(_c64_1d), np.complex64)
assert_type(tmin(_c128_1d), np.complex128)
assert_type(tmin(_py_i_2d), onp.Array1D[np.float64])
assert_type(tmin(_py_f_2d), onp.Array1D[np.float64])
assert_type(tmin(_py_c_2d), onp.Array1D[np.complex128])
assert_type(tmin(_intp_2d), onp.Array1D[np.float64])
assert_type(tmin(_f32_2d), onp.Array1D[np.float32])
assert_type(tmin(_f64_2d), onp.Array1D[np.float64])
assert_type(tmin(_c64_2d), onp.Array1D[np.complex64])
assert_type(tmin(_c128_2d), onp.Array1D[np.complex128])
assert_type(tmin(_intp_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(tmin(_f32_nd), np.float32 | onp.ArrayND[np.float32])  # pyrefly:ignore[assert-type]
assert_type(tmin(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(tmin(_c64_nd), np.complex64 | onp.ArrayND[np.complex64])  # pyrefly:ignore[assert-type]
assert_type(tmin(_c128_nd), np.complex128 | onp.ArrayND[np.complex128])  # pyrefly:ignore[assert-type]

assert_type(tmin(_py_i_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmin(_py_f_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmin(_py_c_1d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(tmin(_intp_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(tmin(_f32_1d, keepdims=True), onp.Array1D[np.float32])
assert_type(tmin(_f64_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(tmin(_c64_1d, keepdims=True), onp.Array1D[np.complex64])
assert_type(tmin(_c128_1d, keepdims=True), onp.Array1D[np.complex128])
assert_type(tmin(_py_i_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmin(_py_f_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmin(_py_c_2d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(tmin(_intp_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(tmin(_f32_2d, keepdims=True), onp.Array2D[np.float32])
assert_type(tmin(_f64_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(tmin(_c64_2d, keepdims=True), onp.Array2D[np.complex64])
assert_type(tmin(_c128_2d, keepdims=True), onp.Array2D[np.complex128])
assert_type(tmin(_intp_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmin(_f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(tmin(_f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmin(_c64_nd, keepdims=True), onp.ArrayND[np.complex64])
assert_type(tmin(_c128_nd, keepdims=True), onp.ArrayND[np.complex128])

assert_type(tmin(_py_i_1d, axis=None), np.float64)
assert_type(tmin(_py_f_1d, axis=None), np.float64)
assert_type(tmin(_py_c_1d, axis=None), np.complex128)
assert_type(tmin(_intp_1d, axis=None), np.float64)
assert_type(tmin(_f32_1d, axis=None), np.float32)
assert_type(tmin(_f64_1d, axis=None), np.float64)
assert_type(tmin(_c64_1d, axis=None), np.complex64)
assert_type(tmin(_c128_1d, axis=None), np.complex128)
assert_type(tmin(_py_i_2d, axis=None), np.float64)
assert_type(tmin(_py_f_2d, axis=None), np.float64)
assert_type(tmin(_py_c_2d, axis=None), np.complex128)
assert_type(tmin(_intp_2d, axis=None), np.float64)
assert_type(tmin(_f32_2d, axis=None), np.float32)
assert_type(tmin(_f64_2d, axis=None), np.float64)
assert_type(tmin(_c64_2d, axis=None), np.complex64)
assert_type(tmin(_c128_2d, axis=None), np.complex128)
assert_type(tmin(_intp_nd, axis=None), np.float64)
assert_type(tmin(_f32_nd, axis=None), np.float32)
assert_type(tmin(_f64_nd, axis=None), np.float64)
assert_type(tmin(_c64_nd, axis=None), np.complex64)
assert_type(tmin(_c128_nd, axis=None), np.complex128)

# tmax (same as above)

assert_type(tmax(_py_i_1d), np.float64)
assert_type(tmax(_py_f_1d), np.float64)
assert_type(tmax(_py_c_1d), np.complex128)
assert_type(tmax(_intp_1d), np.float64)
assert_type(tmax(_f32_1d), np.float32)
assert_type(tmax(_f64_1d), np.float64)
assert_type(tmax(_c64_1d), np.complex64)
assert_type(tmax(_c128_1d), np.complex128)
assert_type(tmax(_py_i_2d), onp.Array1D[np.float64])
assert_type(tmax(_py_f_2d), onp.Array1D[np.float64])
assert_type(tmax(_py_c_2d), onp.Array1D[np.complex128])
assert_type(tmax(_intp_2d), onp.Array1D[np.float64])
assert_type(tmax(_f32_2d), onp.Array1D[np.float32])
assert_type(tmax(_f64_2d), onp.Array1D[np.float64])
assert_type(tmax(_c64_2d), onp.Array1D[np.complex64])
assert_type(tmax(_c128_2d), onp.Array1D[np.complex128])
assert_type(tmax(_intp_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(tmax(_f32_nd), np.float32 | onp.ArrayND[np.float32])  # pyrefly:ignore[assert-type]
assert_type(tmax(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(tmax(_c64_nd), np.complex64 | onp.ArrayND[np.complex64])  # pyrefly:ignore[assert-type]
assert_type(tmax(_c128_nd), np.complex128 | onp.ArrayND[np.complex128])  # pyrefly:ignore[assert-type]

assert_type(tmax(_py_i_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmax(_py_f_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmax(_py_c_1d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(tmax(_intp_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(tmax(_f32_1d, keepdims=True), onp.Array1D[np.float32])
assert_type(tmax(_f64_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(tmax(_c64_1d, keepdims=True), onp.Array1D[np.complex64])
assert_type(tmax(_c128_1d, keepdims=True), onp.Array1D[np.complex128])
assert_type(tmax(_py_i_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmax(_py_f_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmax(_py_c_2d, keepdims=True), onp.ArrayND[np.complex128])
assert_type(tmax(_intp_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(tmax(_f32_2d, keepdims=True), onp.Array2D[np.float32])
assert_type(tmax(_f64_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(tmax(_c64_2d, keepdims=True), onp.Array2D[np.complex64])
assert_type(tmax(_c128_2d, keepdims=True), onp.Array2D[np.complex128])
assert_type(tmax(_intp_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmax(_f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(tmax(_f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(tmax(_c64_nd, keepdims=True), onp.ArrayND[np.complex64])
assert_type(tmax(_c128_nd, keepdims=True), onp.ArrayND[np.complex128])

assert_type(tmax(_py_i_1d, axis=None), np.float64)
assert_type(tmax(_py_f_1d, axis=None), np.float64)
assert_type(tmax(_py_c_1d, axis=None), np.complex128)
assert_type(tmax(_intp_1d, axis=None), np.float64)
assert_type(tmax(_f32_1d, axis=None), np.float32)
assert_type(tmax(_f64_1d, axis=None), np.float64)
assert_type(tmax(_c64_1d, axis=None), np.complex64)
assert_type(tmax(_c128_1d, axis=None), np.complex128)
assert_type(tmax(_py_i_2d, axis=None), np.float64)
assert_type(tmax(_py_f_2d, axis=None), np.float64)
assert_type(tmax(_py_c_2d, axis=None), np.complex128)
assert_type(tmax(_intp_2d, axis=None), np.float64)
assert_type(tmax(_f32_2d, axis=None), np.float32)
assert_type(tmax(_f64_2d, axis=None), np.float64)
assert_type(tmax(_c64_2d, axis=None), np.complex64)
assert_type(tmax(_c128_2d, axis=None), np.complex128)
assert_type(tmax(_intp_nd, axis=None), np.float64)
assert_type(tmax(_f32_nd, axis=None), np.float32)
assert_type(tmax(_f64_nd, axis=None), np.float64)
assert_type(tmax(_c64_nd, axis=None), np.complex64)
assert_type(tmax(_c128_nd, axis=None), np.complex128)

# gstd

assert_type(gstd(_f64_1d), np.float64)
assert_type(gstd(_f64_nd, axis=None), np.float64)
assert_type(gstd(_f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(gstd(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]

# describe

assert_type(describe(_py_i_1d), DescribeResult[np.int_, np.float64])
assert_type(describe(_py_f_1d), DescribeResult[np.float64, np.float64])
assert_type(describe(_intp_1d), DescribeResult[np.int_, np.float64])
assert_type(describe(_f32_1d), DescribeResult[np.float32, np.float32])
assert_type(describe(_f64_1d), DescribeResult[np.float64, np.float64])
assert_type(describe(_f64_2d), DescribeResult[onp.Array1D[np.float64], onp.Array1D[np.float64]])

assert_type(describe(_py_i_1d, axis=None), DescribeResult[np.int_, np.float64])
assert_type(describe(_py_f_1d, axis=None), DescribeResult[np.float64, np.float64])
assert_type(describe(_intp_1d, axis=None), DescribeResult[np.int_, np.float64])
assert_type(describe(_f32_1d, axis=None), DescribeResult[np.float32, np.float32])
assert_type(describe(_f64_1d, axis=None), DescribeResult[np.float64, np.float64])
assert_type(describe(_f64_2d, axis=None), DescribeResult[np.float64, np.float64])

# skewtest

assert_type(skewtest(_py_f_1d), SkewtestResult[np.float64])
assert_type(skewtest(_intp_1d), SkewtestResult[np.float64])
assert_type(skewtest(_f32_1d), SkewtestResult[np.float32])
assert_type(skewtest(_f64_1d), SkewtestResult[np.float64])
assert_type(skewtest(_f64_2d), SkewtestResult[np.float64])

assert_type(skewtest(_py_f_1d, axis=1), SkewtestResult[onp.ArrayND[np.float64]])
assert_type(skewtest(_intp_1d, axis=1), SkewtestResult[onp.ArrayND[np.float64]])
assert_type(skewtest(_f32_1d, axis=1), SkewtestResult[onp.ArrayND[np.float32]])
assert_type(skewtest(_f64_1d, axis=1), SkewtestResult[onp.ArrayND[np.float64]])
assert_type(skewtest(_f64_2d, axis=1), SkewtestResult[onp.ArrayND[np.float64]])

assert_type(skewtest(_py_f_1d, keepdims=True), SkewtestResult[onp.ArrayND[np.float64]])
assert_type(skewtest(_intp_1d, keepdims=True), SkewtestResult[onp.Array1D[np.float64]])
assert_type(skewtest(_f32_1d, keepdims=True), SkewtestResult[onp.Array1D[np.float32]])
assert_type(skewtest(_f64_1d, keepdims=True), SkewtestResult[onp.Array1D[np.float64]])
assert_type(skewtest(_f64_2d, keepdims=True), SkewtestResult[onp.Array2D[np.float64]])

# kurtosistest

assert_type(kurtosistest(_py_f_1d), KurtosistestResult[np.float64])
assert_type(kurtosistest(_intp_1d), KurtosistestResult[np.float64])
assert_type(kurtosistest(_f32_1d), KurtosistestResult[np.float32])
assert_type(kurtosistest(_f64_1d), KurtosistestResult[np.float64])
assert_type(kurtosistest(_f64_2d), KurtosistestResult[np.float64])

assert_type(kurtosistest(_py_f_1d, axis=1), KurtosistestResult[onp.ArrayND[np.float64]])
assert_type(kurtosistest(_intp_1d, axis=1), KurtosistestResult[onp.ArrayND[np.float64]])
assert_type(kurtosistest(_f32_1d, axis=1), KurtosistestResult[onp.ArrayND[np.float32]])
assert_type(kurtosistest(_f64_1d, axis=1), KurtosistestResult[onp.ArrayND[np.float64]])
assert_type(kurtosistest(_f64_2d, axis=1), KurtosistestResult[onp.ArrayND[np.float64]])

assert_type(kurtosistest(_py_f_1d, keepdims=True), KurtosistestResult[onp.ArrayND[np.float64]])
assert_type(kurtosistest(_intp_1d, keepdims=True), KurtosistestResult[onp.Array1D[np.float64]])
assert_type(kurtosistest(_f32_1d, keepdims=True), KurtosistestResult[onp.Array1D[np.float32]])
assert_type(kurtosistest(_f64_1d, keepdims=True), KurtosistestResult[onp.Array1D[np.float64]])
assert_type(kurtosistest(_f64_2d, keepdims=True), KurtosistestResult[onp.Array2D[np.float64]])

# normaltest

assert_type(normaltest(_py_f_1d), NormaltestResult[np.float64])
assert_type(normaltest(_intp_1d), NormaltestResult[np.float64])
assert_type(normaltest(_f32_1d), NormaltestResult[np.float32])
assert_type(normaltest(_f64_1d), NormaltestResult[np.float64])
assert_type(normaltest(_f64_2d), NormaltestResult[np.float64])

assert_type(normaltest(_py_f_1d, axis=1), NormaltestResult[onp.ArrayND[np.float64]])
assert_type(normaltest(_intp_1d, axis=1), NormaltestResult[onp.ArrayND[np.float64]])
assert_type(normaltest(_f32_1d, axis=1), NormaltestResult[onp.ArrayND[np.float32]])
assert_type(normaltest(_f64_1d, axis=1), NormaltestResult[onp.ArrayND[np.float64]])
assert_type(normaltest(_f64_2d, axis=1), NormaltestResult[onp.ArrayND[np.float64]])

assert_type(normaltest(_py_f_1d, keepdims=True), NormaltestResult[onp.ArrayND[np.float64]])
assert_type(normaltest(_intp_1d, keepdims=True), NormaltestResult[onp.Array1D[np.float64]])
assert_type(normaltest(_f32_1d, keepdims=True), NormaltestResult[onp.Array1D[np.float32]])
assert_type(normaltest(_f64_1d, keepdims=True), NormaltestResult[onp.Array1D[np.float64]])
assert_type(normaltest(_f64_2d, keepdims=True), NormaltestResult[onp.Array2D[np.float64]])

# jarque_bera

assert_type(jarque_bera(_py_f_1d), SignificanceResult[np.float64])
assert_type(jarque_bera(_intp_1d), SignificanceResult[np.float64])
assert_type(jarque_bera(_f32_1d), SignificanceResult[np.float32])
assert_type(jarque_bera(_f64_1d), SignificanceResult[np.float64])
assert_type(jarque_bera(_f64_2d), SignificanceResult[np.float64])

assert_type(jarque_bera(_py_f_1d, axis=1), SignificanceResult[onp.ArrayND[np.float64]])
assert_type(jarque_bera(_intp_1d, axis=1), SignificanceResult[onp.ArrayND[np.float64]])
assert_type(jarque_bera(_f32_1d, axis=1), SignificanceResult[onp.ArrayND[np.float32]])
assert_type(jarque_bera(_f64_1d, axis=1), SignificanceResult[onp.ArrayND[np.float64]])
assert_type(jarque_bera(_f64_2d, axis=1), SignificanceResult[onp.ArrayND[np.float64]])

assert_type(jarque_bera(_py_f_1d, keepdims=True), SignificanceResult[onp.ArrayND[np.float64]])
assert_type(jarque_bera(_intp_1d, keepdims=True), SignificanceResult[onp.Array1D[np.float64]])
assert_type(jarque_bera(_f32_1d, keepdims=True), SignificanceResult[onp.Array1D[np.float32]])
assert_type(jarque_bera(_f64_1d, keepdims=True), SignificanceResult[onp.Array1D[np.float64]])
assert_type(jarque_bera(_f64_2d, keepdims=True), SignificanceResult[onp.Array2D[np.float64]])

# scoreatpercentile

assert_type(scoreatpercentile(_f64_1d, 50), np.float64)
assert_type(scoreatpercentile(_f64_1d, _py_f_1d), onp.Array1D[np.float64])
assert_type(scoreatpercentile(_f64_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(scoreatpercentile(_f64_1d, _f64_2d), onp.Array2D[np.float64])
assert_type(scoreatpercentile(_f64_1d, _f64_nd), onp.ArrayND[np.float64])

# percentileofscore

assert_type(percentileofscore(_f64_1d, 0.5), np.float64)
assert_type(percentileofscore(_f64_1d, _py_f_1d), onp.Array1D[np.float64])
assert_type(percentileofscore(_f64_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(percentileofscore(_f64_1d, _f64_2d), onp.Array2D[np.float64])
assert_type(percentileofscore(_f64_1d, _f64_nd), onp.ArrayND[np.float64])

# cumfreq, relfreq

assert_type(cumfreq(_f64_1d), CumfreqResult)
assert_type(relfreq(_f64_1d), RelfreqResult)

# obrientransform

assert_type(obrientransform(_f64_1d, _f64_1d), onp.Array2D[np.float64] | onp.Array1D[np.object_])

# sigmaclip

assert_type(sigmaclip(_f64_1d), SigmaclipResult)

# trimboth, trim1

assert_type(trimboth(_f64_1d, 0.1), onp.ArrayND[npc.integer | npc.floating])
assert_type(trim1(_f64_1d, 0.1), onp.ArrayND[npc.integer | npc.floating])

# trim_mean

assert_type(trim_mean(_f64_1d, 0.1), np.float64)
assert_type(trim_mean(_f64_nd, 0.1, axis=None), np.float64)
assert_type(trim_mean(_f64_nd, 0.1, keepdims=True), onp.ArrayND[np.float64])
assert_type(trim_mean(_f64_nd, 0.1), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]

# f_oneway

assert_type(f_oneway(_f64_1d, _f64_1d), F_onewayResult)

# alexandergovern

assert_type(alexandergovern(_f64_1d, _f64_1d), AlexanderGovernResult)

# pointbiserialr

assert_type(pointbiserialr(_bool_1d, _f64_1d, axis=None), SignificanceResult[np.float64])
assert_type(pointbiserialr(_bool_1d, _f64_1d), SignificanceResult[np.float64])
assert_type(pointbiserialr(_bool_2d, _f64_2d, axis=0), SignificanceResult[onp.Array1D[np.float64]])
assert_type(pointbiserialr(_bool_nd, _f64_nd, keepdims=True), SignificanceResult[onp.ArrayND[np.float64]])

# weightedtau

assert_type(weightedtau(_f64_1d, _f64_1d, axis=None), SignificanceResult[np.float64])
assert_type(weightedtau(_f64_1d, _f64_1d), SignificanceResult[np.float64])
assert_type(weightedtau(_f64_2d, _f64_2d, axis=0), SignificanceResult[onp.Array1D[np.float64]])
assert_type(weightedtau(_f64_3d, _f64_3d, axis=0), SignificanceResult[onp.Array2D[np.float64]])
assert_type(weightedtau(_f64_nd, _f64_nd, keepdims=True), SignificanceResult[onp.ArrayND[np.float64]])

# ttest_1samp

assert_type(ttest_1samp(_f64_1d, 0.0), TtestResult)

# ttest_ind_from_stats

assert_type(ttest_ind_from_stats(0.0, 1.0, 10, 0.0, 1.0, 10), Ttest_indResult)

# power_divergence

assert_type(power_divergence(_f64_1d), Power_divergenceResult[np.float64])
assert_type(power_divergence(_f64_nd, axis=None), Power_divergenceResult[np.float64])
assert_type(power_divergence(_f64_nd, keepdims=True), Power_divergenceResult[onp.ArrayND[np.float64]])

# chisquare

assert_type(chisquare(_f64_1d), Power_divergenceResult[np.float64])
assert_type(chisquare(_f64_nd, axis=None), Power_divergenceResult[np.float64])
assert_type(chisquare(_f64_nd, keepdims=True), Power_divergenceResult[onp.ArrayND[np.float64]])

# ks_1samp, ks_2samp, kstest

assert_type(ks_1samp(_f64_1d, lambda x: x), KstestResult)
assert_type(ks_2samp(_f64_1d, _f64_1d), KstestResult)
assert_type(kstest(_f64_1d, "norm"), KstestResult)

# tiecorrect

assert_type(tiecorrect(_intp_1d), float)

# ranksums

assert_type(ranksums(_f64_1d, _f64_1d), RanksumsResult)

# kruskal

assert_type(kruskal(_f64_1d, _f64_1d), KruskalResult)

# friedmanchisquare

assert_type(friedmanchisquare(_f64_1d, _f64_1d, _f64_1d), FriedmanchisquareResult)

# brunnermunzel

assert_type(brunnermunzel(_f64_1d, _f64_1d), BrunnerMunzelResult)

# combine_pvalues

assert_type(combine_pvalues(_py_f_1d), SignificanceResult[np.float64])
assert_type(combine_pvalues(_f32_1d), SignificanceResult[np.float32])
assert_type(combine_pvalues(_f64_1d), SignificanceResult[np.float64])
assert_type(combine_pvalues(_py_f_2d), SignificanceResult[onp.Array1D[np.float64]])
assert_type(combine_pvalues(_f32_2d), SignificanceResult[onp.Array1D[np.float32]])
assert_type(combine_pvalues(_f64_2d), SignificanceResult[onp.Array1D[np.float64]])

assert_type(combine_pvalues(_py_f_1d, axis=None), SignificanceResult[np.float64])
assert_type(combine_pvalues(_f32_1d, axis=None), SignificanceResult[np.float32])
assert_type(combine_pvalues(_f64_1d, axis=None), SignificanceResult[np.float64])
assert_type(combine_pvalues(_py_f_2d, axis=None), SignificanceResult[np.float64])
assert_type(combine_pvalues(_f32_2d, axis=None), SignificanceResult[np.float32])
assert_type(combine_pvalues(_f64_2d, axis=None), SignificanceResult[np.float64])

assert_type(combine_pvalues(_py_f_1d, keepdims=True), SignificanceResult[onp.ArrayND[np.float64]])
assert_type(combine_pvalues(_f32_1d, keepdims=True), SignificanceResult[onp.Array1D[np.float32]])
assert_type(combine_pvalues(_f64_1d, keepdims=True), SignificanceResult[onp.Array1D[np.float64]])
assert_type(combine_pvalues(_py_f_2d, keepdims=True), SignificanceResult[onp.ArrayND[np.float64]])
assert_type(combine_pvalues(_f32_2d, keepdims=True), SignificanceResult[onp.Array2D[np.float32]])
assert_type(combine_pvalues(_f64_2d, keepdims=True), SignificanceResult[onp.Array2D[np.float64]])

# fisher_exact

assert_type(fisher_exact(_f64_2d), SignificanceResult[float])

# quantile_test

assert_type(quantile_test(_f64_1d), QuantileTestResult)

# wasserstein_distance, wasserstein_distance_nd, energy_distance

assert_type(wasserstein_distance(_f64_1d, _f64_1d), np.float64)
assert_type(wasserstein_distance_nd(_f64_1d, _f64_1d), np.float64)
assert_type(energy_distance(_f64_1d, _f64_1d), np.float64)

# expectile

assert_type(expectile(_f64_1d), np.float64)

# iqr

assert_type(iqr(_f64_1d), npc.floating | onp.ArrayND[npc.floating])

# median_abs_deviation

assert_type(median_abs_deviation(_f64_1d), np.float64)
assert_type(median_abs_deviation(_f64_nd, axis=None), onp.ArrayND[np.float64])
assert_type(median_abs_deviation(_f64_nd, keepdims=True), onp.ArrayND[np.float64])

# zmap

assert_type(zmap(_f64_1d, _f64_1d), onp.Array1D[npc.floating])
assert_type(zmap(_f64_nd, _f64_nd), onp.ArrayND[npc.floating])

# siegelslopes

assert_type(siegelslopes(_f64_1d), SiegelslopesResult[np.float64])
assert_type(siegelslopes(_f64_nd, axis=None), SiegelslopesResult[np.float64])
assert_type(siegelslopes(_f64_2d, axis=0), SiegelslopesResult[onp.Array1D[np.float64]])
assert_type(siegelslopes(_f64_nd, keepdims=True), SiegelslopesResult[onp.ArrayND[np.float64]])

# theilslopes

assert_type(theilslopes(_f64_1d), TheilslopesResult[np.float64])
assert_type(theilslopes(_f64_nd, axis=None), TheilslopesResult[np.float64])
assert_type(theilslopes(_f64_2d, axis=0), TheilslopesResult[onp.Array1D[np.float64]])
assert_type(theilslopes(_f64_nd, keepdims=True), TheilslopesResult[onp.ArrayND[np.float64]])
