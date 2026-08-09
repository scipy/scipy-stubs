# type-tests for `stats/mstats.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.stats._mstats_basic import F_onewayResult, FriedmanchisquareResult, KruskalResult
from scipy.stats.mstats import (
    argstoarray,
    compare_medians_ms,
    count_tied_groups,
    describe,
    f_oneway,
    friedmanchisquare,
    hdmedian,
    hdquantiles,
    hdquantiles_sd,
    idealfourths,
    kendalltau,
    kendalltau_seasonal,
    kruskal,
    ks_2samp,
    kurtosis,
    kurtosistest,
    linregress,
    mannwhitneyu,
    median_cihs,
    mjci,
    mode,
    moment,
    mquantiles,
    mquantiles_cimj,
    normaltest,
    pearsonr,
    pointbiserialr,
    rsh,
    sen_seasonal_slopes,
    siegelslopes,
    skew,
    skewtest,
    spearmanr,
    theilslopes,
    tmax,
    tmean,
    tmin,
    trim,
    trimboth,
    trimmed_mean,
    trimmed_mean_ci,
    ttest_1samp,
    ttest_ind,
    ttest_rel,
    variation,
)

###

_py_b_1d: list[bool]
_py_b_2d: list[list[bool]]
_py_b_3d: list[list[list[bool]]]
_py_i_1d: list[int]
_py_i_2d: list[list[int]]
_py_i_3d: list[list[list[int]]]
_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_c_1d: list[complex]
_py_c_2d: list[list[complex]]

_b_nd: onp.ArrayND[np.bool]

_i8_1d: onp.Array1D[np.int8]
_i8_2d: onp.Array2D[np.int8]
_i8_3d: onp.Array3D[np.int8]
_i8_nd: onp.ArrayND[np.int8]

_i64_1d: onp.Array1D[np.int64]
_i64_2d: onp.Array2D[np.int64]
_i64_3d: onp.Array3D[np.int64]
_i64_nd: onp.ArrayND[np.int64]

_f16_1d: onp.Array1D[np.float16]
_f16_2d: onp.Array2D[np.float16]
_f16_nd: onp.ArrayND[np.float16]

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f32_3d: onp.Array3D[np.float32]
_f32_nd: onp.ArrayND[np.float32]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_f80_1d: onp.Array1D[np.float128]
_f80_2d: onp.Array2D[np.float128]
_f80_3d: onp.Array3D[np.float128]
_f80_nd: onp.ArrayND[np.float128]

_c64_1d: onp.Array1D[np.complex64]
_c64_2d: onp.Array2D[np.complex64]
_c64_3d: onp.Array3D[np.complex64]
_c64_nd: onp.ArrayND[np.complex64]

_c160_1d: onp.Array1D[np.complex256]
_c160_2d: onp.Array2D[np.complex256]

_c128_1d: onp.Array1D[np.complex128]
_c128_2d: onp.Array2D[np.complex128]
_c128_3d: onp.Array3D[np.complex128]
_c128_nd: onp.ArrayND[np.complex128]

_m_f32_nd: onp.MArray[np.float32]
_m_f64_nd: onp.MArray[np.float64]

###

# argstoarray
assert_type(argstoarray(_f64_nd), onp.MArray[np.float64])
assert_type(argstoarray(_f64_nd, _i8_nd), onp.MArray[np.float64])
assert_type(argstoarray(_py_f_1d, _py_i_1d, _f32_nd), onp.MArray[np.float64])

# find_repeats
# TODO

# count_tied_groups
assert_type(count_tied_groups(_f64_nd), dict[np.intp, np.intp | int])

# rankdata
# TODO

# mode
assert_type(mode(_py_f_2d, axis=None).mode, onp.Array1D[np.float64])
assert_type(mode(_f32_1d).mode, onp.Array1D[np.float64])
assert_type(mode(_f16_2d).count, onp.MArray2D[np.float64])
assert_type(mode(_f64_3d, axis=1).mode, onp.MArray3D[np.float64])
assert_type(mode(_py_i_2d, axis=None).mode, onp.Array1D[np.float64 | Any])
assert_type(mode(_i64_1d).mode, onp.Array1D[np.float64 | Any])
assert_type(mode(_i8_2d).count, onp.MArray2D[np.float64 | Any])
assert_type(mode(_f80_3d, axis=1).mode, onp.MArray3D[np.float64 | Any])
assert_type(mode(_m_f64_nd).mode, onp.ArrayND[np.float64 | Any])

# msign
# TODO

# pearsonr
assert_type(pearsonr(_py_i_1d, _f16_1d).statistic, np.float64)
assert_type(pearsonr(_f16_2d, _f64_2d).statistic, np.float64)
assert_type(pearsonr(_f32_2d, _f16_2d).statistic, np.float32)
assert_type(pearsonr(_f16_1d, _f32_1d).statistic, np.float32)
assert_type(pearsonr(_f16_2d, _f16_2d).statistic, npc.floating)
assert_type(pearsonr(_i64_1d, _f64_1d).pvalue, np.float64)

# spearmanr
assert_type(spearmanr(_py_i_1d, _py_i_1d).statistic, np.float64)
assert_type(spearmanr(_f64_nd, _f64_nd).statistic, np.float64)
assert_type(spearmanr(_f32_3d, _f32_3d, axis=None).statistic, np.float64)
assert_type(spearmanr(_py_f_1d, _py_f_1d, axis=0).statistic, np.float64)
assert_type(spearmanr(_i64_1d, _i64_1d, axis=1).statistic, np.float64)
assert_type(spearmanr(_py_i_2d, _py_i_2d, axis=0).statistic, onp.Array2D[np.float64])
assert_type(spearmanr(_f64_2d, _f64_2d, axis=1).statistic, onp.Array2D[np.float64])
assert_type(spearmanr(_m_f64_nd, _m_f64_nd, axis=0).statistic, onp.Array2D[np.float64] | Any)
assert_type(spearmanr(_f32_3d, _f32_3d, axis=0).statistic, onp.Array2D[np.float64] | Any)
assert_type(spearmanr(_f64_2d, axis=1).statistic, onp.Array2D[np.float64] | Any)

# kentalltau
assert_type(kendalltau(_py_i_1d, _f16_1d).statistic, np.float64)
assert_type(kendalltau(_f64_2d, _f32_2d, method="exact").pvalue, np.float64)

# kendalltau_seasonal
assert_type(kendalltau_seasonal(_f32_2d)["seasonal tau"], onp.MArray1D[np.float64])
assert_type(kendalltau_seasonal(_py_i_2d)["seasonal p-value"], onp.Array1D[np.float64])
assert_type(kendalltau_seasonal(_m_f64_nd)["chi2 total"], np.float64)

# pointbiserialr
assert_type(pointbiserialr(_py_b_1d, _py_i_1d).correlation, np.float64)
assert_type(pointbiserialr(_i8_2d, _f16_2d).pvalue, onp.MArray0D[np.float64])

# linregress
assert_type(linregress(_py_i_1d, _f16_1d).slope, np.float64)
assert_type(linregress(_f64_2d).intercept, np.float64)

# theilslopes
assert_type(theilslopes(_py_i_1d, _f16_1d).slope, np.float64)
assert_type(theilslopes(_f64_2d).intercept, np.float64)

# siegelslopes
assert_type(siegelslopes(_py_i_1d, _f16_1d).slope, np.float64)
assert_type(siegelslopes(_f64_2d).intercept, np.float64)

# sen_seasonal_slopes
assert_type(sen_seasonal_slopes(_py_i_2d).intra_slope, onp.MArray1D[np.float64])
assert_type(sen_seasonal_slopes(_i8_2d).inter_slope, np.float64)
assert_type(sen_seasonal_slopes(_f16_2d).inter_slope, np.float64)
assert_type(sen_seasonal_slopes(_f32_3d).inter_slope, np.float64)
assert_type(sen_seasonal_slopes(_m_f64_nd).inter_slope, np.float64)
assert_type(sen_seasonal_slopes(_py_c_2d).intra_slope, onp.MArray1D[np.complex128])
assert_type(sen_seasonal_slopes(_c64_2d).inter_slope, np.complex128)
assert_type(sen_seasonal_slopes(_f80_2d).intra_slope, onp.MArray1D[np.float128])
assert_type(sen_seasonal_slopes(_c160_2d).inter_slope, np.complex256)

# ttest_1samp
assert_type(ttest_1samp(_f32_3d, 0.5, axis=None).statistic, np.float64)
assert_type(ttest_1samp(_py_i_1d, 0.5).statistic, np.float64)
assert_type(ttest_1samp(_f16_2d, _f64_1d).statistic, onp.MArray1D[np.float64])
assert_type(ttest_1samp(_i8_3d, 0).pvalue, onp.MArray2D[np.float64])
assert_type(ttest_1samp(_m_f64_nd, 0.5).statistic, onp.MArray[np.float64] | Any)
assert_type(ttest_1samp(_py_c_2d, 0.5j, axis=None).statistic, np.complex128)
assert_type(ttest_1samp(_c64_1d, 0.5).statistic, np.complex128)
assert_type(ttest_1samp(_c128_2d, 0.5j).statistic, onp.MArray1D[np.complex128])
assert_type(ttest_1samp(_c128_2d, 0.5j).pvalue, onp.MArray1D[np.float64])
assert_type(ttest_1samp(_c64_3d, 0.5).statistic, onp.MArray2D[np.complex128])

# ttest_ind
assert_type(ttest_ind(_f32_3d, _i8_3d, axis=None).statistic, np.float64)
assert_type(ttest_ind(_py_i_1d, _f16_1d).statistic, np.float64)
assert_type(ttest_ind(_f16_2d, _f64_2d, equal_var=False).statistic, onp.MArray1D[np.float64])
assert_type(ttest_ind(_i8_2d, _f32_2d).pvalue, onp.MArray1D[np.float64])
assert_type(ttest_ind(_m_f64_nd, _f64_nd).statistic, onp.MArray[np.float64] | Any)
assert_type(ttest_ind(_py_c_2d, _py_f_2d, axis=None).statistic, np.complex128)
assert_type(ttest_ind(_f64_1d, _c64_1d).statistic, np.complex128)
assert_type(ttest_ind(_c128_2d, _f16_2d).statistic, onp.MArray1D[np.complex128])
assert_type(ttest_ind(_c128_2d, _f16_2d).pvalue, onp.MArray1D[np.float64])
assert_type(ttest_ind(_f32_2d, _c64_2d).statistic, onp.MArray1D[np.complex128])

# ttest_rel
assert_type(ttest_rel(_f32_3d, _i8_3d, axis=None).statistic, np.float64)
assert_type(ttest_rel(_py_i_1d, _f16_1d).statistic, np.float64)
assert_type(ttest_rel(_f16_2d, _f64_2d).statistic, onp.MArray1D[np.float64])
assert_type(ttest_rel(_i8_2d, _f32_2d).pvalue, onp.MArray1D[np.float64])
assert_type(ttest_rel(_m_f64_nd, _f64_nd).statistic, onp.MArray[np.float64] | Any)

# mannwhitneyu
assert_type(mannwhitneyu(_py_i_1d, _f16_1d).statistic, np.float64)
assert_type(mannwhitneyu(_f80_3d, _i8_2d, use_continuity=False).pvalue, np.float64)

# kruskal
assert_type(kruskal(_f64_1d), KruskalResult)
assert_type(kruskal(_f64_1d, _i8_1d), KruskalResult)
assert_type(kruskal(_py_f_1d, _f32_1d, _f16_1d), KruskalResult)

# ks_1samp
# TODO

# ks_2samp
assert_type(ks_2samp(_py_f_1d, _f64_1d).statistic, np.float64)
assert_type(ks_2samp(_i8_1d, _f32_1d).statistic_sign, np.int8)
assert_type(ks_2samp(_f64_2d, _i8_2d).statistic, onp.Array1D[np.float64])
assert_type(ks_2samp(_f16_2d, _f64_1d).statistic_sign, onp.Array1D[np.int8])
assert_type(ks_2samp(_m_f64_nd, _f64_nd).statistic, np.float64 | Any)

# ktest
# TODO

# trima
# TODO

# trimr
# TODO

# trim
assert_type(trim(_py_b_1d), onp.MArray1D[np.bool])
assert_type(trim(_py_b_2d), onp.MArray[np.bool])
assert_type(trim(_py_i_1d), onp.MArray1D[np.int_])
assert_type(trim(_py_i_2d), onp.MArray[np.int_])
assert_type(trim(_py_f_1d), onp.MArray1D[np.float64])
assert_type(trim(_py_f_2d), onp.MArray[np.float64])
assert_type(trim(_py_c_1d), onp.MArray1D[np.complex128])
assert_type(trim(_py_c_2d), onp.MArray[np.complex128])
assert_type(trim(_i64_1d), onp.MArray1D[np.int64])
assert_type(trim(_f16_2d), onp.MArray2D[np.float16])
assert_type(trim(_f32_3d), onp.MArray3D[np.float32])
assert_type(trim(_f80_2d), onp.MArray2D[np.float128])
assert_type(trim(_c64_nd), onp.MArray[np.complex64])
assert_type(trim(_m_f32_nd), onp.MArray[np.float32])
assert_type(trim(_f64_2d, (0.1, 0.1), (True, True), True, 0), onp.MArray2D[np.float64])

# trimboth
assert_type(trimboth(_py_b_1d), onp.MArray1D[np.bool])
assert_type(trimboth(_py_b_2d), onp.MArray[np.bool])
assert_type(trimboth(_py_i_1d), onp.MArray1D[np.int_])
assert_type(trimboth(_py_i_2d), onp.MArray[np.int_])
assert_type(trimboth(_py_f_1d), onp.MArray1D[np.float64])
assert_type(trimboth(_py_c_2d), onp.MArray[np.complex128])
assert_type(trimboth(_i64_1d), onp.MArray1D[np.int64])
assert_type(trimboth(_f16_2d), onp.MArray2D[np.float16])
assert_type(trimboth(_f32_3d), onp.MArray3D[np.float32])
assert_type(trimboth(_f80_2d), onp.MArray2D[np.float128])
assert_type(trimboth(_c64_nd), onp.MArray[np.complex64])
assert_type(trimboth(_m_f32_nd), onp.MArray[np.float32])
assert_type(trimboth(_f64_2d, 0.1, (True, True), 0), onp.MArray2D[np.float64])

# trimtail
# TODO

# trimmed_mean
assert_type(trimmed_mean(_py_i_1d), np.float64)
assert_type(trimmed_mean(_py_c_2d), np.complex128)
assert_type(trimmed_mean(_i64_2d), np.float64)
assert_type(trimmed_mean(_f16_1d), np.float16)
assert_type(trimmed_mean(_f32_2d), np.float64)
assert_type(trimmed_mean(_c64_nd), np.complex128)
assert_type(trimmed_mean(_f80_3d), np.float128)

assert_type(trimmed_mean(_py_f_1d, axis=0), np.float64)
assert_type(trimmed_mean(_f16_1d, axis=0), np.float16)
assert_type(trimmed_mean(_c128_1d, axis=0), np.complex128)

assert_type(trimmed_mean(_i64_2d, axis=0), onp.MArray1D[np.float64])
assert_type(trimmed_mean(_f32_2d, axis=1), onp.MArray1D[np.float64])
assert_type(trimmed_mean(_c64_2d, axis=0), onp.MArray1D[np.complex128])
assert_type(trimmed_mean(_f80_2d, axis=0), onp.MArray1D[np.float128])

assert_type(trimmed_mean(_f32_3d, axis=0), onp.MArray2D[np.float64])
assert_type(trimmed_mean(_f80_3d, axis=0), onp.MArray2D[np.float128])

assert_type(trimmed_mean(_f64_nd, axis=0), onp.MArray[np.float64] | Any)
assert_type(trimmed_mean(_m_f32_nd, axis=0), onp.MArray[np.float64] | Any)
assert_type(trimmed_mean(_f64_2d, (0.2, 0.2), (1, 1), False, 0), onp.MArray[np.float64] | Any)

# trimmed_var
# TODO

# trimmed_std
# TODO

# trimmed_stde
# TODO

# tmean
assert_type(tmean(_py_i_1d), np.float64)
assert_type(tmean(_py_f_1d), np.float64)
assert_type(tmean(_py_c_1d), np.complex128)
assert_type(tmean(_py_i_2d), np.float64)
assert_type(tmean(_py_f_2d), np.float64)
assert_type(tmean(_py_c_2d), np.complex128)
assert_type(tmean(_i64_1d), np.float64)
assert_type(tmean(_i64_2d), np.float64)
assert_type(tmean(_i64_nd), np.float64)
assert_type(tmean(_f16_1d), np.float16)
assert_type(tmean(_f16_nd), np.float16)
assert_type(tmean(_f32_1d), np.float64)
assert_type(tmean(_f32_2d), np.float64)
assert_type(tmean(_f32_nd), np.float64)
assert_type(tmean(_f64_1d), np.float64)
assert_type(tmean(_f64_nd), np.float64)
assert_type(tmean(_c64_1d), np.complex128)
assert_type(tmean(_c64_nd), np.complex128)
assert_type(tmean(_c128_1d), np.complex128)
assert_type(tmean(_c128_nd), np.complex128)
assert_type(tmean(_m_f32_nd), np.float64)
assert_type(tmean(_m_f64_nd), np.float64)

assert_type(tmean(_py_i_1d, axis=0), np.float64)
assert_type(tmean(_py_f_1d, axis=0), np.float64)
assert_type(tmean(_py_c_1d, axis=0), np.complex128)
assert_type(tmean(_i64_1d, axis=0), np.float64)
assert_type(tmean(_f16_1d, axis=0), np.float16)
assert_type(tmean(_f32_1d, axis=0), np.float64)
assert_type(tmean(_f64_1d, axis=0), np.float64)
assert_type(tmean(_c64_1d, axis=0), np.complex128)
assert_type(tmean(_c128_1d, axis=0), np.complex128)

assert_type(tmean(_py_i_2d, axis=0), onp.MArray1D[np.float64])
assert_type(tmean(_py_f_2d, axis=0), onp.MArray1D[np.float64])
assert_type(tmean(_py_c_2d, axis=0), onp.MArray1D[np.complex128])
assert_type(tmean(_i64_2d, axis=0), onp.MArray1D[np.float64])
assert_type(tmean(_f16_2d, axis=0), onp.MArray1D[np.float16])
assert_type(tmean(_f32_2d, axis=0), onp.MArray1D[np.float64])
assert_type(tmean(_f64_2d, axis=0), onp.MArray1D[np.float64])
assert_type(tmean(_c64_2d, axis=0), onp.MArray1D[np.complex128])
assert_type(tmean(_c128_2d, axis=0), onp.MArray1D[np.complex128])

assert_type(tmean(_i64_nd, axis=0), onp.MArray[np.float64] | Any)
assert_type(tmean(_f16_nd, axis=0), onp.MArray[np.float16] | Any)
assert_type(tmean(_f32_nd, axis=0), onp.MArray[np.float64] | Any)
assert_type(tmean(_f64_nd, axis=0), onp.MArray[np.float64] | Any)
assert_type(tmean(_c64_nd, axis=0), onp.MArray[np.complex128] | Any)
assert_type(tmean(_c128_nd, axis=0), onp.MArray[np.complex128] | Any)
assert_type(tmean(_m_f32_nd, axis=0), onp.MArray[np.float64] | Any)
assert_type(tmean(_m_f64_nd, axis=0), onp.MArray[np.float64] | Any)

assert_type(tmean(_f32_3d, axis=0), onp.MArray[np.float64] | Any)
assert_type(tmean(_f64_nd, (0.0, 1.0), (True, True), 0), onp.MArray[np.float64] | Any)

# tvar
# TODO

# tmin
assert_type(tmin(_py_i_1d), np.int_ | onp.MArray[np.int_])
assert_type(tmin(_f32_1d), np.float32 | onp.MArray[np.float32])

# tmax
assert_type(tmax(_py_i_1d), np.int_ | onp.MArray[np.int_])
assert_type(tmax(_f32_1d), np.float32 | onp.MArray[np.float32])

# tsem
# TODO

# winsorize
# TODO

# moment
assert_type(moment(_f64_2d, 2, None), np.float64)
assert_type(moment(_c128_2d, axis=None), np.complex128)
assert_type(moment(_f80_2d, 2, axis=None), np.float128)

assert_type(moment(_py_i_1d), np.float64)
assert_type(moment(_f32_1d, 2), np.float64)
assert_type(moment(_py_c_1d), np.complex128)
assert_type(moment(_f80_1d), np.float128)

assert_type(moment(_f16_2d, 2), onp.MArray1D[np.float64])
assert_type(moment(_c64_2d, 3), onp.MArray1D[np.complex128])
assert_type(moment(_f80_2d), onp.MArray1D[np.float128])

assert_type(moment(_f32_3d, 2), onp.MArray2D[np.float64])
assert_type(moment(_f80_3d), onp.MArray2D[np.float128])

assert_type(moment(_m_f64_nd, 2, axis=0), onp.MArray[np.float64] | Any)
assert_type(moment(_i64_nd, 2), onp.MArray[np.float64] | Any)

assert_type(moment(_f64_2d, [2, 3]), onp.MArray[np.float64])
assert_type(moment(_c64_1d, [2, 3], None), onp.MArray[np.complex128])

# variation
assert_type(variation(_f64_2d, None), np.float64)
assert_type(variation(_c64_2d, axis=None), np.complex64)
assert_type(variation(_f80_2d, None), np.float128)

assert_type(variation(_py_i_1d), np.float64)
assert_type(variation(_py_c_1d), np.complex128)
assert_type(variation(_i64_1d), np.float64)
assert_type(variation(_f16_1d), np.float16)
assert_type(variation(_f32_1d), np.float32)
assert_type(variation(_c128_1d), np.complex128)
assert_type(variation(_f80_1d), np.float128)

assert_type(variation(_py_f_2d), onp.MArray1D[np.float64])
assert_type(variation(_f16_2d), onp.MArray1D[np.float16])
assert_type(variation(_c64_2d, 1), onp.MArray1D[np.complex64])
assert_type(variation(_f80_2d), onp.MArray1D[np.float128])

assert_type(variation(_f32_3d), onp.MArray2D[np.float32])
assert_type(variation(_f80_3d), onp.MArray2D[np.float128])

assert_type(variation(_i64_nd, 0, 1), onp.MArray[np.float64] | Any)
assert_type(variation(_m_f32_nd), onp.MArray[np.float32] | Any)

# skew
assert_type(skew(_f64_2d, None), onp.MArray0D[np.float64])
assert_type(skew(_c128_2d, axis=None), onp.MArray0D[np.complex128])
assert_type(skew(_f80_2d, None), onp.MArray0D[np.float128])

assert_type(skew(_py_i_1d), onp.MArray0D[np.float64])
assert_type(skew(_f32_1d), onp.MArray0D[np.float64])
assert_type(skew(_py_c_1d), onp.MArray0D[np.complex128])
assert_type(skew(_f80_1d), onp.MArray0D[np.float128])

assert_type(skew(_f16_2d), onp.MArray1D[np.float64])
assert_type(skew(_c64_2d, 1), onp.MArray1D[np.complex128])
assert_type(skew(_f80_2d), onp.MArray1D[np.float128])

assert_type(skew(_f32_3d), onp.MArray2D[np.float64])
assert_type(skew(_f80_3d), onp.MArray2D[np.float128])

assert_type(skew(_m_f64_nd, 0, False), onp.MArray[np.float64] | Any)
assert_type(skew(_c64_nd), onp.MArray[np.complex128] | Any)

# kurtosis
assert_type(kurtosis(_f64_2d, None), np.float64)
assert_type(kurtosis(_c128_2d, axis=None), np.complex128)
assert_type(kurtosis(_f80_2d, None), np.float128)

assert_type(kurtosis(_py_i_1d), np.float64)
assert_type(kurtosis(_f32_1d), np.float64)
assert_type(kurtosis(_py_c_1d), np.complex128)
assert_type(kurtosis(_f80_1d), np.float128)

assert_type(kurtosis(_f16_2d), onp.MArray1D[np.float64])
assert_type(kurtosis(_c64_2d, 1), onp.MArray1D[np.complex128])
assert_type(kurtosis(_f80_2d), onp.MArray1D[np.float128])

assert_type(kurtosis(_f32_3d), onp.MArray2D[np.float64])
assert_type(kurtosis(_f80_3d), onp.MArray2D[np.float128])

assert_type(kurtosis(_m_f64_nd, 0, True, False), onp.MArray[np.float64] | Any)
assert_type(kurtosis(_c64_nd), onp.MArray[np.complex128] | Any)

assert_type(kurtosis(_f64_1d, 0, False), onp.MArray[np.float64] | Any)
assert_type(kurtosis(_f64_2d, None, False), onp.MArray[np.float64] | Any)

# describe
assert_type(describe(_py_b_2d, None).mean, np.float64)
assert_type(describe(_i8_2d, None).mean, np.float64)
assert_type(describe(_py_i_2d, None).mean, np.float64)
assert_type(describe(_f32_2d, None).mean, np.float32)
assert_type(describe(_f64_2d, None).mean, np.float64)
assert_type(describe(_f80_2d, None).mean, np.float128)
assert_type(describe(_c64_2d, None).mean, np.complex64)
assert_type(describe(_c128_2d, None).mean, np.complex128)

assert_type(describe(_i8_nd).variance, onp.MArray[np.float64] | Any)
assert_type(describe(_f32_nd).variance, onp.MArray[np.float32] | Any)
assert_type(describe(_f64_nd).variance, onp.MArray[np.float64] | Any)
assert_type(describe(_f80_nd).variance, onp.MArray[np.longdouble] | Any)
assert_type(describe(_c64_nd).variance, onp.MArray[np.float32] | Any)
assert_type(describe(_c128_nd).variance, onp.MArray[np.float64] | Any)

assert_type(describe(_py_b_1d).minmax[0], onp.MArray0D[np.bool])
assert_type(describe(_i8_1d).minmax[0], onp.MArray0D[np.int8])
assert_type(describe(_py_i_1d).minmax[0], onp.MArray0D[np.int_])
assert_type(describe(_f32_1d).minmax[0], onp.MArray0D[np.float32])
assert_type(describe(_f64_1d).minmax[0], onp.MArray0D[np.float64])
assert_type(describe(_f80_1d).minmax[0], onp.MArray0D[np.float128])
assert_type(describe(_c64_1d).minmax[0], onp.MArray0D[np.complex64])
assert_type(describe(_c128_1d).minmax[0], onp.MArray0D[np.complex128])

assert_type(describe(_py_b_2d).skewness, onp.MArray1D[np.float64])
assert_type(describe(_i8_2d).skewness, onp.MArray1D[np.float64])
assert_type(describe(_py_i_2d).skewness, onp.MArray1D[np.float64])
assert_type(describe(_f32_2d).skewness, onp.MArray1D[np.float64])
assert_type(describe(_f64_2d).skewness, onp.MArray1D[np.float64])
assert_type(describe(_f80_2d).skewness, onp.MArray1D[np.float128])
assert_type(describe(_c64_2d).skewness, onp.MArray1D[np.complex128])
assert_type(describe(_c128_2d).skewness, onp.MArray1D[np.complex128])

assert_type(describe(_py_b_3d).kurtosis, onp.MArray[np.float64] | Any)
assert_type(describe(_i8_3d).kurtosis, onp.MArray[np.float64] | Any)
assert_type(describe(_py_i_3d).kurtosis, onp.MArray[np.float64] | Any)
assert_type(describe(_f32_3d).kurtosis, onp.MArray[np.float64] | Any)
assert_type(describe(_f64_3d).kurtosis, onp.MArray[np.float64] | Any)
assert_type(describe(_f80_3d).kurtosis, onp.MArray[np.float128] | Any)
assert_type(describe(_c64_3d).kurtosis, onp.MArray[np.complex128] | Any)
assert_type(describe(_c128_3d).kurtosis, onp.MArray[np.complex128] | Any)

assert_type(describe(_f64_1d).nobs, onp.Array0D[np.int_])
assert_type(describe(_f64_2d).nobs, onp.Array1D[np.int_])
assert_type(describe(_f64_nd).nobs, onp.ArrayND[np.int_])
assert_type(describe(_m_f32_nd).mean, onp.MArray[np.float32] | Any)
assert_type(describe(_i8_2d).minmax[0], onp.MArray1D[np.int8])
assert_type(describe(_b_nd).kurtosis, onp.MArray[np.float64] | Any)

# stde_median
# TODO

# skewtest
assert_type(skewtest(_f32_3d, axis=None).statistic, np.float64)
assert_type(skewtest(_py_c_2d, axis=None).statistic, np.complex128)
assert_type(skewtest(_py_i_1d).statistic, np.float64)
assert_type(skewtest(_f16_2d).statistic, onp.MArray1D[np.float64])
assert_type(skewtest(_m_f64_nd).statistic, onp.MArray[np.float64] | Any)
assert_type(skewtest(_c64_1d).statistic, np.complex128)
assert_type(skewtest(_c128_2d).statistic, onp.MArray1D[np.complex128])
assert_type(skewtest(_i8_3d).pvalue, onp.Array2D[np.float64])
assert_type(skewtest(_c128_2d).pvalue, onp.Array1D[np.float64])
assert_type(skewtest(_c64_3d).statistic, onp.MArray2D[np.complex128])

# kurtosistest
assert_type(kurtosistest(_f32_3d, axis=None).statistic, np.float64)
assert_type(kurtosistest(_py_i_1d).statistic, np.float64)
assert_type(kurtosistest(_f16_2d).statistic, onp.MArray1D[np.float64])
assert_type(kurtosistest(_i8_3d).pvalue, onp.Array2D[np.float64])
assert_type(kurtosistest(_m_f64_nd).statistic, onp.MArray[np.float64] | Any)
assert_type(kurtosistest(_py_c_2d, axis=None).statistic, np.complex128)
assert_type(kurtosistest(_c64_1d).statistic, np.complex128)
assert_type(kurtosistest(_c128_2d).statistic, onp.MArray1D[np.complex128])
assert_type(kurtosistest(_c128_2d).pvalue, onp.Array1D[np.float64])
assert_type(kurtosistest(_c64_3d).statistic, onp.MArray2D[np.complex128])

# normaltest
assert_type(normaltest(_f64_nd, axis=None).statistic, np.float64)
assert_type(normaltest(_py_i_1d).statistic, np.float64)
assert_type(normaltest(_f64_1d).pvalue, np.float64)
assert_type(normaltest(_i64_2d).statistic, onp.MArray1D[np.float64])
assert_type(normaltest(_f32_2d, axis=1).pvalue, onp.Array1D[np.float64])
assert_type(normaltest(_f32_3d).statistic, onp.MArray2D[np.float64])
assert_type(normaltest(_f32_nd).statistic, onp.MArray[np.float64] | Any)
assert_type(normaltest(_m_f64_nd, axis=0).statistic, onp.MArray[np.float64] | Any)
assert_type(normaltest(_f64_nd, axis=0).pvalue, onp.ArrayND[np.float64] | Any)

# mquantiles
assert_type(mquantiles(_py_i_2d), onp.Array1D[np.float64])
assert_type(mquantiles(_f16_1d, 0.5), onp.Array1D[np.float64])
assert_type(mquantiles(_f80_2d), onp.Array1D[np.float128])
assert_type(mquantiles(_f32_1d, axis=0), onp.MArray1D[np.float64])
assert_type(mquantiles(_f80_1d, axis=0), onp.MArray1D[np.float128])
assert_type(mquantiles(_i64_2d, axis=1), onp.MArray2D[np.float64])
assert_type(mquantiles(_f80_2d, axis=0), onp.MArray2D[np.float128])
assert_type(mquantiles(_f64_nd, axis=0), onp.MArray[np.float64] | Any)
assert_type(mquantiles(_f80_nd, axis=0), onp.MArray[np.float128] | Any)

# scoreatpercentile
# TODO

# obrientransform
# TODO

# sem
# TODO

# f_oneway
assert_type(f_oneway(_f64_1d), F_onewayResult)
assert_type(f_oneway(_f64_1d, _i8_1d), F_onewayResult)
assert_type(f_oneway(_py_f_1d, _f32_1d, _f16_1d), F_onewayResult)

# friedmanchisquare
assert_type(friedmanchisquare(_f64_1d, _i8_1d, _f32_1d), FriedmanchisquareResult)
assert_type(friedmanchisquare(_py_f_1d, _py_i_1d, _f16_1d, _f64_1d), FriedmanchisquareResult)

# brunnermunzel
# TODO

###

# hdquantiles
assert_type(hdquantiles(_f64_3d), onp.MArray1D[np.float64])
assert_type(hdquantiles(_i8_2d, var=True), onp.MArray2D[np.float64])
assert_type(hdquantiles(_f32_2d, (0.1, 0.9), None, True), onp.MArray2D[np.float64])
assert_type(hdquantiles(_py_f_1d, axis=0), onp.MArray1D[np.float64])
assert_type(hdquantiles(_f64_1d, axis=0, var=True), onp.MArray2D[np.float64])
assert_type(hdquantiles(_f64_2d, axis=0), onp.MArray2D[np.float64])
assert_type(hdquantiles(_f16_2d, axis=1, var=True), onp.MArray3D[np.float64])
assert_type(hdquantiles(_f64_nd, axis=0), onp.MArray[np.float64])
assert_type(hdquantiles(_f64_nd, axis=0, var=True), onp.MArray[np.float64])

# hdmedian
assert_type(hdmedian(_py_f_1d), onp.MArray0D[np.float64])
assert_type(hdmedian(_f32_3d, None), onp.MArray0D[np.float64])
assert_type(hdmedian(_i8_2d), onp.MArray1D[np.float64])
assert_type(hdmedian(_f64_nd, 0), onp.MArray[np.float64])
assert_type(hdmedian(_f64_nd, 0, True), onp.MArray[np.float64])
assert_type(hdmedian(_f64_nd, var=True), onp.MArray[np.float64])
assert_type(hdmedian(_f16_1d, var=True), onp.MArray1D[np.float64])
assert_type(hdmedian(_f16_1d, 0, True), onp.MArray1D[np.float64])
assert_type(hdmedian(_f64_2d, None, True), onp.MArray1D[np.float64])
assert_type(hdmedian(_f64_2d, 0, True), onp.MArray2D[np.float64])
assert_type(hdmedian(_py_i_2d, var=True), onp.MArray2D[np.float64])

# hdquantiles_sd
assert_type(hdquantiles_sd(_py_i_1d), onp.MArray1D[np.float64])
assert_type(hdquantiles_sd(_f80_3d, 0.5), onp.MArray1D[np.float64])
assert_type(hdquantiles_sd(_f32_2d, axis=1), onp.MArray1D[np.float64])

# trimmed_mean_ci
assert_type(trimmed_mean_ci(_py_f_1d), onp.Array1D[np.float64])
assert_type(trimmed_mean_ci(_f32_3d), onp.Array1D[np.float64])
assert_type(trimmed_mean_ci(_f80_1d), onp.Array1D[np.longdouble])
assert_type(trimmed_mean_ci(_f16_1d, axis=0), onp.Array1D[np.float64])
assert_type(trimmed_mean_ci(_f80_1d, axis=0), onp.Array1D[np.longdouble])
assert_type(trimmed_mean_ci(_i8_2d, axis=0), onp.Array2D[np.float64])
assert_type(trimmed_mean_ci(_f80_2d, axis=1), onp.Array2D[np.longdouble])
assert_type(trimmed_mean_ci(_f64_nd, axis=0), onp.ArrayND[np.float64] | Any)

# mjci
assert_type(mjci(_py_f_1d), onp.Array1D[np.float64])
assert_type(mjci(_i8_2d, 0.5), onp.Array1D[np.float64])
assert_type(mjci(_f16_1d, axis=0), onp.MArray1D[np.float64])
assert_type(mjci(_f80_2d, axis=1), onp.MArray2D[np.float64])
assert_type(mjci(_f64_nd, axis=0), onp.ArrayND[np.float64] | Any)

# mquantiles_cimj
assert_type(mquantiles_cimj(_py_f_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(mquantiles_cimj(_f80_1d), tuple[onp.Array1D[np.longdouble], onp.Array1D[np.longdouble]])
assert_type(mquantiles_cimj(_f64_nd, axis=0), tuple[onp.ArrayND[np.float64] | Any, onp.ArrayND[np.float64] | Any])
assert_type(mquantiles_cimj(_f16_1d, axis=0), tuple[onp.MArray1D[np.float64], onp.MArray1D[np.float64]])
assert_type(mquantiles_cimj(_f80_1d, axis=0), tuple[onp.MArray1D[np.longdouble], onp.MArray1D[np.longdouble]])
assert_type(mquantiles_cimj(_i8_2d, 0.5, axis=0), tuple[onp.MArray2D[np.float64], onp.MArray2D[np.float64]])
assert_type(mquantiles_cimj(_f80_2d, axis=1), tuple[onp.MArray2D[np.longdouble], onp.MArray2D[np.longdouble]])

# median_cihs
assert_type(median_cihs(_py_f_1d), tuple[np.float64, np.float64])
assert_type(median_cihs(_f32_3d), tuple[np.float64, np.float64])
assert_type(median_cihs(_f80_1d), tuple[np.longdouble, np.longdouble])
assert_type(median_cihs(_f64_nd, axis=0), onp.MArray[np.float64] | Any)
assert_type(median_cihs(_f16_1d, axis=0), onp.MArray1D[np.float64])
assert_type(median_cihs(_f80_1d, axis=0), onp.MArray1D[np.longdouble])
assert_type(median_cihs(_i8_2d, axis=0), onp.MArray2D[np.float64])
assert_type(median_cihs(_f80_2d, axis=1), onp.MArray2D[np.longdouble])

# compare_medians_ms
assert_type(compare_medians_ms(_py_f_1d, _i8_1d), np.float64)
assert_type(compare_medians_ms(_f32_3d, _f64_3d), np.float64)
assert_type(compare_medians_ms(_f16_1d, _f64_1d, 0), np.float64)
assert_type(compare_medians_ms(_i8_2d, _f32_2d, 0), onp.Array1D[np.float64])
assert_type(compare_medians_ms(_f64_nd, _f64_nd, 0), onp.ArrayND[np.float64] | Any)

# idealfourths
assert_type(idealfourths(_f16_1d), list[np.float16])
assert_type(idealfourths(_f80_2d), list[np.float128])
assert_type(idealfourths(_py_i_2d), list[np.float64])
assert_type(idealfourths(_f64_nd, 0), onp.MArray[np.float64] | Any)
assert_type(idealfourths(_f32_1d, 0), onp.MArray1D[np.float32])
assert_type(idealfourths(_i8_1d, 0), onp.MArray1D[np.float64])
assert_type(idealfourths(_f80_2d, 1), onp.MArray2D[np.float128])
assert_type(idealfourths(_py_f_2d, 0), onp.MArray2D[np.float64])

# rsh
assert_type(rsh(_py_i_1d), onp.MArray1D[np.float64])
assert_type(rsh(_f32_1d, 0.5), onp.MArray1D[np.float64])
assert_type(rsh(_m_f64_nd, _f64_1d), onp.MArray1D[np.float64])
assert_type(rsh(_f80_1d), onp.MArray1D[np.longdouble])
