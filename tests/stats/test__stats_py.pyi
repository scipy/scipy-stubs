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

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f32_nd: onp.ArrayND[np.float32]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_py_f_1d: list[float]

###

# gmean, hmean, pmean

assert_type(gmean(_f64_1d), npc.integer | npc.floating | onp.ArrayND[npc.integer | npc.floating])
assert_type(hmean(_f64_1d), npc.integer | npc.floating | onp.ArrayND[npc.integer | npc.floating])
assert_type(pmean(_f64_1d, 2), npc.integer | npc.floating | onp.ArrayND[npc.integer | npc.floating])

# tmean, tvar, tmin, tmax, tstd, tsem

assert_type(tmean(_f64_1d), npc.floating | onp.ArrayND[npc.floating])
assert_type(tvar(_f64_1d), npc.floating | onp.ArrayND[npc.floating])
assert_type(tmin(_f64_1d), npc.integer | npc.floating | onp.ArrayND[npc.integer | npc.floating])
assert_type(tmax(_f64_1d), npc.integer | npc.floating | onp.ArrayND[npc.integer | npc.floating])
assert_type(tstd(_f64_1d), npc.floating | onp.ArrayND[npc.floating])
assert_type(tsem(_f64_1d), npc.floating | onp.ArrayND[npc.floating])

# gstd

assert_type(gstd(_f64_1d), np.float64)
assert_type(gstd(_f64_nd, axis=None), np.float64)
assert_type(gstd(_f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(gstd(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]

# describe

assert_type(describe(_f64_1d), DescribeResult)

# skewtest, kurtosistest, normaltest

assert_type(skewtest(_f64_1d), SkewtestResult)
assert_type(kurtosistest(_f64_1d), KurtosistestResult)
assert_type(normaltest(_f64_1d), NormaltestResult)

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

assert_type(combine_pvalues(_f64_1d), SignificanceResult)

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
