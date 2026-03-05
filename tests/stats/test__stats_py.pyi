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

_f1d: onp.Array1D[np.float64]
_f2d: onp.Array2D[np.float64]
_f3d: onp.Array3D[np.float64]
_fnd: onp.ArrayND[np.float64]

_b1d: onp.Array1D[np.bool_]
_b2d: onp.Array2D[np.bool_]
_bnd: onp.ArrayND[np.bool_]

_i1d: onp.Array1D[np.intp]

_py_f_1d: list[float]

###
# gmean, hmean, pmean

assert_type(gmean(_f1d), npc.integer | npc.floating | onp.ArrayND[npc.integer | npc.floating])
assert_type(hmean(_f1d), npc.integer | npc.floating | onp.ArrayND[npc.integer | npc.floating])
assert_type(pmean(_f1d, 2), npc.integer | npc.floating | onp.ArrayND[npc.integer | npc.floating])

###
# tmean, tvar, tmin, tmax, tstd, tsem

assert_type(tmean(_f1d), npc.floating | onp.ArrayND[npc.floating])
assert_type(tvar(_f1d), npc.floating | onp.ArrayND[npc.floating])
assert_type(tmin(_f1d), npc.integer | npc.floating | onp.ArrayND[npc.integer | npc.floating])
assert_type(tmax(_f1d), npc.integer | npc.floating | onp.ArrayND[npc.integer | npc.floating])
assert_type(tstd(_f1d), npc.floating | onp.ArrayND[npc.floating])
assert_type(tsem(_f1d), npc.floating | onp.ArrayND[npc.floating])

###
# gstd

assert_type(gstd(_f1d), np.float64)
assert_type(gstd(_fnd, axis=None), np.float64)
assert_type(gstd(_fnd, keepdims=True), onp.ArrayND[np.float64])
assert_type(gstd(_fnd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]

###
# describe

assert_type(describe(_f1d), DescribeResult)

###
# skewtest, kurtosistest, normaltest

assert_type(skewtest(_f1d), SkewtestResult)
assert_type(kurtosistest(_f1d), KurtosistestResult)
assert_type(normaltest(_f1d), NormaltestResult)

###
# jarque_bera

assert_type(jarque_bera(_f1d), SignificanceResult)

###
# scoreatpercentile

assert_type(scoreatpercentile(_f1d, 50), npc.floating | onp.ArrayND[npc.floating])

###
# percentileofscore

assert_type(percentileofscore(_f1d, 0.5), np.float64)

###
# cumfreq, relfreq

assert_type(cumfreq(_f1d), CumfreqResult)
assert_type(relfreq(_f1d), RelfreqResult)

###
# obrientransform

assert_type(obrientransform(_f1d, _f1d), onp.Array2D[npc.floating] | onp.Array1D[np.object_])

###
# sigmaclip

assert_type(sigmaclip(_f1d), SigmaclipResult)

###
# trimboth, trim1

assert_type(trimboth(_f1d, 0.1), onp.ArrayND[npc.integer | npc.floating])
assert_type(trim1(_f1d, 0.1), onp.ArrayND[npc.integer | npc.floating])

###
# trim_mean

assert_type(trim_mean(_f1d, 0.1), np.float64)
assert_type(trim_mean(_fnd, 0.1, axis=None), np.float64)
assert_type(trim_mean(_fnd, 0.1, keepdims=True), onp.ArrayND[np.float64])
assert_type(trim_mean(_fnd, 0.1), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]

###
# f_oneway

assert_type(f_oneway(_f1d, _f1d), F_onewayResult)

###
# alexandergovern

assert_type(alexandergovern(_f1d, _f1d), AlexanderGovernResult)

###
# pointbiserialr

assert_type(pointbiserialr(_b1d, _f1d, axis=None), SignificanceResult[np.float64])
assert_type(pointbiserialr(_b1d, _f1d), SignificanceResult[np.float64])
assert_type(pointbiserialr(_b2d, _f2d, axis=0), SignificanceResult[onp.Array1D[np.float64]])
assert_type(pointbiserialr(_bnd, _fnd, keepdims=True), SignificanceResult[onp.ArrayND[np.float64]])

###
# weightedtau

assert_type(weightedtau(_f1d, _f1d, axis=None), SignificanceResult[np.float64])
assert_type(weightedtau(_f1d, _f1d), SignificanceResult[np.float64])
assert_type(weightedtau(_f2d, _f2d, axis=0), SignificanceResult[onp.Array1D[np.float64]])
assert_type(weightedtau(_f3d, _f3d, axis=0), SignificanceResult[onp.Array2D[np.float64]])
assert_type(weightedtau(_fnd, _fnd, keepdims=True), SignificanceResult[onp.ArrayND[np.float64]])

###
# ttest_1samp

assert_type(ttest_1samp(_f1d, 0.0), TtestResult)

###
# ttest_ind_from_stats

assert_type(ttest_ind_from_stats(0.0, 1.0, 10, 0.0, 1.0, 10), Ttest_indResult)

###
# power_divergence

assert_type(power_divergence(_f1d), Power_divergenceResult[np.float64])
assert_type(power_divergence(_fnd, axis=None), Power_divergenceResult[np.float64])
assert_type(power_divergence(_fnd, keepdims=True), Power_divergenceResult[onp.ArrayND[np.float64]])

###
# chisquare

assert_type(chisquare(_f1d), Power_divergenceResult[np.float64])
assert_type(chisquare(_fnd, axis=None), Power_divergenceResult[np.float64])
assert_type(chisquare(_fnd, keepdims=True), Power_divergenceResult[onp.ArrayND[np.float64]])

###
# ks_1samp, ks_2samp, kstest

assert_type(ks_1samp(_f1d, lambda x: x), KstestResult)
assert_type(ks_2samp(_f1d, _f1d), KstestResult)
assert_type(kstest(_f1d, "norm"), KstestResult)

###
# tiecorrect

assert_type(tiecorrect(_i1d), float)

###
# ranksums

assert_type(ranksums(_f1d, _f1d), RanksumsResult)

###
# kruskal

assert_type(kruskal(_f1d, _f1d), KruskalResult)

###
# friedmanchisquare

assert_type(friedmanchisquare(_f1d, _f1d, _f1d), FriedmanchisquareResult)

###
# brunnermunzel

assert_type(brunnermunzel(_f1d, _f1d), BrunnerMunzelResult)

###
# combine_pvalues

assert_type(combine_pvalues(_f1d), SignificanceResult)

###
# fisher_exact

assert_type(fisher_exact(_f2d), SignificanceResult[float])

###
# quantile_test

assert_type(quantile_test(_f1d), QuantileTestResult)

###
# wasserstein_distance, wasserstein_distance_nd, energy_distance

assert_type(wasserstein_distance(_f1d, _f1d), np.float64)
assert_type(wasserstein_distance_nd(_f1d, _f1d), np.float64)
assert_type(energy_distance(_f1d, _f1d), np.float64)

###
# expectile

assert_type(expectile(_f1d), np.float64)

###
# iqr

assert_type(iqr(_f1d), npc.floating | onp.ArrayND[npc.floating])

###
# median_abs_deviation

assert_type(median_abs_deviation(_f1d), np.float64)
assert_type(median_abs_deviation(_fnd, axis=None), onp.ArrayND[np.float64])
assert_type(median_abs_deviation(_fnd, keepdims=True), onp.ArrayND[np.float64])

###
# zmap

assert_type(zmap(_f1d, _f1d), onp.Array1D[npc.floating])
assert_type(zmap(_fnd, _fnd), onp.ArrayND[npc.floating])

###
# siegelslopes

assert_type(siegelslopes(_f1d), SiegelslopesResult[np.float64])
assert_type(siegelslopes(_fnd, axis=None), SiegelslopesResult[np.float64])
assert_type(siegelslopes(_f2d, axis=0), SiegelslopesResult[onp.Array1D[np.float64]])
assert_type(siegelslopes(_fnd, keepdims=True), SiegelslopesResult[onp.ArrayND[np.float64]])

###
# theilslopes

assert_type(theilslopes(_f1d), TheilslopesResult[np.float64])
assert_type(theilslopes(_fnd, axis=None), TheilslopesResult[np.float64])
assert_type(theilslopes(_f2d, axis=0), TheilslopesResult[onp.Array1D[np.float64]])
assert_type(theilslopes(_fnd, keepdims=True), TheilslopesResult[onp.ArrayND[np.float64]])
