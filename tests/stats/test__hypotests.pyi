# type-tests for `stats/_hypotests.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import (
    barnard_exact,
    boschloo_exact,
    cramervonmises,
    cramervonmises_2samp,
    epps_singleton_2samp,
    poisson_means_test,
    somersd,
    tukey_hsd,
)
from scipy.stats._hypotests import (
    BarnardExactResult,
    BoschlooExactResult,
    CramerVonMisesResult,
    Epps_Singleton_2sampResult,
    SomersDResult,
    TukeyHSDResult,
)
from scipy.stats._stats_py import SignificanceResult

###

_f1d: onp.Array1D[np.float64]
_f2d: onp.Array2D[np.float64]
_fnd: onp.ArrayND[np.float64]
_i2d: onp.Array2D[np.int64]

_py_f_1d: list[float]
_py_i_2d: list[list[int]]

###
# epps_singleton_2samp

assert_type(epps_singleton_2samp(_py_f_1d, _py_f_1d), Epps_Singleton_2sampResult[float])
assert_type(epps_singleton_2samp(_f1d, _f1d), Epps_Singleton_2sampResult[float])
assert_type(epps_singleton_2samp(_fnd, _fnd, axis=None), Epps_Singleton_2sampResult[float])
assert_type(epps_singleton_2samp(_fnd, _fnd, keepdims=True), Epps_Singleton_2sampResult[onp.ArrayND[np.float64]])

assert_type(epps_singleton_2samp(_f1d, _f1d).statistic, float)
assert_type(epps_singleton_2samp(_f1d, _f1d).pvalue, float)

###
# cramervonmises

assert_type(cramervonmises(_py_f_1d, "norm"), CramerVonMisesResult[float])
assert_type(cramervonmises(_f1d, "norm"), CramerVonMisesResult[float])
assert_type(cramervonmises(_fnd, "norm", axis=None), CramerVonMisesResult[float])
assert_type(cramervonmises(_fnd, "norm", keepdims=True), CramerVonMisesResult[onp.ArrayND[np.float64]])

assert_type(cramervonmises(_f1d, "norm").statistic, float)
assert_type(cramervonmises(_f1d, "norm").pvalue, float)

###
# cramervonmises_2samp

assert_type(cramervonmises_2samp(_py_f_1d, _py_f_1d), CramerVonMisesResult[float])
assert_type(cramervonmises_2samp(_f1d, _f1d), CramerVonMisesResult[float])
assert_type(cramervonmises_2samp(_fnd, _fnd, axis=None), CramerVonMisesResult[float])
assert_type(cramervonmises_2samp(_fnd, _fnd, keepdims=True), CramerVonMisesResult[onp.ArrayND[np.float64]])

assert_type(cramervonmises_2samp(_f1d, _f1d).statistic, float)
assert_type(cramervonmises_2samp(_f1d, _f1d).pvalue, float)

###
# poisson_means_test

assert_type(poisson_means_test(5, 100.0, 3, 80.0), SignificanceResult[np.float64])
assert_type(poisson_means_test(5, 100.0, 3, 80.0).statistic, np.float64)
assert_type(poisson_means_test(5, 100.0, 3, 80.0).pvalue, np.float64)

###
# somersd

assert_type(somersd(_py_f_1d, _py_f_1d), SomersDResult)
assert_type(somersd(_f1d, _f1d), SomersDResult)
assert_type(somersd(_f2d), SomersDResult)
assert_type(somersd(_f1d, _f1d).statistic, float)
assert_type(somersd(_f1d, _f1d).pvalue, float)

###
# barnard_exact

assert_type(barnard_exact(_py_i_2d), BarnardExactResult)
assert_type(barnard_exact(_i2d), BarnardExactResult)
assert_type(barnard_exact(_i2d).statistic, float)
assert_type(barnard_exact(_i2d).pvalue, float)

###
# boschloo_exact

assert_type(boschloo_exact(_py_i_2d), BoschlooExactResult)
assert_type(boschloo_exact(_i2d), BoschlooExactResult)
assert_type(boschloo_exact(_i2d).statistic, float)
assert_type(boschloo_exact(_i2d).pvalue, float)

###
# tukey_hsd

assert_type(tukey_hsd(_f1d, _f1d), TukeyHSDResult)
assert_type(tukey_hsd(_f1d, _f1d, _f1d), TukeyHSDResult)
assert_type(tukey_hsd(_f1d, _f1d).statistic, onp.Array2D[np.float64])
assert_type(tukey_hsd(_f1d, _f1d).pvalue, onp.Array2D[np.float64])
