# type-tests for `stats/_hypotests.pyi`

from typing import Any, assert_type

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

_i64_2d: onp.Array2D[np.int64]

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_py_i_2d: list[list[int]]
_py_f_1d: list[float]

###

# epps_singleton_2samp

assert_type(epps_singleton_2samp(_py_f_1d, _py_f_1d), Epps_Singleton_2sampResult[np.float64])
assert_type(epps_singleton_2samp(_f64_1d, _f64_1d), Epps_Singleton_2sampResult[np.float64])
assert_type(epps_singleton_2samp(_f64_nd, _f64_nd, axis=None), Epps_Singleton_2sampResult[np.float64])
assert_type(epps_singleton_2samp(_f64_nd, _f64_nd, keepdims=True), Epps_Singleton_2sampResult[onp.ArrayND[np.float64]])

assert_type(epps_singleton_2samp(_f32_1d, _f32_1d), Epps_Singleton_2sampResult[np.float32])
assert_type(epps_singleton_2samp(_f32_2d, _f32_2d, axis=None), Epps_Singleton_2sampResult[np.float32])

assert_type(epps_singleton_2samp(_f64_1d, _f64_1d).statistic, np.float64)
assert_type(epps_singleton_2samp(_f64_1d, _f64_1d).pvalue, np.float64)

# cramervonmises

assert_type(cramervonmises(_py_f_1d, "norm"), CramerVonMisesResult[np.float64])
assert_type(cramervonmises(_f64_1d, "norm"), CramerVonMisesResult[np.float64])
assert_type(cramervonmises(_f64_nd, "norm", axis=None), CramerVonMisesResult[np.float64])
assert_type(cramervonmises(_f64_nd, "norm", keepdims=True), CramerVonMisesResult[onp.ArrayND[np.float64]])

assert_type(cramervonmises(_f32_1d, "norm"), CramerVonMisesResult[np.float64])

assert_type(cramervonmises(_f64_1d, "norm").statistic, np.float64)
assert_type(cramervonmises(_f64_1d, "norm").pvalue, np.float64)

# cramervonmises_2samp

assert_type(cramervonmises_2samp(_py_f_1d, _py_f_1d), CramerVonMisesResult[np.float64])
assert_type(cramervonmises_2samp(_f64_1d, _f64_1d), CramerVonMisesResult[np.float64])
assert_type(cramervonmises_2samp(_f64_nd, _f64_nd, axis=None), CramerVonMisesResult[np.float64])
assert_type(cramervonmises_2samp(_f64_nd, _f64_nd, keepdims=True), CramerVonMisesResult[onp.ArrayND[np.float64]])

assert_type(cramervonmises_2samp(_f32_1d, _f32_1d), CramerVonMisesResult[np.float32])
assert_type(cramervonmises_2samp(_f32_2d, _f32_2d, axis=None), CramerVonMisesResult[np.float32])

assert_type(cramervonmises_2samp(_f64_1d, _f64_1d).statistic, np.float64)
assert_type(cramervonmises_2samp(_f64_1d, _f64_1d).pvalue, np.float64)

# poisson_means_test

assert_type(poisson_means_test(5, 100.0, 3, 80.0), SignificanceResult[np.float64])
assert_type(poisson_means_test(5, 100.0, 3, 80.0).statistic, np.float64)
assert_type(poisson_means_test(5, 100.0, 3, 80.0).pvalue, np.float64)

# somersd

assert_type(somersd(_py_f_1d, _py_f_1d), SomersDResult)
assert_type(somersd(_f64_1d, _f64_1d), SomersDResult)
assert_type(somersd(_f64_2d), SomersDResult)
assert_type(somersd(_f64_1d, _f64_1d).statistic, float)
assert_type(somersd(_f64_1d, _f64_1d).pvalue, float)

# barnard_exact

assert_type(barnard_exact(_py_i_2d), BarnardExactResult)
assert_type(barnard_exact(_i64_2d), BarnardExactResult)
assert_type(barnard_exact(_i64_2d).statistic, float)
assert_type(barnard_exact(_i64_2d).pvalue, float)

# boschloo_exact

assert_type(boschloo_exact(_py_i_2d), BoschlooExactResult)
assert_type(boschloo_exact(_i64_2d), BoschlooExactResult)
assert_type(boschloo_exact(_i64_2d).statistic, float)
assert_type(boschloo_exact(_i64_2d).pvalue, float)

# tukey_hsd

assert_type(tukey_hsd(_f64_1d, _f64_1d), TukeyHSDResult)
assert_type(tukey_hsd(_f64_1d, _f64_1d, _f64_1d), TukeyHSDResult)
assert_type(tukey_hsd(_f64_1d, _f64_1d).statistic, onp.Array2D[np.float64 | Any])
assert_type(tukey_hsd(_f64_1d, _f64_1d).pvalue, onp.Array2D[np.float64])
