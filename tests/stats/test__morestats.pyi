# type-tests for `stats/_morestats.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.stats import (
    anderson,
    anderson_ksamp,
    ansari,
    bartlett,
    bayes_mvs,
    circmean,
    circstd,
    circvar,
    directional_stats,
    false_discovery_control,
    fligner,
    kstat,
    kstatvar,
    levene,
    median_test,
    mood,
    mvsdist,
    ppcc_max,
    ppcc_plot,
    probplot,
    shapiro,
    wilcoxon,
)
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from scipy.stats._morestats import (
    AndersonResult,
    Anderson_ksampResult,
    AnsariResult,
    BartlettResult,
    DirectionalStats,
    FlignerResult,
    LeveneResult,
    Mean,
    MedianTestResult,
    ShapiroResult,
    Std_dev,
    Variance,
    WilcoxonResult,
)
from scipy.stats._stats_py import SignificanceResult

###

_i16_2d: onp.Array2D[np.int16]
_i16_3d: onp.Array3D[np.int16]
_i16_nd: onp.ArrayND[np.int16]

_f32_2d: onp.Array2D[np.float32]
_f32_3d: onp.Array3D[np.float32]
_f32_nd: onp.ArrayND[np.float32]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_c64_2d: onp.Array2D[np.complex64]
_c64_3d: onp.Array3D[np.complex64]
_c64_nd: onp.ArrayND[np.complex64]

_c128_2d: onp.Array2D[np.complex128]
_c128_3d: onp.Array3D[np.complex128]
_c128_nd: onp.ArrayND[np.complex128]

_c160_2d: onp.Array2D[np.complex256]
_c160_3d: onp.Array3D[np.complex256]
_c160_nd: onp.ArrayND[np.complex256]

_py_f_1d: list[float]
_py_f_2d: list[list[float]]

###
# bayes_mvs

assert_type(bayes_mvs(_py_f_1d), tuple[Mean, Variance, Std_dev])
assert_type(bayes_mvs(_f64_1d), tuple[Mean, Variance, Std_dev])

###
# mvsdist

assert_subtype[rv_continuous_frozen](mvsdist(_f64_1d)[0])
assert_subtype[rv_continuous_frozen](mvsdist(_f64_1d)[1])
assert_subtype[rv_continuous_frozen](mvsdist(_f64_1d)[2])

###
# kstat

assert_type(kstat(_f64_nd), np.float64)
assert_type(kstat(_f64_nd, axis=None), np.float64)
assert_type(kstat(_f64_nd, axis=0), np.float64 | onp.ArrayND[np.float64])
assert_type(kstat(_f64_nd, keepdims=True), onp.ArrayND[np.float64])

###
# kstatvar

assert_type(kstatvar(_f64_nd), np.float64)
assert_type(kstatvar(_f64_nd, axis=None), np.float64)
assert_type(kstatvar(_f64_nd, axis=0), np.float64 | onp.ArrayND[np.float64])
assert_type(kstatvar(_f64_nd, keepdims=True), onp.ArrayND[np.float64])

###
# probplot

assert_type(
    probplot(_f64_1d), tuple[tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]], tuple[np.float64, np.float64, np.float64]]
)
assert_type(probplot(_f64_1d, fit=False), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])

###
# ppcc_max

assert_type(ppcc_max(_f64_1d), np.float64)
assert_type(ppcc_max(_f64_2d), np.float64)

###
# ppcc_plot

assert_type(ppcc_plot(_f64_1d, 0.0, 2.0), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])

###
# anderson

assert_type(anderson(_f64_1d), AndersonResult)  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
assert_type(anderson(_f64_nd), AndersonResult)  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]

###
# anderson_ksamp

assert_type(anderson_ksamp(_f64_nd), Anderson_ksampResult)

###
# shapiro

assert_type(shapiro(_f64_nd), ShapiroResult[np.float64])
assert_type(shapiro(_f64_nd, axis=None), ShapiroResult[np.float64])
assert_type(shapiro(_f64_nd, axis=0), ShapiroResult)
assert_type(shapiro(_f64_nd, keepdims=True), ShapiroResult[onp.ArrayND[np.float64]])

###
# ansari

assert_type(ansari(_f64_nd, _f64_nd, axis=None), AnsariResult[np.float64])
assert_type(ansari(_f64_nd, _f64_nd, keepdims=True), AnsariResult[onp.ArrayND[np.float64]])
assert_type(ansari(_f64_nd, _f64_nd), AnsariResult)

###
# bartlett

assert_type(bartlett(_f64_nd, _f64_nd, axis=None), BartlettResult[np.float64])
assert_type(bartlett(_f64_nd, _f64_nd, keepdims=True), BartlettResult[onp.ArrayND[np.float64]])
assert_type(bartlett(_f64_nd, _f64_nd), BartlettResult)

###
# levene

assert_type(levene(_f64_nd, _f64_nd), LeveneResult[np.float64 | onp.ArrayND[np.float64]])
assert_type(levene(_f64_nd, _f64_nd, axis=None), LeveneResult[np.float64])
assert_type(levene(_f64_nd, _f64_nd, keepdims=True), LeveneResult[onp.ArrayND[np.float64]])

###
# fligner

assert_type(fligner(_f64_nd, _f64_nd, axis=None), FlignerResult[np.float64])
assert_type(fligner(_f64_nd, _f64_nd, keepdims=True), FlignerResult[onp.ArrayND[np.float64]])
assert_type(fligner(_f64_nd, _f64_nd), FlignerResult)

###
# mood

assert_type(mood(_f64_nd, _f64_nd, None), SignificanceResult[np.float64])
assert_type(mood(_f64_nd, _f64_nd, keepdims=True), SignificanceResult[onp.ArrayND[np.float64]])
assert_type(mood(_f64_nd, _f64_nd), SignificanceResult[np.float64 | onp.ArrayND[np.float64]])

###
# wilcoxon

assert_type(wilcoxon(_f64_nd, axis=None), WilcoxonResult[np.float64])
assert_type(wilcoxon(_f64_nd, keepdims=True), WilcoxonResult[onp.ArrayND[np.float64]])
assert_type(wilcoxon(_f64_nd), WilcoxonResult)

###
# median_test

assert_type(median_test(_f64_1d, _f64_1d), MedianTestResult)
assert_type(median_test(_f64_1d, _f64_1d, _f64_1d), MedianTestResult)

###
# circmean

assert_type(circmean(_f64_nd), np.float64)
assert_type(circmean(_f64_nd, axis=None), np.float64)
assert_type(circmean(_f64_nd, axis=0), np.float64 | onp.ArrayND[np.float64])
assert_type(circmean(_f64_nd, keepdims=True), onp.ArrayND[np.float64])

###
# circvar

assert_type(circvar(_f64_nd), np.float64)
assert_type(circvar(_f64_nd, axis=None), np.float64)
assert_type(circvar(_f64_nd, axis=0), np.float64 | onp.ArrayND[np.float64])
assert_type(circvar(_f64_nd, keepdims=True), onp.ArrayND[np.float64])

###
# circstd

assert_type(circstd(_f64_nd), np.float64)
assert_type(circstd(_f64_nd, axis=None), np.float64)
assert_type(circstd(_f64_nd, axis=0), np.float64 | onp.ArrayND[np.float64])
assert_type(circstd(_f64_nd, keepdims=True), onp.ArrayND[np.float64])

###
# directional_stats

assert_type(directional_stats(_i16_2d), DirectionalStats[onp.Array1D[np.float64], np.float64])
assert_type(directional_stats(_f32_2d), DirectionalStats[onp.Array1D[np.float32], np.float32])
assert_type(directional_stats(_f64_2d), DirectionalStats[onp.Array1D[np.float64], np.float64])
assert_type(directional_stats(_c64_2d), DirectionalStats[onp.Array1D[np.complex64], np.float32])
assert_type(directional_stats(_c128_2d), DirectionalStats[onp.Array1D[np.complex128], np.float64])
assert_type(directional_stats(_c160_2d), DirectionalStats[onp.Array1D[np.clongdouble], np.longdouble])

assert_type(directional_stats(_i16_3d), DirectionalStats[onp.Array2D[np.float64], onp.Array1D[np.float64]])
assert_type(directional_stats(_f32_3d), DirectionalStats[onp.Array2D[np.float32], onp.Array1D[np.float32]])
assert_type(directional_stats(_f64_3d), DirectionalStats[onp.Array2D[np.float64], onp.Array1D[np.float64]])
assert_type(directional_stats(_c64_3d), DirectionalStats[onp.Array2D[np.complex64], onp.Array1D[np.float32]])
assert_type(directional_stats(_c128_3d), DirectionalStats[onp.Array2D[np.complex128], onp.Array1D[np.float64]])
assert_type(directional_stats(_c160_3d), DirectionalStats[onp.Array2D[np.clongdouble], onp.Array1D[np.longdouble]])

assert_type(directional_stats(_i16_nd), DirectionalStats[onp.ArrayND[np.float64], np.float64 | onp.ArrayND[np.float64]])
assert_type(directional_stats(_f32_nd), DirectionalStats[onp.ArrayND[np.float32], np.float32 | onp.ArrayND[np.float32]])
assert_type(directional_stats(_f64_nd), DirectionalStats[onp.ArrayND[np.float64], np.float64 | onp.ArrayND[np.float64]])
assert_type(directional_stats(_c64_nd), DirectionalStats[onp.ArrayND[np.complex64], np.float32 | onp.ArrayND[np.float32]])
assert_type(directional_stats(_c128_nd), DirectionalStats[onp.ArrayND[np.complex128], np.float64 | onp.ArrayND[np.float64]])
assert_type(
    directional_stats(_c160_nd), DirectionalStats[onp.ArrayND[np.clongdouble], np.longdouble | onp.ArrayND[np.longdouble]]
)

###
# false_discovery_control

assert_type(false_discovery_control(_f64_1d), onp.ArrayND[np.float64])
assert_type(false_discovery_control(_f64_nd), onp.ArrayND[np.float64])
assert_type(false_discovery_control(0.05), np.float64)
