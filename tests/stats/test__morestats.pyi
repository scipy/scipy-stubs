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

_f1d: onp.Array1D[np.float64]
_f2d: onp.Array2D[np.float64]
_fnd: onp.ArrayND[np.float64]

_py_f_1d: list[float]
_py_f_2d: list[list[float]]

###
# bayes_mvs

assert_type(bayes_mvs(_py_f_1d), tuple[Mean, Variance, Std_dev])
assert_type(bayes_mvs(_f1d), tuple[Mean, Variance, Std_dev])

###
# mvsdist

assert_subtype[rv_continuous_frozen](mvsdist(_f1d)[0])
assert_subtype[rv_continuous_frozen](mvsdist(_f1d)[1])
assert_subtype[rv_continuous_frozen](mvsdist(_f1d)[2])

###
# kstat

assert_type(kstat(_fnd), np.float64)
assert_type(kstat(_fnd, axis=None), np.float64)
assert_type(kstat(_fnd, axis=0), np.float64 | onp.ArrayND[np.float64])
assert_type(kstat(_fnd, keepdims=True), onp.ArrayND[np.float64])

###
# kstatvar

assert_type(kstatvar(_fnd), np.float64)
assert_type(kstatvar(_fnd, axis=None), np.float64)
assert_type(kstatvar(_fnd, axis=0), np.float64 | onp.ArrayND[np.float64])
assert_type(kstatvar(_fnd, keepdims=True), onp.ArrayND[np.float64])

###
# probplot

assert_type(
    probplot(_f1d), tuple[tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]], tuple[np.float64, np.float64, np.float64]]
)
assert_type(probplot(_f1d, fit=False), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])

###
# ppcc_max

assert_type(ppcc_max(_f1d), np.float64)
assert_type(ppcc_max(_f2d), np.float64)

###
# ppcc_plot

assert_type(ppcc_plot(_f1d, 0.0, 2.0), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])

###
# anderson

assert_type(anderson(_f1d), AndersonResult)  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
assert_type(anderson(_fnd), AndersonResult)  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]

###
# anderson_ksamp

assert_type(anderson_ksamp(_fnd), Anderson_ksampResult)

###
# shapiro

assert_type(shapiro(_fnd), ShapiroResult[np.float64])
assert_type(shapiro(_fnd, axis=None), ShapiroResult[np.float64])
assert_type(shapiro(_fnd, axis=0), ShapiroResult)
assert_type(shapiro(_fnd, keepdims=True), ShapiroResult[onp.ArrayND[np.float64]])

###
# ansari

assert_type(ansari(_fnd, _fnd, axis=None), AnsariResult[np.float64])
assert_type(ansari(_fnd, _fnd, keepdims=True), AnsariResult[onp.ArrayND[np.float64]])
assert_type(ansari(_fnd, _fnd), AnsariResult)

###
# bartlett

assert_type(bartlett(_fnd, _fnd, axis=None), BartlettResult[np.float64])
assert_type(bartlett(_fnd, _fnd, keepdims=True), BartlettResult[onp.ArrayND[np.float64]])
assert_type(bartlett(_fnd, _fnd), BartlettResult)

###
# levene

assert_type(levene(_fnd, _fnd), LeveneResult[np.float64 | onp.ArrayND[np.float64]])
assert_type(levene(_fnd, _fnd, axis=None), LeveneResult[np.float64])
assert_type(levene(_fnd, _fnd, keepdims=True), LeveneResult[onp.ArrayND[np.float64]])

###
# fligner

assert_type(fligner(_fnd, _fnd, axis=None), FlignerResult[np.float64])
assert_type(fligner(_fnd, _fnd, keepdims=True), FlignerResult[onp.ArrayND[np.float64]])
assert_type(fligner(_fnd, _fnd), FlignerResult)

###
# mood

assert_type(mood(_fnd, _fnd, None), SignificanceResult[np.float64])
assert_type(mood(_fnd, _fnd, keepdims=True), SignificanceResult[onp.ArrayND[np.float64]])
assert_type(mood(_fnd, _fnd), SignificanceResult[np.float64 | onp.ArrayND[np.float64]])

###
# wilcoxon

assert_type(wilcoxon(_fnd, axis=None), WilcoxonResult[np.float64])
assert_type(wilcoxon(_fnd, keepdims=True), WilcoxonResult[onp.ArrayND[np.float64]])
assert_type(wilcoxon(_fnd), WilcoxonResult)

###
# median_test

assert_type(median_test(_f1d, _f1d), MedianTestResult)
assert_type(median_test(_f1d, _f1d, _f1d), MedianTestResult)

###
# circmean

assert_type(circmean(_fnd), np.float64)
assert_type(circmean(_fnd, axis=None), np.float64)
assert_type(circmean(_fnd, axis=0), np.float64 | onp.ArrayND[np.float64])
assert_type(circmean(_fnd, keepdims=True), onp.ArrayND[np.float64])

###
# circvar

assert_type(circvar(_fnd), np.float64)
assert_type(circvar(_fnd, axis=None), np.float64)
assert_type(circvar(_fnd, axis=0), np.float64 | onp.ArrayND[np.float64])
assert_type(circvar(_fnd, keepdims=True), onp.ArrayND[np.float64])

###
# circstd

assert_type(circstd(_fnd), np.float64)
assert_type(circstd(_fnd, axis=None), np.float64)
assert_type(circstd(_fnd, axis=0), np.float64 | onp.ArrayND[np.float64])
assert_type(circstd(_fnd, keepdims=True), onp.ArrayND[np.float64])

###
# false_discovery_control

assert_type(false_discovery_control(_f1d), onp.ArrayND[np.float64])
assert_type(false_discovery_control(_fnd), onp.ArrayND[np.float64])
assert_type(false_discovery_control(0.05), onp.ArrayND[np.float64])
