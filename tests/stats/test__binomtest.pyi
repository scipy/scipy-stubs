# type-tests for `stats/_binomtest.pyi`

from typing import assert_type

from scipy.stats import binomtest
from scipy.stats._binomtest import BinomTestResult
from scipy.stats._common import ConfidenceInterval

###
# binomtest

_r = binomtest(5, 10)
assert_type(_r, BinomTestResult)
assert_type(_r.k, int)
assert_type(_r.n, int)
assert_type(_r.statistic, float)
assert_type(_r.pvalue, float)
assert_type(_r.proportion_ci(), ConfidenceInterval)
assert_type(_r.proportion_ci(confidence_level=0.99, method="wilson"), ConfidenceInterval)

assert_type(binomtest(5, 10, p=0.3, alternative="less"), BinomTestResult)
assert_type(binomtest(5, 10, alternative="greater"), BinomTestResult)
