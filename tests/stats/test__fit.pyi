# type-tests for `stats/_fit.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import fit, goodness_of_fit
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete
from scipy.stats._fit import FitResult, GoodnessOfFitResult, _PXF1n, _PXF2n

###

_f64_1d: onp.Array1D[np.float64]

_rv_c: rv_continuous
_rv_d: rv_discrete

###
# fit

_fit_c = fit(_rv_c, _f64_1d)
assert_type(_fit_c, FitResult[_PXF2n])
assert_type(_fit_c.success, bool | None)
assert_type(_fit_c.message, str | None)
assert_type(_fit_c.discrete, bool)
assert_type(_fit_c.nllf(), np.float64)

_fit_d = fit(_rv_d, _f64_1d)
assert_type(_fit_d, FitResult[_PXF1n])

assert_type(fit(_rv_c, _f64_1d, method="mse"), FitResult[_PXF2n])

###
# goodness_of_fit

_gof = goodness_of_fit(_rv_c, _f64_1d)
assert_type(_gof, GoodnessOfFitResult)
assert_type(_gof.fit_result, FitResult[_PXF2n])
assert_type(_gof.statistic, float | np.float64)
assert_type(_gof.pvalue, float | np.float64)
assert_type(_gof.null_distribution, onp.Array1D[np.float64])

assert_type(goodness_of_fit(_rv_c, _f64_1d, statistic="ks", n_mc_samples=100), GoodnessOfFitResult)
