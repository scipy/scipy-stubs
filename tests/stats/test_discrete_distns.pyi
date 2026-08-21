from typing import assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.stats import (
    bernoulli,
    betabinom,
    betanbinom,
    binom,
    boltzmann,
    dlaplace,
    geom,
    hypergeom,
    logser,
    nbinom,
    nchypergeom_fisher,
    nchypergeom_wallenius,
    nhypergeom,
    planck,
    poisson,
    poisson_binom,
    randint,
    rv_discrete,
    skellam,
    yulesimon,
    zipf,
    zipfian,
)

assert_subtype[rv_discrete](bernoulli)
assert_subtype[rv_discrete](betabinom)
assert_subtype[rv_discrete](betanbinom)
assert_subtype[rv_discrete](binom)
assert_subtype[rv_discrete](boltzmann)
assert_subtype[rv_discrete](dlaplace)
assert_subtype[rv_discrete](geom)
assert_subtype[rv_discrete](hypergeom)
assert_subtype[rv_discrete](hypergeom)
assert_subtype[rv_discrete](hypergeom)
assert_subtype[rv_discrete](nchypergeom_fisher)
assert_subtype[rv_discrete](nchypergeom_wallenius)
assert_subtype[rv_discrete](logser)
assert_subtype[rv_discrete](nbinom)
assert_subtype[rv_discrete](nbinom)
assert_subtype[rv_discrete](planck)
assert_subtype[rv_discrete](poisson)
assert_subtype[rv_discrete](poisson_binom)
assert_subtype[rv_discrete](randint)
assert_subtype[rv_discrete](skellam)
assert_subtype[rv_discrete](zipf)
assert_subtype[rv_discrete](zipfian)
assert_subtype[rv_discrete](zipfian)
assert_subtype[rv_discrete](yulesimon)
assert_subtype[rv_discrete](nhypergeom)

###

# .rvs

assert_type(poisson.rvs(1.0), int)
assert_type(poisson.rvs(1.0, size=None), int)
assert_type(poisson.rvs(1.0, size=()), int)
assert_type(poisson.rvs(1.0, size=4), onp.ArrayND[np.int64])

# .expect

_frozen = poisson(1.0)
assert_type(_frozen.expect(), float | np.float64)
assert_type(_frozen.expect(lambda k: k), float | np.float64)
assert_type(_frozen.expect(lambda k: k, lb=0, ub=5, conditional=True), float | np.float64)
assert_type(_frozen.expect(lambda k: k, maxcount=2000), float | np.float64)
assert_type(_frozen.expect(lambda k: k, tolerance=1e-8), float | np.float64)
assert_type(_frozen.expect(lambda k: k, chunksize=64), float | np.float64)
