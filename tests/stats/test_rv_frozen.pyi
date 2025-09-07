from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import distributions as d

_Float: TypeAlias = float | np.float64
_FloatND: TypeAlias = _Float | onp.ArrayND[np.float64]

###
# Continuous distributions (`rv_continuous_frozen`)
# .mean()
assert_type(d.uniform().mean(), _Float)
assert_type(d.uniform(0).mean(), _Float)
assert_type(d.uniform(0.5, 2).mean(), _Float)
assert_type(d.uniform([0, -1]).mean(), _FloatND)
assert_type(d.uniform([0, 0.5], 2).mean(), _FloatND)
assert_type(d.uniform(0, [0.5, 2]).mean(), _FloatND)

# .expect()
assert_type(d.uniform().expect(), _Float)
assert_type(d.uniform(0).expect(), _Float)
assert_type(d.uniform(0.5, 2).expect(), _Float)
assert_type(d.uniform([0, -1]).expect(), _FloatND)
assert_type(d.uniform([0, 0.5], 2).expect(), _FloatND)
assert_type(d.uniform(0, [0.5, 2]).expect(), _FloatND)

# Additional continuous distributions
assert_type(d.norm().mean(), _Float)
assert_type(d.norm(0, 1).mean(), _Float)
assert_type(d.norm([0, 1], 1).mean(), _FloatND)
assert_type(d.expon().mean(), _Float)
assert_type(d.expon([0, 1]).mean(), _FloatND)

assert_type(d.norm().expect(), _Float)
assert_type(d.norm([0, 1], 1).expect(), _FloatND)
assert_type(d.expon().expect(), _Float)
assert_type(d.expon([0, 1]).expect(), _FloatND)
###

###
# Discrete distributions (`rv_discrete_frozen`)
# .mean()
assert_type(d.bernoulli(0.5).mean(), _Float)
assert_type(d.bernoulli(0.5, 1).mean(), _Float)
assert_type(d.bernoulli(0.5, loc=1).mean(), _Float)
assert_type(d.bernoulli([0, 0.5]).mean(), _FloatND)
assert_type(d.binom(10, 0.5).mean(), _Float)
assert_type(d.binom([5, 10], 0.5).mean(), _FloatND)
assert_type(d.poisson(3).mean(), _Float)
assert_type(d.poisson([1, 3]).mean(), _FloatND)

# .expect()
assert_type(d.bernoulli(0.5).expect(), _Float)
assert_type(d.bernoulli(0.5, 1).expect(), _Float)
assert_type(d.bernoulli(0.5, loc=1).expect(), _Float)
assert_type(d.bernoulli([0, 0.5]).expect(), _FloatND)
assert_type(d.binom(10, 0.5).expect(), _Float)
assert_type(d.binom([5, 10], 0.5).expect(), _FloatND)
assert_type(d.poisson(3).expect(), _Float)
assert_type(d.poisson([1, 3]).expect(), _FloatND)
###
