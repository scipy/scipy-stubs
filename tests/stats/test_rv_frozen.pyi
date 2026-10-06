from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import distributions as d

###

type _Float = np.float64
type _FloatND = onp.ArrayND[np.float64] | Any

def _f2c(x: float, /) -> complex: ...

###
# `rv_continuous_frozen`
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
assert_type(d.uniform().expect(_f2c, complex_func=True), np.complex128)
# pyrefly: ignore [bad-argument-type]
d.uniform([0, -1]).expect()  # type: ignore[misc]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
# pyrefly: ignore [bad-argument-type]
d.uniform([0, 0.5], 2).expect()  # type: ignore[misc]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
# pyrefly: ignore [bad-argument-type]
d.uniform(0, [0.5, 2]).expect()  # type: ignore[misc]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
###

###
# `rv_discrete_frozen`
# .mean()
assert_type(d.bernoulli().mean(), _Float)  # TODO: Reject empty constructor (for all `rv_discrete`?)
assert_type(d.bernoulli(0.5).mean(), _Float)
assert_type(d.bernoulli(0.5, 1).mean(), _Float)
assert_type(d.bernoulli(0.5, loc=1).mean(), _Float)
assert_type(d.bernoulli([0, 0.5]).mean(), _FloatND)
# .expect()
assert_type(d.bernoulli(0.5).mean(), _Float)
assert_type(d.bernoulli(0.5, 1).mean(), _Float)
assert_type(d.bernoulli(0.5, loc=1).mean(), _Float)
# pyrefly: ignore [bad-argument-type]
d.bernoulli([0, 0.5]).expect()  # type: ignore[misc]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
# .support()
assert_type(d.binom(10, 0.3).support(), tuple[np.float64 | Any, np.float64 | Any])
###

# TODO: more tests
