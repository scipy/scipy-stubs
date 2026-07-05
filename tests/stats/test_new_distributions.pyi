# type-tests for `stats/_new_distributions.pyi`

from typing import assert_type

import numpy as np
import optype.numpy.compat as npc

from scipy.stats import Binomial, Logistic, Normal, Uniform
from scipy.stats._new_distributions import StandardNormal

###

# Normal
assert_type(Normal(), StandardNormal)  # type:ignore[assert-type]
assert_type(Normal(mu=0.0, sigma=1.0), Normal[tuple[()], np.float64])
assert_type(Normal(mu=[0.0, 1.0], sigma=1.0), Normal[tuple[int], npc.floating])

# Logistic
assert_type(Logistic(), Logistic)

# Binomial
assert_type(Binomial(n=9, p=0.5), Binomial[tuple[()]])
assert_type(Binomial(n=9, p=[0.5]), Binomial[tuple[int]])
assert_type(Binomial(n=9, p=[[0.5]]), Binomial[tuple[int, int]])
assert_type(Binomial(n=9, p=[[[0.5]]]), Binomial[tuple[int, int, int]])
assert_type(Binomial(n=[9], p=0.5), Binomial[tuple[int]])
assert_type(Binomial(n=[9], p=[0.5]), Binomial[tuple[int]])
assert_type(Binomial(n=[9], p=[[0.5]]), Binomial[tuple[int, int]])
assert_type(Binomial(n=[9], p=[[[0.5]]]), Binomial[tuple[int, int, int]])
assert_type(Binomial(n=[[9]], p=0.5), Binomial[tuple[int, int]])
assert_type(Binomial(n=[[9]], p=[0.5]), Binomial[tuple[int, int]])
assert_type(Binomial(n=[[9]], p=[[0.5]]), Binomial[tuple[int, int]])
assert_type(Binomial(n=[[9]], p=[[[0.5]]]), Binomial[tuple[int, int, int]])
assert_type(Binomial(n=[[[9]]], p=0.5), Binomial[tuple[int, int, int]])
assert_type(Binomial(n=[[[9]]], p=[0.5]), Binomial[tuple[int, int, int]])
assert_type(Binomial(n=[[[9]]], p=[[0.5]]), Binomial[tuple[int, int, int]])
assert_type(Binomial(n=[[[9]]], p=[[[0.5]]]), Binomial[tuple[int, int, int]])

# Uniform
assert_type(Uniform(a=0.0, b=1.0), Uniform[tuple[()], np.float64])
assert_type(Uniform(a=[0.0, 1.0], b=2.0), Uniform[tuple[int]])
assert_type(Uniform(a=0.0, b=[1.0, 2.0]), Uniform[tuple[int]])
