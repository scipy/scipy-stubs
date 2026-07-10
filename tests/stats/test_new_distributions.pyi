# type-tests for `stats/_new_distributions.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import Binomial, Logistic, Normal, Uniform
from scipy.stats._new_distributions import StandardNormal

###

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]

###
# Normal

_norm_0d_f32: Normal[tuple[()], np.float32]
_norm_1d_f32: Normal[tuple[int], np.float32]

assert_type(_norm_0d_f32.mu, np.float32)
assert_type(_norm_0d_f32.sigma, np.float32)
assert_type(_norm_1d_f32.mu, onp.Array1D[np.float32])  # pyrefly: ignore[assert-type]
assert_type(_norm_1d_f32.sigma, onp.Array1D[np.float32])  # pyrefly: ignore[assert-type]

assert_type(Normal(), StandardNormal)  # type:ignore[assert-type]
assert_type(Normal(mu=0.0, sigma=1.0), Normal[tuple[()], np.float64])
assert_type(Normal(mu=0.0, sigma=[1.0]), Normal[tuple[int]])
assert_type(Normal(mu=0.0, sigma=[[1.0]]), Normal[tuple[int, int]])
assert_type(Normal(mu=0.0, sigma=[[[1.0]]]), Normal[tuple[int, int, int]])
assert_type(Normal(mu=[0.0], sigma=1.0), Normal[tuple[int]])
assert_type(Normal(mu=[0.0], sigma=[1.0]), Normal[tuple[int]])
assert_type(Normal(mu=[0.0], sigma=[[1.0]]), Normal[tuple[int, int]])
assert_type(Normal(mu=[0.0], sigma=[[[1.0]]]), Normal[tuple[int, int, int]])
assert_type(Normal(mu=[[0.0]], sigma=1.0), Normal[tuple[int, int]])
assert_type(Normal(mu=[[0.0]], sigma=[1.0]), Normal[tuple[int, int]])
assert_type(Normal(mu=[[0.0]], sigma=[[1.0]]), Normal[tuple[int, int]])
assert_type(Normal(mu=[[0.0]], sigma=[[[1.0]]]), Normal[tuple[int, int, int]])
assert_type(Normal(mu=[[[0.0]]], sigma=1.0), Normal[tuple[int, int, int]])
assert_type(Normal(mu=[[[0.0]]], sigma=[1.0]), Normal[tuple[int, int, int]])
assert_type(Normal(mu=[[[0.0]]], sigma=[[1.0]]), Normal[tuple[int, int, int]])
assert_type(Normal(mu=[[[0.0]]], sigma=[[[1.0]]]), Normal[tuple[int, int, int]])

###
# Logistic

assert_type(Logistic(), Logistic)

###
# Uniform

_unif_0d_f32: Uniform[tuple[()], np.float32]
_unif_1d_f32: Uniform[tuple[int], np.float32]

assert_type(_unif_0d_f32.a, np.float32)
assert_type(_unif_0d_f32.b, np.float32)
assert_type(_unif_0d_f32.ab, np.float32)
assert_type(_unif_1d_f32.a, onp.Array1D[np.float32])  # pyrefly: ignore[assert-type]
assert_type(_unif_1d_f32.b, onp.Array1D[np.float32])  # pyrefly: ignore[assert-type]
assert_type(_unif_1d_f32.ab, onp.Array1D[np.float32])  # pyrefly: ignore[assert-type]

assert_type(Uniform(a=0.0, b=1.0), Uniform[tuple[()], np.float64])
assert_type(Uniform(a=0.0, b=[1.0]), Uniform[tuple[int]])
assert_type(Uniform(a=0.0, b=[[1.0]]), Uniform[tuple[int, int]])
assert_type(Uniform(a=0.0, b=[[[1.0]]]), Uniform[tuple[int, int, int]])
assert_type(Uniform(a=[0.0], b=1.0), Uniform[tuple[int]])
assert_type(Uniform(a=[0.0], b=[1.0]), Uniform[tuple[int]])
assert_type(Uniform(a=[0.0], b=[[1.0]]), Uniform[tuple[int, int]])
assert_type(Uniform(a=[0.0], b=[[[1.0]]]), Uniform[tuple[int, int, int]])
assert_type(Uniform(a=[[0.0]], b=1.0), Uniform[tuple[int, int]])
assert_type(Uniform(a=[[0.0]], b=[1.0]), Uniform[tuple[int, int]])
assert_type(Uniform(a=[[0.0]], b=[[1.0]]), Uniform[tuple[int, int]])
assert_type(Uniform(a=[[0.0]], b=[[[1.0]]]), Uniform[tuple[int, int, int]])
assert_type(Uniform(a=[[[0.0]]], b=1.0), Uniform[tuple[int, int, int]])
assert_type(Uniform(a=[[[0.0]]], b=[1.0]), Uniform[tuple[int, int, int]])
assert_type(Uniform(a=[[[0.0]]], b=[[1.0]]), Uniform[tuple[int, int, int]])
assert_type(Uniform(a=[[[0.0]]], b=[[[1.0]]]), Uniform[tuple[int, int, int]])

###
# Binomial

_binom_0d: Binomial[tuple[()]]
_binom_1d: Binomial[tuple[int]]

assert_type(_binom_0d.n, np.float64)
assert_type(_binom_0d.p, np.float64)
assert_type(_binom_1d.n, onp.Array1D[np.float64])  # pyrefly: ignore[assert-type]
assert_type(_binom_1d.p, onp.Array1D[np.float64])  # pyrefly: ignore[assert-type]

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
