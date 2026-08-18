from collections.abc import Callable
from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.stats import (
    alpha,
    anglit,
    arcsine,
    argus,
    beta,
    betaprime,
    bradford,
    burr,
    burr12,
    cauchy,
    chi,
    chi2,
    cosine,
    crystalball,
    dgamma,
    dpareto_lognorm,
    dweibull,
    erlang,
    expon,
    exponnorm,
    exponpow,
    exponweib,
    f,
    fatiguelife,
    fisk,
    foldcauchy,
    foldnorm,
    gamma,
    gausshyper,
    genexpon,
    genextreme,
    gengamma,
    genhalflogistic,
    genhyperbolic,
    geninvgauss,
    genlogistic,
    gennorm,
    genpareto,
    gibrat,
    gompertz,
    gumbel_l,
    gumbel_r,
    halfcauchy,
    halfgennorm,
    halflogistic,
    halfnorm,
    hypsecant,
    invgamma,
    invgauss,
    invweibull,
    irwinhall,
    jf_skew_t,
    johnsonsb,
    johnsonsu,
    kappa3,
    kappa4,
    ksone,
    kstwo,
    kstwobign,
    landau,
    laplace,
    laplace_asymmetric,
    levy,
    levy_l,
    levy_stable,
    loggamma,
    logistic,
    loglaplace,
    lognorm,
    loguniform,
    lomax,
    maxwell,
    mielke,
    moyal,
    nakagami,
    ncf,
    nct,
    ncx2,
    norm,
    norminvgauss,
    pareto,
    pearson3,
    powerlaw,
    powerlognorm,
    powernorm,
    rayleigh,
    rdist,
    recipinvgauss,
    reciprocal,
    rel_breitwigner,
    rice,
    rv_continuous,
    rv_histogram,
    semicircular,
    skewcauchy,
    skewnorm,
    studentized_range,
    t,
    trapezoid,
    triang,
    truncexpon,
    truncnorm,
    truncpareto,
    truncweibull_min,
    tukeylambda,
    uniform,
    vonmises,
    vonmises_line,
    wald,
    weibull_max,
    weibull_min,
    wrapcauchy,
)

###

_f32: np.float32
_f64_nd: onp.ArrayND[np.float64]

_n: int
_shape_nd: tuple[int, ...]
_np_shape: tuple[np.intp, np.intp]

def _optimizer(
    func: Callable[[list[float], onp.ArrayND[np.float64]], np.float64],
    x0: list[float],
    args: tuple[onp.ArrayND[np.float64]],
    disp: int,
) -> onp.Array1D[np.float64]: ...

###

assert_subtype[rv_continuous](alpha)
assert_subtype[rv_continuous](anglit)
assert_subtype[rv_continuous](arcsine)
assert_subtype[rv_continuous](argus)
assert_subtype[rv_continuous](beta)
assert_subtype[rv_continuous](betaprime)
assert_subtype[rv_continuous](bradford)
assert_subtype[rv_continuous](burr)
assert_subtype[rv_continuous](burr12)
assert_subtype[rv_continuous](cauchy)
assert_subtype[rv_continuous](chi)
assert_subtype[rv_continuous](chi2)
assert_subtype[rv_continuous](cosine)
assert_subtype[rv_continuous](crystalball)
assert_subtype[rv_continuous](dgamma)
assert_subtype[rv_continuous](dpareto_lognorm)
assert_subtype[rv_continuous](dweibull)
assert_subtype[rv_continuous](erlang)
assert_subtype[rv_continuous](expon)
assert_subtype[rv_continuous](exponnorm)
assert_subtype[rv_continuous](exponpow)
assert_subtype[rv_continuous](exponweib)
assert_subtype[rv_continuous](f)
assert_subtype[rv_continuous](fatiguelife)
assert_subtype[rv_continuous](fisk)
assert_subtype[rv_continuous](foldcauchy)
assert_subtype[rv_continuous](foldnorm)
assert_subtype[rv_continuous](gamma)
assert_subtype[rv_continuous](gausshyper)
assert_subtype[rv_continuous](genexpon)
assert_subtype[rv_continuous](genextreme)
assert_subtype[rv_continuous](gengamma)
assert_subtype[rv_continuous](gengamma)
assert_subtype[rv_continuous](genhalflogistic)
assert_subtype[rv_continuous](genhyperbolic)
assert_subtype[rv_continuous](geninvgauss)
assert_subtype[rv_continuous](genlogistic)
assert_subtype[rv_continuous](gennorm)
assert_subtype[rv_continuous](halfgennorm)
assert_subtype[rv_continuous](genpareto)
assert_subtype[rv_continuous](gibrat)
assert_subtype[rv_continuous](gompertz)
assert_subtype[rv_continuous](gumbel_l)
assert_subtype[rv_continuous](gumbel_r)
assert_subtype[rv_continuous](halfcauchy)
assert_subtype[rv_continuous](halflogistic)
assert_subtype[rv_continuous](halfnorm)
assert_subtype[rv_continuous](hypsecant)
assert_subtype[rv_continuous](invgamma)
assert_subtype[rv_continuous](invgauss)
assert_subtype[rv_continuous](invweibull)
assert_subtype[rv_continuous](irwinhall)
assert_subtype[rv_continuous](jf_skew_t)
assert_subtype[rv_continuous](johnsonsb)
assert_subtype[rv_continuous](johnsonsu)
assert_subtype[rv_continuous](kappa4)
assert_subtype[rv_continuous](kappa4)
assert_subtype[rv_continuous](kappa4)
assert_subtype[rv_continuous](kappa4)
assert_subtype[rv_continuous](kappa3)
assert_subtype[rv_continuous](ksone)
assert_subtype[rv_continuous](kstwo)
assert_subtype[rv_continuous](kstwobign)
assert_subtype[rv_continuous](landau)
assert_subtype[rv_continuous](laplace)
assert_subtype[rv_continuous](laplace_asymmetric)
assert_subtype[rv_continuous](levy)
assert_subtype[rv_continuous](levy_l)
assert_subtype[rv_continuous](levy_stable)
assert_subtype[rv_continuous](loggamma)
assert_subtype[rv_continuous](logistic)
assert_subtype[rv_continuous](loglaplace)
assert_subtype[rv_continuous](lognorm)
assert_subtype[rv_continuous](loguniform)
assert_subtype[rv_continuous](lomax)
assert_subtype[rv_continuous](maxwell)
assert_subtype[rv_continuous](mielke)
assert_subtype[rv_continuous](moyal)
assert_subtype[rv_continuous](nakagami)
assert_subtype[rv_continuous](ncf)
assert_subtype[rv_continuous](nct)
assert_subtype[rv_continuous](ncx2)
assert_subtype[rv_continuous](norm)
assert_subtype[rv_continuous](norminvgauss)
assert_subtype[rv_continuous](pareto)
assert_subtype[rv_continuous](pearson3)
assert_subtype[rv_continuous](pearson3)
assert_subtype[rv_continuous](powerlaw)
assert_subtype[rv_continuous](powerlaw)
assert_subtype[rv_continuous](powerlognorm)
assert_subtype[rv_continuous](powernorm)
assert_subtype[rv_continuous](rayleigh)
assert_subtype[rv_continuous](rdist)
assert_subtype[rv_continuous](recipinvgauss)
assert_subtype[rv_continuous](reciprocal)
assert_subtype[rv_continuous](rel_breitwigner)
assert_subtype[rv_continuous](rice)
assert_subtype[rv_continuous](semicircular)
assert_subtype[rv_continuous](skewcauchy)
assert_subtype[rv_continuous](skewnorm)
assert_subtype[rv_continuous](studentized_range)
assert_subtype[rv_continuous](t)
assert_subtype[rv_continuous](trapezoid)
assert_subtype[rv_continuous](triang)
assert_subtype[rv_continuous](truncexpon)
assert_subtype[rv_continuous](truncnorm)
assert_subtype[rv_continuous](truncnorm)
assert_subtype[rv_continuous](truncpareto)
assert_subtype[rv_continuous](truncpareto)
assert_subtype[rv_continuous](truncpareto)
assert_subtype[rv_continuous](truncweibull_min)
assert_subtype[rv_continuous](tukeylambda)
assert_subtype[rv_continuous](uniform)
assert_subtype[rv_continuous](vonmises)
assert_subtype[rv_continuous](vonmises_line)
assert_subtype[rv_continuous](wald)
assert_subtype[rv_continuous](weibull_max)
assert_subtype[rv_continuous](weibull_min)
assert_subtype[rv_continuous](wrapcauchy)

assert_subtype[type[rv_continuous]](rv_histogram)

###

# .rvs

assert_type(norm.rvs(), np.float64)
assert_type(norm.rvs(size=None), np.float64)
assert_type(norm.rvs(size=()), np.float64)
assert_type(norm.rvs(0.5, 1.2), np.float64)
assert_type(norm.rvs(loc=0.5, scale=1.2), np.float64)
assert_type(expon.rvs(), np.float64)
assert_type(gamma.rvs(2.0), np.float64)
assert_type(gamma.rvs(2.0, size=None), np.float64)
assert_type(gamma.rvs(2.0, size=()), np.float64)

assert_type(norm.rvs(size=4), onp.Array1D[np.float64])
assert_type(norm.rvs(size=(_n,)), onp.Array1D[np.float64])
assert_type(norm.rvs(size=(_n, _n)), onp.Array2D[np.float64])
assert_type(norm.rvs(size=(_n, _n, _n)), onp.Array3D[np.float64])
assert_type(uniform.rvs(size=4), onp.Array1D[np.float64])
assert_type(norm.rvs(size=_np_shape), onp.ArrayND[np.float64] | Any)

assert_type(gamma.rvs(2.0, size=4), onp.Array1D[np.float64])

assert_type(norm.rvs(_f64_nd), onp.ArrayND[np.float64])
assert_type(gamma.rvs(_f64_nd, size=None), onp.ArrayND[np.float64])
assert_type(norm.rvs(loc=_f64_nd), onp.ArrayND[np.float64])
assert_type(norm.rvs(scale=_f64_nd), onp.ArrayND[np.float64])

# .fit

assert_type(norm.fit(_f64_nd), tuple[np.float64, np.float64])
assert_type(norm.fit(_f64_nd, floc=0), tuple[int, np.float64])
assert_type(norm.fit(_f64_nd, floc=0.0), tuple[float, np.float64])
assert_type(norm.fit(_f64_nd, floc=_f32), tuple[np.float32, np.float64])
assert_type(norm.fit(_f64_nd, fscale=1), tuple[np.float64, int])
assert_type(norm.fit(_f64_nd, fscale=2.0), tuple[np.float64, float])
assert_type(gamma.fit(_f64_nd, optimizer=_optimizer), tuple[float | np.float64, ...])
