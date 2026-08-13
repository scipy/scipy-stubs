from typing import assert_type, type_check_only

import numpy as np
import optype.numpy as onp

from scipy.stats import norm
from scipy.stats.qmc import Halton
from scipy.stats.sampling import (
    DiscreteAliasUrn,
    DiscreteGuideTable,
    FastGeneratorInversion,
    NumericalInverseHermite,
    NumericalInversePolynomial,
    RatioUniforms,
    SimpleRatioUniforms,
    TransformedDensityRejection,
)

###

def _f(x: onp.ArrayND[np.float64]) -> list[float]: ...

@type_check_only
class _ContRV:
    def pdf(self, x: float, /) -> float: ...
    def dpdf(self, x: float, /) -> float: ...
    def logpdf(self, x: float, /) -> np.float64: ...
    def cdf(self, x: float, /) -> np.float64: ...
    def support(self) -> tuple[float, float]: ...

@type_check_only
class _DiscrRV:
    def pmf(self, k: int, /) -> float: ...

_cont: _ContRV
_discr: _DiscrRV

_py_f_1d: list[float]
_f64_1d: onp.Array1D[np.float64]

_1d: tuple[int]
_2d: tuple[int, int]
_3d: tuple[int, int, int]

###
# RatioUniforms

_ru = RatioUniforms(_f, umax=1.0, vmin=-1.0, vmax=1.0)
assert_type(_ru, RatioUniforms)
assert_type(_ru.rvs(), onp.Array1D[np.float64])
assert_type(_ru.rvs(3), onp.Array1D[np.float64])
assert_type(_ru.rvs(3), onp.Array1D[np.float64])
assert_type(_ru.rvs(_1d), onp.Array1D[np.float64])
assert_type(_ru.rvs(_2d), onp.Array2D[np.float64])
assert_type(_ru.rvs(_3d), onp.Array3D[np.float64])

###
# FastGeneratorInversion

_fgi = FastGeneratorInversion(norm())
assert_type(_fgi, FastGeneratorInversion)
assert_type(FastGeneratorInversion(norm(), domain=(-5.0, 5.0), ignore_shape_range=True), FastGeneratorInversion)
assert_type(FastGeneratorInversion(norm(), random_state=0), FastGeneratorInversion)

assert_type(_fgi.random_state, np.random.Generator)
assert_type(_fgi.loc, float | np.float64)
assert_type(_fgi.scale, float | np.float64)
assert_type(_fgi.rvs(), np.float64)
assert_type(_fgi.rvs(3), onp.ArrayND[np.float64])
assert_type(_fgi.qrvs(), np.float64)
assert_type(_fgi.qrvs(4, d=1, qmc_engine=Halton(1)), onp.ArrayND[np.float64])
assert_type(_fgi.ppf(0.5), np.float64)
assert_type(_fgi.ppf(_f64_1d), onp.ArrayND[np.float64])
assert_type(_fgi.evaluate_error(), tuple[np.float64, np.float64])
assert_type(_fgi.support(), tuple[float, float] | tuple[np.float64, np.float64])

###
# TransformedDensityRejection

_tdr = TransformedDensityRejection(_cont)
assert_type(_tdr, TransformedDensityRejection)
assert_type(TransformedDensityRejection(_cont, mode=0.0, center=0.0, domain=(-5.0, 5.0)), TransformedDensityRejection)
assert_type(TransformedDensityRejection(_cont, c=0.0, construction_points=50, use_dars=False), TransformedDensityRejection)
assert_type(TransformedDensityRejection(_cont, max_squeeze_hat_ratio=0.95, random_state=0), TransformedDensityRejection)

assert_type(_tdr.hat_area, float)
assert_type(_tdr.squeeze_hat_ratio, float)
assert_type(_tdr.squeeze_area, float)
assert_type(_tdr.ppf_hat(0.5), np.float64)
assert_type(_tdr.ppf_hat([0.25, 0.75]), onp.ArrayND[np.float64])
assert_type(_tdr.rvs(), float)
assert_type(_tdr.rvs(3), onp.ArrayND[np.float64])
assert_type(_tdr.set_random_state(0), None)

###
# SimpleRatioUniforms

_sru = SimpleRatioUniforms(_cont, mode=0.0, domain=(-5.0, 5.0))
assert_type(_sru, SimpleRatioUniforms)
assert_type(SimpleRatioUniforms(_cont, pdf_area=2.0, cdf_at_mode=0.5, random_state=0), SimpleRatioUniforms)

assert_type(_sru.rvs(), float)
assert_type(_sru.rvs(3), onp.ArrayND[np.float64])

###
# NumericalInversePolynomial

_pinv = NumericalInversePolynomial(_cont)
assert_type(_pinv, NumericalInversePolynomial)
assert_type(NumericalInversePolynomial(_cont, mode=0.0, center=0.0, domain=(-5.0, 5.0)), NumericalInversePolynomial)
assert_type(NumericalInversePolynomial(_cont, order=3, u_resolution=1e-8, random_state=0), NumericalInversePolynomial)

assert_type(_pinv.intervals, int)
assert_type(_pinv.ppf(0.5), np.float64)
assert_type(_pinv.ppf([0.25, 0.75]), onp.ArrayND[np.float64])
assert_type(_pinv.cdf(0.5), np.float64)
assert_type(_pinv.cdf([0.25, 0.75]), onp.ArrayND[np.float64])
assert_type(_pinv.u_error().max_error, float)
assert_type(_pinv.u_error(1_000).mean_absolute_error, float)
assert_type(_pinv.qrvs(), np.float64)
assert_type(_pinv.qrvs(4), onp.ArrayND[np.float64])
assert_type(_pinv.qrvs(4, d=1, qmc_engine=Halton(1)), onp.ArrayND[np.float64])
assert_type(_pinv.rvs(), float)
assert_type(_pinv.rvs(3), onp.ArrayND[np.float64])

###
# NumericalInverseHermite

_hinv = NumericalInverseHermite(_cont)
assert_type(_hinv, NumericalInverseHermite)
assert_type(NumericalInverseHermite(_cont, domain=(-5.0, 5.0), order=1), NumericalInverseHermite)
assert_type(NumericalInverseHermite(_cont, u_resolution=1e-8, construction_points=_f64_1d), NumericalInverseHermite)
assert_type(NumericalInverseHermite(_cont, random_state=0), NumericalInverseHermite)

assert_type(_hinv.intervals, int)
assert_type(_hinv.midpoint_error, float)
assert_type(_hinv.ppf(0.5), np.float64)
assert_type(_hinv.ppf([0.25, 0.75]), onp.ArrayND[np.float64])
assert_type(_hinv.u_error().max_error, float)
assert_type(_hinv.qrvs(), np.float64)
assert_type(_hinv.qrvs(4, d=1, qmc_engine=Halton(1)), onp.ArrayND[np.float64])
assert_type(_hinv.rvs(), float)
assert_type(_hinv.rvs(3), onp.ArrayND[np.float64])

###
# DiscreteAliasUrn

_dau = DiscreteAliasUrn(_py_f_1d)
assert_type(_dau, DiscreteAliasUrn)
assert_type(DiscreteAliasUrn(_discr, domain=(0.0, 3.0)), DiscreteAliasUrn)
assert_type(DiscreteAliasUrn(_py_f_1d, urn_factor=2.0, random_state=0), DiscreteAliasUrn)

assert_type(_dau.rvs(), int)
assert_type(_dau.rvs(3), onp.ArrayND[np.int32])

###
# DiscreteGuideTable

_dgt = DiscreteGuideTable(_py_f_1d)
assert_type(_dgt, DiscreteGuideTable)
assert_type(DiscreteGuideTable(_discr, domain=(0.0, 3.0)), DiscreteGuideTable)
assert_type(DiscreteGuideTable(_py_f_1d, guide_factor=1.5, random_state=0), DiscreteGuideTable)

assert_type(_dgt.ppf(0.5), np.float64)
assert_type(_dgt.ppf([0.25, 0.75]), onp.ArrayND[np.float64])
assert_type(_dgt.rvs(), int)
assert_type(_dgt.rvs(3), onp.ArrayND[np.int32])
