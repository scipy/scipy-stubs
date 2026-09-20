from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import (
    Covariance,
    dirichlet,
    dirichlet_multinomial,
    invwishart,
    matrix_normal,
    matrix_t,
    multinomial,
    multivariate_hypergeom,
    multivariate_normal,
    multivariate_t,
    normal_inverse_gamma,
    ortho_group,
    random_correlation,
    random_table,
    special_ortho_group,
    uniform_direction,
    unitary_group,
    vonmises_fisher,
    wishart,
)

###

_i_1d: list[int]
_i_2d: list[list[int]]
_i_3d: list[list[list[int]]]

_i64: np.int64
_i64_2d: onp.Array2D[np.int64]
_i64_nd: onp.ArrayND[np.int64]

_f_1d: list[float] | onp.Array1D[np.float64]
_f_2d: list[list[float]] | onp.Array2D[np.float64]
_f_3d: list[list[list[float]]] | onp.Array3D[np.float64]
_f_4d: onp.Array[tuple[int, int, int, int], np.float64]
_f_nd: onp.ArrayND[np.float64]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_shape_nd: tuple[int, ...]

_cov: Covariance

###

# multivariate_normal

# NOTE: the pyrefly ignores are needed because of https://github.com/facebook/pyrefly/issues/4910

assert_type(multivariate_normal.logpdf(1.0), np.float64)
assert_type(multivariate_normal().logpdf(1.0), np.float64)
assert_type(multivariate_normal.logpdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal().logpdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal.logpdf(_f64_nd, _f_1d), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal(_f_1d).logpdf(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal.logpdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_normal().logpdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_normal.logpdf(_f_1d, None, _f_2d), np.float64)
assert_type(multivariate_normal(None, _f_2d).logpdf(_f_1d), np.float64)
assert_type(multivariate_normal.logpdf(_f_1d, cov=_f_2d), np.float64)
assert_type(multivariate_normal(cov=_f_2d).logpdf(_f_1d), np.float64)
assert_type(multivariate_normal.logpdf(_f_1d, 0.0, _cov), np.float64)
assert_type(multivariate_normal(0.0, _cov).logpdf(_f_1d), np.float64)
assert_type(multivariate_normal.logpdf(_f_1d, _f_1d), np.float64)
assert_type(multivariate_normal(_f_1d).logpdf(_f_1d), np.float64)
assert_type(multivariate_normal.logpdf(_f_2d), onp.Array1D[np.float64])
assert_type(multivariate_normal().logpdf(_f64_2d), onp.Array1D[np.float64])
assert_type(multivariate_normal.logpdf(_f_3d), onp.Array2D[np.float64])
assert_type(multivariate_normal(_f_1d).logpdf(_f_3d), onp.Array2D[np.float64])

assert_type(multivariate_normal.pdf(1.0), np.float64)
assert_type(multivariate_normal().pdf(1.0), np.float64)
assert_type(multivariate_normal.pdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal().pdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal.pdf(_f64_nd, _f_1d), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal(_f_1d).pdf(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal.pdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_normal().pdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_normal.pdf(_f_1d, None, _f_2d), np.float64)
assert_type(multivariate_normal(None, _f_2d).pdf(_f_1d), np.float64)
assert_type(multivariate_normal.pdf(_f_1d, cov=_f_2d), np.float64)
assert_type(multivariate_normal(cov=_f_2d).pdf(_f_1d), np.float64)
assert_type(multivariate_normal.pdf(_f_1d, 0.0, _cov), np.float64)
assert_type(multivariate_normal(0.0, _cov).pdf(_f_1d), np.float64)
assert_type(multivariate_normal.pdf(_f_1d, _f_1d), np.float64)
assert_type(multivariate_normal(_f_1d).pdf(_f_1d), np.float64)
assert_type(multivariate_normal.pdf(_f_2d), onp.Array1D[np.float64])
assert_type(multivariate_normal().pdf(_f64_2d), onp.Array1D[np.float64])
assert_type(multivariate_normal.pdf(_f_3d), onp.Array2D[np.float64])
assert_type(multivariate_normal(_f_1d).pdf(_f_3d), onp.Array2D[np.float64])

assert_type(multivariate_normal.logcdf(1.0), np.float64)
assert_type(multivariate_normal().logcdf(1.0), np.float64)
assert_type(multivariate_normal.logcdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal().logcdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal.logcdf(_f64_nd, _f_1d), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal(_f_1d).logcdf(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal.logcdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_normal().logcdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_normal.logcdf(_f_1d, None, _f_2d), np.float64)
assert_type(multivariate_normal(None, _f_2d).logcdf(_f_1d), np.float64)
assert_type(multivariate_normal.logcdf(_f_1d, cov=_f_2d), np.float64)
assert_type(multivariate_normal(cov=_f_2d).logcdf(_f_1d), np.float64)
assert_type(multivariate_normal.logcdf(_f_1d, 0.0, _cov), np.float64)
assert_type(multivariate_normal(0.0, _cov).logcdf(_f_1d), np.float64)
assert_type(multivariate_normal.logcdf(_f_1d, _f_1d), np.float64)
assert_type(multivariate_normal(_f_1d).logcdf(_f_1d), np.float64)
assert_type(multivariate_normal.logcdf(_f_2d), onp.Array1D[np.float64])
assert_type(multivariate_normal().logcdf(_f64_2d), onp.Array1D[np.float64])
assert_type(multivariate_normal.logcdf(_f_3d), onp.Array2D[np.float64])
assert_type(multivariate_normal(_f_1d).logcdf(_f_3d), onp.Array2D[np.float64])

assert_type(multivariate_normal.cdf(1.0), np.float64)
assert_type(multivariate_normal().cdf(1.0), np.float64)
assert_type(multivariate_normal.cdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal().cdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal.cdf(_f64_nd, _f_1d), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal(_f_1d).cdf(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_normal.cdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_normal().cdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_normal.cdf(_f_1d, None, _f_2d), np.float64)
assert_type(multivariate_normal(None, _f_2d).cdf(_f_1d), np.float64)
assert_type(multivariate_normal.cdf(_f_1d, cov=_f_2d), np.float64)
assert_type(multivariate_normal(cov=_f_2d).cdf(_f_1d), np.float64)
assert_type(multivariate_normal.cdf(_f_1d, 0.0, _cov), np.float64)
assert_type(multivariate_normal(0.0, _cov).cdf(_f_1d), np.float64)
assert_type(multivariate_normal.cdf(_f_1d, _f_1d), np.float64)
assert_type(multivariate_normal(_f_1d).cdf(_f_1d), np.float64)
assert_type(multivariate_normal.cdf(_f_2d), onp.Array1D[np.float64])
assert_type(multivariate_normal().cdf(_f64_2d), onp.Array1D[np.float64])
assert_type(multivariate_normal.cdf(_f_3d), onp.Array2D[np.float64])
assert_type(multivariate_normal(_f_1d).cdf(_f_3d), onp.Array2D[np.float64])

assert_type(multivariate_normal.entropy(), np.float64)
assert_type(multivariate_normal().entropy(), np.float64)

assert_type(multivariate_normal.rvs(), np.float64)
assert_type(multivariate_normal().rvs(), np.float64)
assert_type(multivariate_normal.rvs(_f_1d), np.float64 | onp.ArrayND[np.float64])
assert_type(multivariate_normal(_f_1d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_normal.rvs(None, _f_2d), onp.ArrayND[np.float64])
assert_type(multivariate_normal(None, _f_2d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_normal.rvs(cov=_f_2d), onp.ArrayND[np.float64])
assert_type(multivariate_normal(cov=_f_2d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_normal.rvs(size=(2, 3)), onp.ArrayND[np.float64])
assert_type(multivariate_normal().rvs(size=(2, 3)), onp.ArrayND[np.float64])
assert_type(multivariate_normal.rvs(size=3), np.float64 | onp.ArrayND[np.float64])
assert_type(multivariate_normal().rvs(size=3), onp.Array1D[np.float64])
assert_type(multivariate_normal.rvs(size=_shape_nd), onp.ArrayND[np.float64])
assert_type(multivariate_normal().rvs(size=_shape_nd), onp.ArrayND[np.float64])
assert_type(multivariate_normal(0.0, _cov).rvs(size=(2,)), onp.Array2D[np.float64])
assert_type(multivariate_normal.rvs(_f_1d, size=(2,)), onp.ArrayND[np.float64])
assert_type(multivariate_normal(_f_1d).rvs(size=(2,)), onp.Array2D[np.float64])
assert_type(multivariate_normal.rvs(_f_1d, size=_shape_nd), onp.ArrayND[np.float64])
assert_type(multivariate_normal(_f_1d).rvs(size=_shape_nd), onp.ArrayND[np.float64])

assert_type(multivariate_normal.marginal(0).rvs(), np.float64)
assert_type(multivariate_normal(_f_1d).marginal(0).rvs(), np.float64)
assert_type(multivariate_normal.marginal(_i_1d).rvs(), np.float64)
assert_type(multivariate_normal().marginal(_i_1d).rvs(), np.float64)
assert_type(multivariate_normal.marginal(_i_1d, None, _f_2d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_normal(None, _f_2d).marginal(_i_1d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_normal.marginal(_i_1d, cov=_f_2d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_normal(cov=_f_2d).marginal(_i_1d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_normal.marginal(_i_1d, _f_1d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_normal.marginal(_i_1d, 0.0, _cov).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_normal(0.0, _cov).marginal(_i_1d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_normal(_f_1d).marginal(_i_1d).rvs(), onp.Array1D[np.float64])

# matrix_normal

assert_type(matrix_normal.rvs().dtype, np.dtype[np.float64])
assert_type(matrix_normal().rvs().dtype, np.dtype[np.float64])

# dirichlet

assert_type(dirichlet.rvs([1, 2]).dtype, np.dtype[np.float64])
assert_type(dirichlet([1, 2]).rvs().dtype, np.dtype[np.float64])

assert_type(dirichlet([1, 2]).rvs(size=()), onp.Array1D[np.float64])
assert_type(dirichlet([1, 2]).rvs(size=2), onp.Array2D[np.float64])
assert_type(dirichlet([1, 2]).rvs(size=(2, 3)), onp.Array3D[np.float64])
assert_type(dirichlet.rvs([1, 2], size=_shape_nd), onp.Array[tuple[int, *tuple[Any, ...]], np.float64])
assert_type(dirichlet([1, 2]).rvs(size=_shape_nd), onp.Array[tuple[int, *tuple[Any, ...]], np.float64])

# wishart

assert_type(wishart.rvs(1, 1).dtype, np.dtype[np.float64])
assert_type(wishart().rvs().dtype, np.dtype[np.float64])

# invwishart

assert_type(invwishart.rvs(1, 1).dtype, np.dtype[np.float64])
assert_type(invwishart().rvs().dtype, np.dtype[np.float64])

# multinomial

assert_type(multinomial.mean(1, _f_1d), onp.Array1D[np.float64])
assert_type(multinomial.mean(1, _f_2d), onp.Array2D[np.float64])
assert_type(multinomial.mean(1, _f_3d), onp.Array3D[np.float64])
assert_type(multinomial.mean(1, _f_nd), onp.ArrayND[np.float64])
assert_type(multinomial.mean(_i_1d, _f_1d), onp.Array2D[np.float64])
assert_type(multinomial.mean(_i_1d, _f_2d), onp.Array2D[np.float64])
assert_type(multinomial.mean(_i_1d, _f_3d), onp.Array3D[np.float64])
assert_type(multinomial.mean(_i_1d, _f_nd), onp.ArrayND[np.float64])
assert_type(multinomial.mean(_i_2d, _f_1d), onp.Array3D[np.float64])
assert_type(multinomial.mean(_i_2d, _f_2d), onp.Array3D[np.float64])
assert_type(multinomial.mean(_i_2d, _f_3d), onp.Array3D[np.float64])
assert_type(multinomial.mean(_i_2d, _f_nd), onp.ArrayND[np.float64])
assert_type(multinomial(1, _f_1d).mean(), onp.Array1D[np.float64])
assert_type(multinomial(1, _f_2d).mean(), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_3d).mean(), onp.Array3D[np.float64])
assert_type(multinomial(1, _f_nd).mean(), onp.ArrayND[np.float64])
assert_type(multinomial(_i_1d, _f_1d).mean(), onp.Array2D[np.float64])
assert_type(multinomial(_i_1d, _f_2d).mean(), onp.Array2D[np.float64])
assert_type(multinomial(_i_1d, _f_3d).mean(), onp.Array3D[np.float64])
assert_type(multinomial(_i_1d, _f_nd).mean(), onp.ArrayND[np.float64])
assert_type(multinomial(_i_2d, _f_1d).mean(), onp.Array3D[np.float64])
assert_type(multinomial(_i_2d, _f_2d).mean(), onp.Array3D[np.float64])
assert_type(multinomial(_i_2d, _f_3d).mean(), onp.Array3D[np.float64])
assert_type(multinomial(_i_2d, _f_nd).mean(), onp.ArrayND[np.float64])

assert_type(multinomial.cov(1, _f_1d), onp.Array2D[np.float64])
assert_type(multinomial.cov(1, _f_2d), onp.Array3D[np.float64])
assert_type(multinomial.cov(1, _f_3d), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multinomial.cov(1, _f_nd), onp.ArrayND[np.float64])
assert_type(multinomial.cov(_i_1d, _f_1d), onp.Array3D[np.float64])
assert_type(multinomial.cov(_i_1d, _f_2d), onp.Array3D[np.float64])
assert_type(multinomial.cov(_i_1d, _f_3d), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multinomial.cov(_i_1d, _f_nd), onp.ArrayND[np.float64])
assert_type(multinomial.cov(_i_2d, _f_1d), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multinomial.cov(_i_2d, _f_2d), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multinomial.cov(_i_2d, _f_3d), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multinomial.cov(_i_2d, _f_nd), onp.ArrayND[np.float64])
assert_type(multinomial(1, _f_1d).cov(), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_2d).cov(), onp.Array3D[np.float64])
assert_type(multinomial(1, _f_3d).cov(), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multinomial(1, _f_nd).cov(), onp.ArrayND[np.float64])
assert_type(multinomial(_i_1d, _f_1d).cov(), onp.Array3D[np.float64])
assert_type(multinomial(_i_1d, _f_2d).cov(), onp.Array3D[np.float64])
assert_type(multinomial(_i_1d, _f_3d).cov(), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multinomial(_i_1d, _f_nd).cov(), onp.ArrayND[np.float64])
assert_type(multinomial(_i_2d, _f_1d).cov(), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multinomial(_i_2d, _f_2d).cov(), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multinomial(_i_2d, _f_3d).cov(), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multinomial(_i_2d, _f_nd).cov(), onp.ArrayND[np.float64])

assert_type(multinomial.entropy(1, _f_1d), onp.Array0D[np.float64])
assert_type(multinomial.entropy(1, _f_2d), onp.Array1D[np.float64])
assert_type(multinomial.entropy(1, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.entropy(1, _f_nd), onp.ArrayND[np.float64])
assert_type(multinomial.entropy(_i_1d, _f_1d), onp.Array1D[np.float64])
assert_type(multinomial.entropy(_i_1d, _f_2d), onp.Array1D[np.float64])
assert_type(multinomial.entropy(_i_1d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.entropy(_i_1d, _f_nd), onp.ArrayND[np.float64])
assert_type(multinomial.entropy(_i_2d, _f_1d), onp.Array2D[np.float64])
assert_type(multinomial.entropy(_i_2d, _f_2d), onp.Array2D[np.float64])
assert_type(multinomial.entropy(_i_2d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.entropy(_i_2d, _f_nd), onp.ArrayND[np.float64])
assert_type(multinomial(1, _f_1d).entropy(), onp.Array0D[np.float64])
assert_type(multinomial(1, _f_2d).entropy(), onp.Array1D[np.float64])
assert_type(multinomial(1, _f_3d).entropy(), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_nd).entropy(), onp.ArrayND[np.float64])
assert_type(multinomial(_i_1d, _f_1d).entropy(), onp.Array1D[np.float64])
assert_type(multinomial(_i_1d, _f_2d).entropy(), onp.Array1D[np.float64])
assert_type(multinomial(_i_1d, _f_3d).entropy(), onp.Array2D[np.float64])
assert_type(multinomial(_i_1d, _f_nd).entropy(), onp.ArrayND[np.float64])
assert_type(multinomial(_i_2d, _f_1d).entropy(), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_2d).entropy(), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_3d).entropy(), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_nd).entropy(), onp.ArrayND[np.float64])

assert_type(multinomial.rvs(1, _f_1d), onp.Array1D[np.int_])  # `multinomial_gen.rvs` defaults to `size=None`
assert_type(multinomial.rvs(1, _f_1d, size=()), onp.Array1D[np.int_])
assert_type(multinomial.rvs(1, _f_1d, size=1), onp.Array2D[np.int_])
assert_type(multinomial.rvs(1, _f_1d, size=(1,)), onp.Array2D[np.int_])
assert_type(multinomial.rvs(1, _f_1d, size=(1, 2)), onp.Array3D[np.int_])
assert_type(multinomial.rvs(1, _f_1d, size=(1, 2, 4, 8, 16, 31)), onp.ArrayND[np.int_])
assert_type(multinomial(1, _f_1d).rvs(), onp.Array2D[np.int_])  # `multinomial_frozen.rvs` defaults to `size=1`
assert_type(multinomial(1, _f_1d).rvs(size=()), onp.Array1D[np.int_])
assert_type(multinomial(1, _f_1d).rvs(size=1), onp.Array2D[np.int_])
assert_type(multinomial(1, _f_1d).rvs(size=(1,)), onp.Array2D[np.int_])
assert_type(multinomial(1, _f_1d).rvs(size=(1, 2)), onp.Array3D[np.int_])
assert_type(multinomial(1, _f_1d).rvs(size=(1, 2, 4, 8, 16, 31)), onp.ArrayND[np.int_])

assert_type(multinomial.logpmf(_i_1d, 1, _f_1d), onp.Array0D[np.float64])
assert_type(multinomial.logpmf(_i_2d, 1, _f_1d), onp.Array1D[np.float64])
assert_type(multinomial.logpmf(_i_3d, 1, _f_1d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_1d, 1, _f_2d), onp.Array1D[np.float64])
assert_type(multinomial.logpmf(_i_2d, 1, _f_2d), onp.Array1D[np.float64])
assert_type(multinomial.logpmf(_i_3d, 1, _f_2d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_1d, 1, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_2d, 1, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_3d, 1, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_1d, _i_1d, _f_1d), onp.Array1D[np.float64])
assert_type(multinomial.logpmf(_i_2d, _i_1d, _f_1d), onp.Array1D[np.float64])
assert_type(multinomial.logpmf(_i_3d, _i_1d, _f_1d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_1d, _i_1d, _f_2d), onp.Array1D[np.float64])
assert_type(multinomial.logpmf(_i_2d, _i_1d, _f_2d), onp.Array1D[np.float64])
assert_type(multinomial.logpmf(_i_3d, _i_1d, _f_2d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_1d, _i_1d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_2d, _i_1d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_3d, _i_1d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_1d, _i_2d, _f_1d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_2d, _i_2d, _f_1d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_3d, _i_2d, _f_1d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_1d, _i_2d, _f_2d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_2d, _i_2d, _f_2d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_3d, _i_2d, _f_2d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_1d, _i_2d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_2d, _i_2d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.logpmf(_i_3d, _i_2d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_1d).logpmf(_i_1d), onp.Array0D[np.float64])
assert_type(multinomial(1, _f_1d).logpmf(_i_2d), onp.Array1D[np.float64])
assert_type(multinomial(1, _f_1d).logpmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_2d).logpmf(_i_1d), onp.Array1D[np.float64])
assert_type(multinomial(1, _f_2d).logpmf(_i_2d), onp.Array1D[np.float64])
assert_type(multinomial(1, _f_2d).logpmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_3d).logpmf(_i_1d), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_3d).logpmf(_i_2d), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_3d).logpmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(_i_1d, _f_1d).logpmf(_i_1d), onp.Array1D[np.float64])
assert_type(multinomial(_i_1d, _f_1d).logpmf(_i_2d), onp.Array1D[np.float64])
assert_type(multinomial(_i_1d, _f_1d).logpmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(_i_1d, _f_2d).logpmf(_i_1d), onp.Array1D[np.float64])
assert_type(multinomial(_i_1d, _f_2d).logpmf(_i_2d), onp.Array1D[np.float64])
assert_type(multinomial(_i_1d, _f_2d).logpmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(_i_1d, _f_3d).logpmf(_i_1d), onp.Array2D[np.float64])
assert_type(multinomial(_i_1d, _f_3d).logpmf(_i_2d), onp.Array2D[np.float64])
assert_type(multinomial(_i_1d, _f_3d).logpmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_1d).logpmf(_i_1d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_1d).logpmf(_i_2d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_1d).logpmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_2d).logpmf(_i_1d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_2d).logpmf(_i_2d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_2d).logpmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_3d).logpmf(_i_1d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_3d).logpmf(_i_2d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_3d).logpmf(_i_3d), onp.Array2D[np.float64])

assert_type(multinomial.pmf(_i_1d, 1, _f_1d), np.float64)
assert_type(multinomial.pmf(_i_2d, 1, _f_1d), onp.Array1D[np.float64])
assert_type(multinomial.pmf(_i_3d, 1, _f_1d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_1d, 1, _f_2d), onp.Array1D[np.float64])
assert_type(multinomial.pmf(_i_2d, 1, _f_2d), onp.Array1D[np.float64])
assert_type(multinomial.pmf(_i_3d, 1, _f_2d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_1d, 1, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_2d, 1, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_3d, 1, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_1d, _i_1d, _f_1d), onp.Array1D[np.float64])
assert_type(multinomial.pmf(_i_2d, _i_1d, _f_1d), onp.Array1D[np.float64])
assert_type(multinomial.pmf(_i_3d, _i_1d, _f_1d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_1d, _i_1d, _f_2d), onp.Array1D[np.float64])
assert_type(multinomial.pmf(_i_2d, _i_1d, _f_2d), onp.Array1D[np.float64])
assert_type(multinomial.pmf(_i_3d, _i_1d, _f_2d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_1d, _i_1d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_2d, _i_1d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_3d, _i_1d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_1d, _i_2d, _f_1d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_2d, _i_2d, _f_1d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_3d, _i_2d, _f_1d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_1d, _i_2d, _f_2d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_2d, _i_2d, _f_2d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_3d, _i_2d, _f_2d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_1d, _i_2d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_2d, _i_2d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial.pmf(_i_3d, _i_2d, _f_3d), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_1d).pmf(_i_1d), np.float64)
assert_type(multinomial(1, _f_1d).pmf(_i_2d), onp.Array1D[np.float64])
assert_type(multinomial(1, _f_1d).pmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_2d).pmf(_i_1d), onp.Array1D[np.float64])
assert_type(multinomial(1, _f_2d).pmf(_i_2d), onp.Array1D[np.float64])
assert_type(multinomial(1, _f_2d).pmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_3d).pmf(_i_1d), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_3d).pmf(_i_2d), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_3d).pmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(_i_1d, _f_1d).pmf(_i_1d), onp.Array1D[np.float64])
assert_type(multinomial(_i_1d, _f_1d).pmf(_i_2d), onp.Array1D[np.float64])
assert_type(multinomial(_i_1d, _f_1d).pmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(_i_1d, _f_2d).pmf(_i_1d), onp.Array1D[np.float64])
assert_type(multinomial(_i_1d, _f_2d).pmf(_i_2d), onp.Array1D[np.float64])
assert_type(multinomial(_i_1d, _f_2d).pmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(_i_1d, _f_3d).pmf(_i_1d), onp.Array2D[np.float64])
assert_type(multinomial(_i_1d, _f_3d).pmf(_i_2d), onp.Array2D[np.float64])
assert_type(multinomial(_i_1d, _f_3d).pmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_1d).pmf(_i_1d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_1d).pmf(_i_2d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_1d).pmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_2d).pmf(_i_1d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_2d).pmf(_i_2d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_2d).pmf(_i_3d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_3d).pmf(_i_1d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_3d).pmf(_i_2d), onp.Array2D[np.float64])
assert_type(multinomial(_i_2d, _f_3d).pmf(_i_3d), onp.Array2D[np.float64])

# ortho_group

assert_type(ortho_group.rvs(3).dtype, np.dtype[np.float64])
assert_type(ortho_group().rvs(3).dtype, np.dtype[np.float64])

# special_ortho_group

assert_type(special_ortho_group.rvs(3).dtype, np.dtype[np.float64])
assert_type(special_ortho_group().rvs(3).dtype, np.dtype[np.float64])

# unitary_group

assert_type(unitary_group.rvs(3).dtype, np.dtype[np.complex128])
assert_type(unitary_group().rvs(3).dtype, np.dtype[np.complex128])

# uniform_direction

assert_type(uniform_direction.rvs(2).dtype, np.dtype[np.float64])
assert_type(uniform_direction(2).rvs().dtype, np.dtype[np.float64])

assert_type(uniform_direction.rvs(2, size=_i64), onp.Array2D[np.float64])
assert_type(uniform_direction(2).rvs(size=_i64), onp.Array2D[np.float64])
assert_type(uniform_direction.rvs(2, size=(2,)), onp.Array2D[np.float64])
assert_type(uniform_direction(2).rvs(size=(2,)), onp.Array2D[np.float64])
assert_type(uniform_direction.rvs(2, size=(2, 3)), onp.Array3D[np.float64])
assert_type(uniform_direction(2).rvs(size=(2, 3)), onp.Array3D[np.float64])
assert_type(uniform_direction.rvs(2, size=(2, 3, 4)), onp.Array[tuple[int, int, int, *tuple[Any, ...]], np.float64])
assert_type(uniform_direction(2).rvs(size=(2, 3, 4)), onp.Array[tuple[int, int, int, *tuple[Any, ...]], np.float64])
assert_type(uniform_direction.rvs(2, size=_shape_nd), onp.Array[tuple[int, int, *tuple[Any, ...]], np.float64])
assert_type(uniform_direction(2).rvs(size=_shape_nd), onp.Array[tuple[int, int, *tuple[Any, ...]], np.float64])

# random_correlation

assert_type(random_correlation.rvs([1, 1]).dtype, np.dtype[np.float64])
assert_type(random_correlation([1, 1]).rvs().dtype, np.dtype[np.float64])

# multivariate_t

assert_type(multivariate_t.logpdf(1.0), np.float64)
assert_type(multivariate_t().logpdf(1.0), np.float64)
assert_type(multivariate_t.logpdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_t().logpdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_t.logpdf(_f64_nd, _f_1d), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_t(_f_1d).logpdf(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_t.logpdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_t().logpdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_t.logpdf(_f_1d, None, _f_2d), np.float64)
assert_type(multivariate_t(None, _f_2d).logpdf(_f_1d), np.float64)
assert_type(multivariate_t.logpdf(_f_1d, shape=_f_2d), np.float64)
assert_type(multivariate_t(shape=_f_2d).logpdf(_f_1d), np.float64)
assert_type(multivariate_t.logpdf(_f_1d, _f_1d), np.float64)
assert_type(multivariate_t(_f_1d).logpdf(_f_1d), np.float64)
assert_type(multivariate_t.logpdf(_f_2d), onp.Array1D[np.float64])
assert_type(multivariate_t().logpdf(_f64_2d), onp.Array1D[np.float64])
assert_type(multivariate_t.logpdf(_f_3d), onp.Array2D[np.float64])
assert_type(multivariate_t(_f_1d).logpdf(_f_3d), onp.Array2D[np.float64])

assert_type(multivariate_t.pdf(1.0), np.float64)
assert_type(multivariate_t().pdf(1.0), np.float64)
assert_type(multivariate_t.pdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_t().pdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_t.pdf(_f64_nd, _f_1d), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_t(_f_1d).pdf(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_t.pdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_t().pdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_t.pdf(_f_1d, None, _f_2d), np.float64)
assert_type(multivariate_t(None, _f_2d).pdf(_f_1d), np.float64)
assert_type(multivariate_t.pdf(_f_1d, shape=_f_2d), np.float64)
assert_type(multivariate_t(shape=_f_2d).pdf(_f_1d), np.float64)
assert_type(multivariate_t.pdf(_f_1d, _f_1d), np.float64)
assert_type(multivariate_t(_f_1d).pdf(_f_1d), np.float64)
assert_type(multivariate_t.pdf(_f_2d), onp.Array1D[np.float64])
assert_type(multivariate_t().pdf(_f64_2d), onp.Array1D[np.float64])
assert_type(multivariate_t.pdf(_f_3d), onp.Array2D[np.float64])
assert_type(multivariate_t(_f_1d).pdf(_f_3d), onp.Array2D[np.float64])

assert_type(multivariate_t.cdf(1.0), np.float64)
assert_type(multivariate_t().cdf(1.0), np.float64)
assert_type(multivariate_t.cdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_t().cdf(_f64_nd), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_t.cdf(_f64_nd, _f_1d), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_t(_f_1d).cdf(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_t.cdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_t().cdf(_f64_1d), onp.Array1D[np.float64])
assert_type(multivariate_t.cdf(_f_1d, None, _f_2d), np.float64)
assert_type(multivariate_t(None, _f_2d).cdf(_f_1d), np.float64)
assert_type(multivariate_t.cdf(_f_1d, shape=_f_2d), np.float64)
assert_type(multivariate_t(shape=_f_2d).cdf(_f_1d), np.float64)
assert_type(multivariate_t.cdf(_f_1d, _f_1d), np.float64)
assert_type(multivariate_t(_f_1d).cdf(_f_1d), np.float64)
assert_type(multivariate_t.cdf(_f_2d), onp.Array1D[np.float64])
assert_type(multivariate_t().cdf(_f64_2d), onp.Array1D[np.float64])
assert_type(multivariate_t.cdf(_f_3d), onp.Array2D[np.float64])
assert_type(multivariate_t(_f_1d).cdf(_f_3d), onp.Array2D[np.float64])

assert_type(multivariate_t.entropy(), onp.Array0D[np.float64])
assert_type(multivariate_t().entropy(), onp.Array0D[np.float64])

assert_type(multivariate_t.rvs(), np.float64)
assert_type(multivariate_t().rvs(), np.float64)
assert_type(multivariate_t.rvs(_f_1d), np.float64 | onp.ArrayND[np.float64])
assert_type(multivariate_t(_f_1d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_t.rvs(None, _f_2d), onp.ArrayND[np.float64])
assert_type(multivariate_t(None, _f_2d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_t.rvs(shape=_f_2d), onp.ArrayND[np.float64])
assert_type(multivariate_t(shape=_f_2d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_t.rvs(size=(2, 3)), onp.ArrayND[np.float64])
assert_type(multivariate_t().rvs(size=(2, 3)), onp.ArrayND[np.float64])
assert_type(multivariate_t.rvs(size=3), np.float64 | onp.ArrayND[np.float64])
assert_type(multivariate_t().rvs(size=3), onp.Array1D[np.float64])
assert_type(multivariate_t.rvs(size=_shape_nd), onp.ArrayND[np.float64])
assert_type(multivariate_t().rvs(size=_shape_nd), onp.ArrayND[np.float64])
assert_type(multivariate_t.rvs(_f_1d, size=(2,)), onp.ArrayND[np.float64])
assert_type(multivariate_t(_f_1d).rvs(size=(2,)), onp.Array2D[np.float64])
assert_type(multivariate_t.rvs(_f_1d, size=_shape_nd), onp.ArrayND[np.float64])
assert_type(multivariate_t(_f_1d).rvs(size=_shape_nd), onp.ArrayND[np.float64])

assert_type(multivariate_t.marginal(0).rvs(), np.float64)
assert_type(multivariate_t(_f_1d).marginal(0).rvs(), np.float64)
assert_type(multivariate_t.marginal(_i_1d).rvs(), np.float64)
assert_type(multivariate_t().marginal(_i_1d).rvs(), np.float64)
assert_type(multivariate_t.marginal(_i_1d, None, _f_2d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_t(None, _f_2d).marginal(_i_1d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_t.marginal(_i_1d, shape=_f_2d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_t(shape=_f_2d).marginal(_i_1d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_t.marginal(_i_1d, _f_1d).rvs(), onp.Array1D[np.float64])
assert_type(multivariate_t(_f_1d).marginal(_i_1d).rvs(), onp.Array1D[np.float64])

# multivariate_hypergeom

assert_type(multivariate_hypergeom.logpmf(_i64_nd, _i_1d, 1), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_hypergeom(_i_1d, 1).logpmf(_i64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_hypergeom.logpmf(_i_1d, _i64_nd, 1), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_hypergeom(_i64_nd, 1).logpmf(_i_1d), np.float64 | onp.ArrayND[np.float64])
assert_type(multivariate_hypergeom.logpmf(_i_1d, _i_1d, _i64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_hypergeom(_i_1d, _i64_nd).logpmf(_i_1d), np.float64 | onp.ArrayND[np.float64])
assert_type(multivariate_hypergeom.logpmf(_i_1d, _i_1d, 1), np.float64)
assert_type(multivariate_hypergeom(_i_1d, 1).logpmf(_i_1d), np.float64)
assert_type(multivariate_hypergeom.logpmf(_i_1d, _i_2d, 1), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom(_i_2d, 1).logpmf(_i_1d), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom.logpmf(_i_1d, _i_1d, _i_1d), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, _i_1d).logpmf(_i_1d), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom.logpmf(_i_1d, _i_3d, 1), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_3d, 1).logpmf(_i_1d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.logpmf(_i_2d, _i_1d, _i_2d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, _i_2d).logpmf(_i_2d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.logpmf(_i_2d, _i_1d, 1), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, 1).logpmf(_i_2d), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom.logpmf(_i_2d, _i_2d, _i_1d), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom(_i_2d, _i_1d).logpmf(_i_2d), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom.logpmf(_i_3d, _i_1d, 1), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, 1).logpmf(_i_3d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.logpmf(_i_3d, _i_2d, _i_1d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_2d, _i_1d).logpmf(_i_3d), onp.Array2D[np.float64])

assert_type(multivariate_hypergeom.pmf(_i64_nd, _i_1d, 1), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_hypergeom(_i_1d, 1).pmf(_i64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_hypergeom.pmf(_i_1d, _i64_nd, 1), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_hypergeom(_i64_nd, 1).pmf(_i_1d), np.float64 | onp.ArrayND[np.float64])
assert_type(multivariate_hypergeom.pmf(_i_1d, _i_1d, _i64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_hypergeom(_i_1d, _i64_nd).pmf(_i_1d), np.float64 | onp.ArrayND[np.float64])
assert_type(multivariate_hypergeom.pmf(_i_1d, _i_1d, 1), np.float64)
assert_type(multivariate_hypergeom(_i_1d, 1).pmf(_i_1d), np.float64)
assert_type(multivariate_hypergeom.pmf(_i_1d, _i_2d, 1), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom(_i_2d, 1).pmf(_i_1d), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom.pmf(_i_1d, _i_1d, _i_1d), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, _i_1d).pmf(_i_1d), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom.pmf(_i_1d, _i_3d, 1), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_3d, 1).pmf(_i_1d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.pmf(_i_2d, _i_1d, _i_2d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, _i_2d).pmf(_i_2d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.pmf(_i_2d, _i_1d, 1), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, 1).pmf(_i_2d), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom.pmf(_i_2d, _i_2d, _i_1d), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom(_i_2d, _i_1d).pmf(_i_2d), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom.pmf(_i_3d, _i_1d, 1), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, 1).pmf(_i_3d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.pmf(_i_3d, _i_2d, _i_1d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_2d, _i_1d).pmf(_i_3d), onp.Array2D[np.float64])

assert_type(multivariate_hypergeom.mean(_i64_2d, 1), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i64_2d, 1).mean(), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.mean(_i_1d, 1), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, 1).mean(), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom.mean(_i_2d, 1), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_2d, 1).mean(), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.mean(_i_3d, 1), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom(_i_3d, 1).mean(), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom.mean(_i64_nd, _i_1d), onp.Array[tuple[int, *tuple[Any, ...]], np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_hypergeom(_i64_nd, _i_1d).mean(), onp.ArrayND[np.float64])
assert_type(multivariate_hypergeom.mean(_i_1d, _i64_nd), onp.Array[tuple[int, *tuple[Any, ...]], np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_hypergeom(_i_1d, _i64_nd).mean(), onp.ArrayND[np.float64])
assert_type(multivariate_hypergeom.mean(_i_1d, _i_1d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, _i_1d).mean(), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.mean(_i_2d, _i_1d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_2d, _i_1d).mean(), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.mean(_i_3d, _i_1d), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom(_i_3d, _i_1d).mean(), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom.mean(_i_1d, _i_2d), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, _i_2d).mean(), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom.mean(_i_3d, _i_2d), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom(_i_3d, _i_2d).mean(), onp.Array3D[np.float64])

assert_type(multivariate_hypergeom.var(_i64_2d, 1), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i64_2d, 1).var(), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.var(_i_1d, 1), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, 1).var(), onp.Array1D[np.float64])
assert_type(multivariate_hypergeom.var(_i_2d, 1), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_2d, 1).var(), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.var(_i_3d, 1), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom(_i_3d, 1).var(), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom.var(_i64_nd, _i_1d), onp.Array[tuple[int, *tuple[Any, ...]], np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_hypergeom(_i64_nd, _i_1d).var(), onp.ArrayND[np.float64])
assert_type(multivariate_hypergeom.var(_i_1d, _i64_nd), onp.Array[tuple[int, *tuple[Any, ...]], np.float64])  # pyrefly:ignore[assert-type]
assert_type(multivariate_hypergeom(_i_1d, _i64_nd).var(), onp.ArrayND[np.float64])
assert_type(multivariate_hypergeom.var(_i_1d, _i_1d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, _i_1d).var(), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.var(_i_2d, _i_1d), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_2d, _i_1d).var(), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.var(_i_3d, _i_1d), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom(_i_3d, _i_1d).var(), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom.var(_i_1d, _i_2d), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, _i_2d).var(), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom.var(_i_3d, _i_2d), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom(_i_3d, _i_2d).var(), onp.Array3D[np.float64])

assert_type(multivariate_hypergeom.cov(_i64_nd, 1), onp.Array[tuple[int, int, *tuple[Any, ...]], np.float64])
assert_type(multivariate_hypergeom(_i64_nd, 1).cov(), onp.Array[tuple[int, int, *tuple[Any, ...]], np.float64])
assert_type(multivariate_hypergeom.cov(_i_1d, _i64_nd), onp.Array[tuple[int, int, *tuple[Any, ...]], np.float64])
assert_type(multivariate_hypergeom(_i_1d, _i64_nd).cov(), onp.Array[tuple[int, int, *tuple[Any, ...]], np.float64])
assert_type(multivariate_hypergeom.cov(_i_1d, 1), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, 1).cov(), onp.Array2D[np.float64])
assert_type(multivariate_hypergeom.cov(_i_2d, 1), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom(_i_2d, 1).cov(), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom.cov(_i_3d, 1), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multivariate_hypergeom(_i_3d, 1).cov(), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multivariate_hypergeom.cov(_i_1d, _i_1d), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom(_i_1d, _i_1d).cov(), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom.cov(_i_2d, _i_1d), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom(_i_2d, _i_1d).cov(), onp.Array3D[np.float64])
assert_type(multivariate_hypergeom.cov(_i_3d, _i_1d), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multivariate_hypergeom(_i_3d, _i_1d).cov(), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multivariate_hypergeom.cov(_i_1d, _i_2d), onp.ArrayND[np.float64, tuple[int, int, int, int]])
assert_type(multivariate_hypergeom(_i_1d, _i_2d).cov(), onp.ArrayND[np.float64, tuple[int, int, int, int]])

assert_type(multivariate_hypergeom.rvs(_i_1d, 1).dtype, np.dtype[np.int_])
assert_type(multivariate_hypergeom(_i_1d, 1).rvs().dtype, np.dtype[np.int_])
assert_type(multivariate_hypergeom.rvs(_i_1d, 1), onp.Array1D[np.int_])
assert_type(multivariate_hypergeom.rvs(_i_2d, 1), onp.Array2D[np.int_])
assert_type(multivariate_hypergeom.rvs(_i_3d, 1), onp.Array3D[np.int_])
assert_type(multivariate_hypergeom.rvs(_i64_nd, _i_1d), onp.Array[tuple[int, *tuple[Any, ...]], np.int_])
assert_type(multivariate_hypergeom.rvs(_i_1d, _i64_nd), onp.Array[tuple[int, *tuple[Any, ...]], np.int_])
assert_type(multivariate_hypergeom.rvs(_i_1d, _i_1d), onp.Array2D[np.int_])
assert_type(multivariate_hypergeom.rvs(_i_3d, _i_1d), onp.Array3D[np.int_])
assert_type(multivariate_hypergeom.rvs(_i_1d, _i_2d), onp.Array3D[np.int_])
assert_type(multivariate_hypergeom.rvs(_i_1d, 1, size=_shape_nd), onp.Array[tuple[int, *tuple[Any, ...]], np.int_])
assert_type(multivariate_hypergeom(_i_1d, 1).rvs(size=_shape_nd), onp.Array[tuple[int, *tuple[Any, ...]], np.int_])
assert_type(multivariate_hypergeom.rvs(_i_1d, 1, size=()), onp.Array1D[np.int_])
assert_type(multivariate_hypergeom(_i_1d, 1).rvs(size=()), onp.Array1D[np.int_])
assert_type(multivariate_hypergeom.rvs(_i_1d, 1, size=3), onp.Array2D[np.int_])
assert_type(multivariate_hypergeom(_i_1d, 1).rvs(), onp.Array2D[np.int_])
assert_type(multivariate_hypergeom(_i_1d, 1).rvs(size=3), onp.Array2D[np.int_])
assert_type(multivariate_hypergeom.rvs(_i_1d, 1, size=(2, 3)), onp.Array3D[np.int_])
assert_type(multivariate_hypergeom(_i_1d, 1).rvs(size=(2, 3)), onp.Array3D[np.int_])

# random_table

assert_type(random_table.rvs([1, 2], [2, 1]).dtype, np.dtype[np.int_])
assert_type(random_table([1, 2], [2, 1]).rvs().dtype, np.dtype[np.int_])

assert_type(random_table.rvs([1, 2], [2, 1], size=_shape_nd), onp.Array[tuple[int, int, int, *tuple[Any, ...]], np.int_])
assert_type(random_table([1, 2], [2, 1]).rvs(size=(2,)), onp.Array3D[np.int_])
assert_type(random_table([1, 2], [2, 1]).rvs(size=_shape_nd), onp.Array[tuple[int, int, int, *tuple[Any, ...]], np.int_])

# dirichlet_multinomial

assert_type(dirichlet_multinomial.pmf([1, 2], [0.5, 0.5], [3]), np.float64 | onp.Array[tuple[int, *tuple[Any, ...]], np.float64])
assert_type(dirichlet_multinomial([0.5, 0.5], [3]).pmf([1, 2]), np.float64 | onp.Array[tuple[int, *tuple[Any, ...]], np.float64])

# matrix_t

assert_type(matrix_t.rvs(df=1).dtype, np.dtype[np.float64])
assert_type(matrix_t(df=1).rvs().dtype, np.dtype[np.float64])

# vonmises_fisher

assert_type(vonmises_fisher.rvs([0.8, 0.6]).dtype, np.dtype[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).rvs().dtype, np.dtype[np.float64])

assert_type(vonmises_fisher.logpdf(_f64_nd, [0.8, 0.6]), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(vonmises_fisher([0.8, 0.6]).logpdf(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(vonmises_fisher.logpdf(_f_1d, [0.8, 0.6]), np.float64)
assert_type(vonmises_fisher([0.8, 0.6]).logpdf(_f_1d), np.float64)
assert_type(vonmises_fisher.logpdf(_f_2d, [0.8, 0.6]), onp.Array1D[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).logpdf(_f_2d), onp.Array1D[np.float64])
assert_type(vonmises_fisher.logpdf(_f_3d, [0.8, 0.6]), onp.Array2D[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).logpdf(_f_3d), onp.Array2D[np.float64])
assert_type(vonmises_fisher.logpdf(_f_4d, [0.8, 0.6]), np.float64 | onp.ArrayND[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).logpdf(_f_4d), np.float64 | onp.ArrayND[np.float64])

assert_type(vonmises_fisher.pdf(_f64_nd, [0.8, 0.6]), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(vonmises_fisher([0.8, 0.6]).pdf(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(vonmises_fisher.pdf(_f_1d, [0.8, 0.6]), np.float64)
assert_type(vonmises_fisher([0.8, 0.6]).pdf(_f_1d), np.float64)
assert_type(vonmises_fisher.pdf(_f_2d, [0.8, 0.6]), onp.Array1D[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).pdf(_f_2d), onp.Array1D[np.float64])
assert_type(vonmises_fisher.pdf(_f_3d, [0.8, 0.6]), onp.Array2D[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).pdf(_f_3d), onp.Array2D[np.float64])
assert_type(vonmises_fisher.pdf(_f_4d, [0.8, 0.6]), np.float64 | onp.ArrayND[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).pdf(_f_4d), np.float64 | onp.ArrayND[np.float64])

assert_type(vonmises_fisher.rvs([0.8, 0.6], size=()), onp.Array1D[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).rvs(size=()), onp.Array1D[np.float64])
assert_type(vonmises_fisher.rvs([0.8, 0.6], size=None), onp.Array1D[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).rvs(size=None), onp.Array1D[np.float64])
assert_type(vonmises_fisher.rvs([0.8, 0.6]), onp.Array2D[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).rvs(), onp.Array2D[np.float64])
assert_type(vonmises_fisher.rvs([0.8, 0.6], 1, 3), onp.Array2D[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).rvs(3), onp.Array2D[np.float64])
assert_type(vonmises_fisher.rvs([0.8, 0.6], size=_i64), onp.Array2D[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).rvs(size=_i64), onp.Array2D[np.float64])
assert_type(vonmises_fisher.rvs([0.8, 0.6], size=(3,)), onp.Array2D[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).rvs(size=(3,)), onp.Array2D[np.float64])
assert_type(vonmises_fisher.rvs([0.8, 0.6], size=(2, 3)), onp.Array3D[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).rvs(size=(2, 3)), onp.Array3D[np.float64])
assert_type(vonmises_fisher.rvs([0.8, 0.6], size=(2, 3, 4)), onp.Array[tuple[int, int, int, *tuple[Any, ...]], np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).rvs(size=(2, 3, 4)), onp.Array[tuple[int, int, int, *tuple[Any, ...]], np.float64])
assert_type(vonmises_fisher.rvs([0.8, 0.6], size=_shape_nd), onp.Array[tuple[int, *tuple[Any, ...]], np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).rvs(size=_shape_nd), onp.Array[tuple[int, *tuple[Any, ...]], np.float64])
assert_type(vonmises_fisher.rvs([0.8, 0.6], kappa=0.5).dtype, np.dtype[np.float64])
assert_type(vonmises_fisher([0.8, 0.6], kappa=0.5).rvs().dtype, np.dtype[np.float64])

# normal_inverse_gamma

assert_type(normal_inverse_gamma.rvs()[0].dtype, np.dtype[np.float64])
assert_type(normal_inverse_gamma().rvs()[0].dtype, np.dtype[np.float64])

assert_type(normal_inverse_gamma.mean(), tuple[np.float64, np.float64])
assert_type(  # pyrefly:ignore[assert-type]
    normal_inverse_gamma.mean(_f64_nd), tuple[np.float64, np.float64] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]
)
assert_type(  # pyrefly:ignore[assert-type]
    normal_inverse_gamma.mean(lmbda=_f64_nd),
    tuple[np.float64, np.float64] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]],
)
assert_type(  # pyrefly:ignore[assert-type]
    normal_inverse_gamma.mean(a=_f64_nd), tuple[np.float64, np.float64] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]
)
assert_type(  # pyrefly:ignore[assert-type]
    normal_inverse_gamma.mean(b=_f64_nd), tuple[np.float64, np.float64] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]
)
assert_type(normal_inverse_gamma.mean(mu=_f_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(normal_inverse_gamma.mean(mu=_f_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(normal_inverse_gamma.mean(mu=_f_3d), tuple[onp.Array3D[np.float64], onp.Array3D[np.float64]])

assert_type(normal_inverse_gamma.var(), tuple[np.float64, np.float64])
assert_type(  # pyrefly:ignore[assert-type]
    normal_inverse_gamma.var(_f64_nd), tuple[np.float64, np.float64] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]
)
assert_type(  # pyrefly:ignore[assert-type]
    normal_inverse_gamma.var(lmbda=_f64_nd),
    tuple[np.float64, np.float64] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]],
)
assert_type(  # pyrefly:ignore[assert-type]
    normal_inverse_gamma.var(a=_f64_nd), tuple[np.float64, np.float64] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]
)
assert_type(  # pyrefly:ignore[assert-type]
    normal_inverse_gamma.var(b=_f64_nd), tuple[np.float64, np.float64] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]
)
assert_type(normal_inverse_gamma.var(mu=_f_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(normal_inverse_gamma.var(mu=_f_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(normal_inverse_gamma.var(mu=_f_3d), tuple[onp.Array3D[np.float64], onp.Array3D[np.float64]])

assert_type(normal_inverse_gamma().mean(), tuple[np.float64, np.float64])
assert_type(
    normal_inverse_gamma(_f64_nd).mean(), tuple[np.float64, np.float64] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]
)
assert_type(
    normal_inverse_gamma(lmbda=_f64_nd).mean(),
    tuple[np.float64, np.float64] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]],
)
assert_type(
    normal_inverse_gamma(a=_f64_nd).mean(),
    tuple[np.float64, np.float64] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]],
)
assert_type(
    normal_inverse_gamma(b=_f64_nd).mean(),
    tuple[np.float64, np.float64] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]],
)
assert_type(normal_inverse_gamma(mu=_f_1d).mean(), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(normal_inverse_gamma(mu=_f_2d).mean(), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(normal_inverse_gamma(mu=_f_3d).mean(), tuple[onp.Array3D[np.float64], onp.Array3D[np.float64]])
assert_type(normal_inverse_gamma(_f_1d).var(), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(normal_inverse_gamma(_f_2d).var(), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(normal_inverse_gamma(_f_3d).var(), tuple[onp.Array3D[np.float64], onp.Array3D[np.float64]])

assert_type(normal_inverse_gamma.logpdf(1.0, 1.0), np.float64)
assert_type(normal_inverse_gamma.logpdf(_f64_nd, 1.0), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(normal_inverse_gamma.logpdf(1.0, _f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(normal_inverse_gamma.logpdf(1.0, 1.0, _f64_nd), np.float64 | onp.ArrayND[np.float64])
assert_type(normal_inverse_gamma.logpdf(1.0, 1.0, lmbda=_f64_nd), np.float64 | onp.ArrayND[np.float64])
assert_type(normal_inverse_gamma.logpdf(1.0, 1.0, a=_f64_nd), np.float64 | onp.ArrayND[np.float64])
assert_type(normal_inverse_gamma.logpdf(1.0, 1.0, b=_f64_nd), np.float64 | onp.ArrayND[np.float64])
assert_type(normal_inverse_gamma.logpdf(_f_1d, 1.0), onp.Array1D[np.float64])
assert_type(normal_inverse_gamma.logpdf(1.0, _f_1d), onp.Array1D[np.float64])
assert_type(normal_inverse_gamma.logpdf(_f_2d, 1.0), onp.Array2D[np.float64])
assert_type(normal_inverse_gamma.logpdf(1.0, _f_2d), onp.Array2D[np.float64])
assert_type(normal_inverse_gamma.logpdf(_f_3d, 1.0), onp.Array3D[np.float64])
assert_type(normal_inverse_gamma.logpdf(1.0, _f_3d), onp.Array3D[np.float64])

assert_type(normal_inverse_gamma.pdf(1.0, 1.0), np.float64)
assert_type(normal_inverse_gamma.pdf(_f64_nd, 1.0), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(normal_inverse_gamma.pdf(1.0, _f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(normal_inverse_gamma.pdf(1.0, 1.0, _f64_nd), np.float64 | onp.ArrayND[np.float64])
assert_type(normal_inverse_gamma.pdf(1.0, 1.0, lmbda=_f64_nd), np.float64 | onp.ArrayND[np.float64])
assert_type(normal_inverse_gamma.pdf(1.0, 1.0, a=_f64_nd), np.float64 | onp.ArrayND[np.float64])
assert_type(normal_inverse_gamma.pdf(1.0, 1.0, b=_f64_nd), np.float64 | onp.ArrayND[np.float64])
assert_type(normal_inverse_gamma.pdf(_f_1d, 1.0), onp.Array1D[np.float64])
assert_type(normal_inverse_gamma.pdf(1.0, _f_1d), onp.Array1D[np.float64])
assert_type(normal_inverse_gamma.pdf(_f_2d, 1.0), onp.Array2D[np.float64])
assert_type(normal_inverse_gamma.pdf(1.0, _f_2d), onp.Array2D[np.float64])
assert_type(normal_inverse_gamma.pdf(_f_3d, 1.0), onp.Array3D[np.float64])
assert_type(normal_inverse_gamma.pdf(1.0, _f_3d), onp.Array3D[np.float64])

assert_type(normal_inverse_gamma(_f64_nd).logpdf(1.0, 1.0), np.float64 | onp.ArrayND[np.float64])
assert_type(normal_inverse_gamma().logpdf(_f64_nd, 1.0), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(normal_inverse_gamma().logpdf(1.0, _f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(normal_inverse_gamma().logpdf(1.0, 1.0), np.float64)
assert_type(normal_inverse_gamma(_f_1d).logpdf(1.0, 1.0), onp.Array1D[np.float64])
assert_type(normal_inverse_gamma().logpdf(_f_1d, 1.0), onp.Array1D[np.float64])
assert_type(normal_inverse_gamma().logpdf(1.0, _f_1d), onp.Array1D[np.float64])
assert_type(normal_inverse_gamma(_f_2d).logpdf(1.0, 1.0), onp.Array2D[np.float64])
assert_type(normal_inverse_gamma(_f_1d).logpdf(_f_2d, 1.0), onp.Array2D[np.float64])
assert_type(normal_inverse_gamma(_f_1d).logpdf(1.0, _f_2d), onp.Array2D[np.float64])
assert_type(normal_inverse_gamma(_f_3d).logpdf(1.0, 1.0), onp.Array3D[np.float64])
assert_type(normal_inverse_gamma(_f_2d).logpdf(_f_3d, 1.0), onp.Array3D[np.float64])
assert_type(normal_inverse_gamma(_f_2d).logpdf(1.0, _f_3d), onp.Array3D[np.float64])

assert_type(normal_inverse_gamma(_f64_nd).pdf(1.0, 1.0), np.float64 | onp.ArrayND[np.float64])
assert_type(normal_inverse_gamma().pdf(_f64_nd, 1.0), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(normal_inverse_gamma().pdf(1.0, _f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(normal_inverse_gamma().pdf(1.0, 1.0), np.float64)
assert_type(normal_inverse_gamma(_f_1d).pdf(1.0, 1.0), onp.Array1D[np.float64])
assert_type(normal_inverse_gamma().pdf(_f_1d, 1.0), onp.Array1D[np.float64])
assert_type(normal_inverse_gamma().pdf(1.0, _f_1d), onp.Array1D[np.float64])
assert_type(normal_inverse_gamma(_f_2d).pdf(1.0, 1.0), onp.Array2D[np.float64])
assert_type(normal_inverse_gamma(_f_1d).pdf(_f_2d, 1.0), onp.Array2D[np.float64])
assert_type(normal_inverse_gamma(_f_1d).pdf(1.0, _f_2d), onp.Array2D[np.float64])
assert_type(normal_inverse_gamma(_f_3d).pdf(1.0, 1.0), onp.Array3D[np.float64])
assert_type(normal_inverse_gamma(_f_2d).pdf(_f_3d, 1.0), onp.Array3D[np.float64])
assert_type(normal_inverse_gamma(_f_2d).pdf(1.0, _f_3d), onp.Array3D[np.float64])

assert_type(normal_inverse_gamma.rvs(), tuple[np.float64, np.float64])
assert_type(normal_inverse_gamma().rvs(), tuple[np.float64, np.float64])
assert_type(
    normal_inverse_gamma(_f64_nd).rvs(),
    tuple[np.float64, np.float64]
    | tuple[onp.ArrayND[np.float64], np.float64]
    | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]],
)
assert_type(
    normal_inverse_gamma(_f_1d).rvs(),
    tuple[onp.Array1D[np.float64], np.float64] | tuple[onp.Array1D[np.float64], onp.ArrayND[np.float64]],
)
assert_type(
    normal_inverse_gamma(_f_2d).rvs(),
    tuple[onp.Array2D[np.float64], np.float64] | tuple[onp.Array2D[np.float64], onp.ArrayND[np.float64]],
)
assert_type(
    normal_inverse_gamma(_f_3d).rvs(),
    tuple[onp.Array3D[np.float64], np.float64] | tuple[onp.Array3D[np.float64], onp.ArrayND[np.float64]],
)
assert_type(
    normal_inverse_gamma.rvs(size=_shape_nd), tuple[np.float64 | onp.ArrayND[np.float64], np.float64 | onp.ArrayND[np.float64]]
)
assert_type(
    normal_inverse_gamma().rvs(size=_shape_nd), tuple[np.float64 | onp.ArrayND[np.float64], np.float64 | onp.ArrayND[np.float64]]
)
assert_type(normal_inverse_gamma.rvs(size=()), tuple[np.float64, np.float64])
assert_type(normal_inverse_gamma().rvs(size=()), tuple[np.float64, np.float64])
assert_type(normal_inverse_gamma.rvs(size=3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(normal_inverse_gamma().rvs(size=3), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(normal_inverse_gamma.rvs(size=(2, 3)), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(normal_inverse_gamma().rvs(size=(2, 3)), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(normal_inverse_gamma.rvs(size=(2, 3, 4)), tuple[onp.Array3D[np.float64], onp.Array3D[np.float64]])
assert_type(normal_inverse_gamma().rvs(size=(2, 3, 4)), tuple[onp.Array3D[np.float64], onp.Array3D[np.float64]])
assert_type(normal_inverse_gamma.rvs(_f_1d), tuple[np.float64 | onp.ArrayND[np.float64], np.float64 | onp.ArrayND[np.float64]])
assert_type(
    normal_inverse_gamma().rvs(size=_i64), tuple[np.float64 | onp.ArrayND[np.float64], np.float64 | onp.ArrayND[np.float64]]
)
