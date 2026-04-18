from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import (
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

_f_1d: list[float] | onp.Array1D[np.float64]
_f_2d: list[list[float]] | onp.Array2D[np.float64]
_f_3d: list[list[list[float]]] | onp.Array3D[np.float64]
_f_nd: onp.ArrayND[np.float64]

###

# multivariate_normal

assert_type(multivariate_normal.rvs().dtype, np.dtype[np.float64])
assert_type(multivariate_normal().rvs().dtype, np.dtype[np.float64])

# matrix_normal

assert_type(matrix_normal.rvs().dtype, np.dtype[np.float64])
assert_type(matrix_normal().rvs().dtype, np.dtype[np.float64])

# dirichlet

assert_type(dirichlet.rvs([1, 2]).dtype, np.dtype[np.float64])
assert_type(dirichlet([1, 2]).rvs().dtype, np.dtype[np.float64])

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

assert_type(multinomial.rvs(1, _f_1d), onp.Array1D[np.float64])  # `multinomial_gen.rvs` defaults to `size=None`
assert_type(multinomial.rvs(1, _f_1d, size=()), onp.Array1D[np.float64])
assert_type(multinomial.rvs(1, _f_1d, size=1), onp.Array2D[np.float64])
assert_type(multinomial.rvs(1, _f_1d, size=(1,)), onp.Array2D[np.float64])
assert_type(multinomial.rvs(1, _f_1d, size=(1, 2)), onp.Array3D[np.float64])
assert_type(multinomial.rvs(1, _f_1d, size=(1, 2, 4, 8, 16, 31)), onp.ArrayND[np.float64])
assert_type(multinomial(1, _f_1d).rvs(), onp.Array2D[np.float64])  # `multinomial_frozen.rvs` defaults to `size=1`
assert_type(multinomial(1, _f_1d).rvs(size=()), onp.Array1D[np.float64])
assert_type(multinomial(1, _f_1d).rvs(size=1), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_1d).rvs(size=(1,)), onp.Array2D[np.float64])
assert_type(multinomial(1, _f_1d).rvs(size=(1, 2)), onp.Array3D[np.float64])
assert_type(multinomial(1, _f_1d).rvs(size=(1, 2, 4, 8, 16, 31)), onp.ArrayND[np.float64])

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

# random_correlation

assert_type(random_correlation.rvs([1, 1]).dtype, np.dtype[np.float64])
assert_type(random_correlation([1, 1]).rvs().dtype, np.dtype[np.float64])

# multivariate_t

assert_type(multivariate_t.rvs().dtype, np.dtype[np.float64])
assert_type(multivariate_t().rvs().dtype, np.dtype[np.float64])

# multivariate_hypergeom

assert_type(multivariate_hypergeom.rvs([1], 1).dtype, np.dtype[np.float64])
assert_type(multivariate_hypergeom([1], 1).rvs().dtype, np.dtype[np.float64])

# random_table

assert_type(random_table.rvs([1, 2], [2, 1]).dtype, np.dtype[np.float64])
assert_type(random_table([1, 2], [2, 1]).rvs().dtype, np.dtype[np.float64])

# dirichlet_multinomial

assert_type(dirichlet_multinomial.pmf([1, 2], [0.5, 0.5], [3]), np.float64 | onp.Array[tuple[int, *tuple[Any, ...]], np.float64])
assert_type(dirichlet_multinomial([0.5, 0.5], [3]).pmf([1, 2]), np.float64 | onp.Array[tuple[int, *tuple[Any, ...]], np.float64])

# matrix_t

assert_type(matrix_t.rvs(df=1).dtype, np.dtype[np.float64])
assert_type(matrix_t(df=1).rvs().dtype, np.dtype[np.float64])

# vonmises_fisher

assert_type(vonmises_fisher.rvs([0.8, 0.6]).dtype, np.dtype[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).rvs().dtype, np.dtype[np.float64])

# normal_inverse_gamma

assert_type(normal_inverse_gamma.rvs()[0].dtype, np.dtype[np.float64])
assert_type(normal_inverse_gamma().rvs()[0].dtype, np.dtype[np.float64])
