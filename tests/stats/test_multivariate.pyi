from typing import assert_type

import numpy as np

from scipy.stats import (
    dirichlet,
    invwishart,
    matrix_normal,
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
# rvs dtype checks

# this pyright ignore is needed because of https://github.com/microsoft/pyright/issues/11127
# pyright: reportUnknownMemberType=false

assert_type(multivariate_normal.rvs().dtype, np.dtype[np.float64])
assert_type(multivariate_normal().rvs().dtype, np.dtype[np.float64])

assert_type(matrix_normal.rvs().dtype, np.dtype[np.float64])
assert_type(matrix_normal().rvs().dtype, np.dtype[np.float64])

assert_type(dirichlet.rvs([1, 2]).dtype, np.dtype[np.float64])
assert_type(dirichlet([1, 2]).rvs().dtype, np.dtype[np.float64])

assert_type(wishart.rvs(1, 1).dtype, np.dtype[np.float64])
assert_type(wishart().rvs().dtype, np.dtype[np.float64])

assert_type(invwishart.rvs(1, 1).dtype, np.dtype[np.float64])
assert_type(invwishart().rvs().dtype, np.dtype[np.float64])

assert_type(multinomial.rvs([1], [0.5]).dtype, np.dtype[np.float64])
assert_type(multinomial([1], [0.5]).rvs().dtype, np.dtype[np.float64])

assert_type(ortho_group.rvs(3).dtype, np.dtype[np.float64])
assert_type(ortho_group().rvs(3).dtype, np.dtype[np.float64])

assert_type(special_ortho_group.rvs(3).dtype, np.dtype[np.float64])
assert_type(special_ortho_group().rvs(3).dtype, np.dtype[np.float64])

assert_type(unitary_group.rvs(3).dtype, np.dtype[np.complex128])
assert_type(unitary_group().rvs(3).dtype, np.dtype[np.complex128])

assert_type(uniform_direction.rvs(2).dtype, np.dtype[np.float64])
assert_type(uniform_direction(2).rvs().dtype, np.dtype[np.float64])

assert_type(random_correlation.rvs([1, 1]).dtype, np.dtype[np.float64])
assert_type(random_correlation([1, 1]).rvs().dtype, np.dtype[np.float64])

assert_type(multivariate_t.rvs().dtype, np.dtype[np.float64])
assert_type(multivariate_t().rvs().dtype, np.dtype[np.float64])

assert_type(multivariate_hypergeom.rvs([1], 1).dtype, np.dtype[np.float64])
assert_type(multivariate_hypergeom([1], 1).rvs().dtype, np.dtype[np.float64])

assert_type(random_table.rvs([1, 2], [2, 1]).dtype, np.dtype[np.float64])
assert_type(random_table([1, 2], [2, 1]).rvs().dtype, np.dtype[np.float64])

# `dirichlet_multinomial` has no `rvs` method

assert_type(vonmises_fisher.rvs([0.8, 0.6]).dtype, np.dtype[np.float64])
assert_type(vonmises_fisher([0.8, 0.6]).rvs().dtype, np.dtype[np.float64])

assert_type(normal_inverse_gamma.rvs()[0].dtype, np.dtype[np.float64])
assert_type(normal_inverse_gamma().rvs()[0].dtype, np.dtype[np.float64])
