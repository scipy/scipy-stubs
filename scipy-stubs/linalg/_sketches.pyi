from typing import overload

import numpy as np
import optype.numpy as onp

from scipy.sparse import csc_matrix

__all__ = ["clarkson_woodruff_transform"]

###

def cwt_matrix(n_rows: onp.ToInt, n_columns: onp.ToInt, rng: onp.random.ToRNG | None = None) -> csc_matrix[np.int_]: ...

#
@overload
def clarkson_woodruff_transform(
    input_matrix: onp.ToIntND,
    sketch_size: onp.ToInt,
    rng: onp.random.ToRNG | None = None,
    *,
    seed: onp.random.ToRNG | None = None,
) -> onp.ArrayND[np.int_]: ...
@overload
def clarkson_woodruff_transform(
    input_matrix: onp.ToJustFloat64_ND,
    sketch_size: onp.ToInt,
    rng: onp.random.ToRNG | None = None,
    *,
    seed: onp.random.ToRNG | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload
def clarkson_woodruff_transform(
    input_matrix: onp.ToJustFloatND,
    sketch_size: onp.ToInt,
    rng: onp.random.ToRNG | None = None,
    *,
    seed: onp.random.ToRNG | None = None,
) -> onp.ArrayND[np.float64 | np.longdouble]: ...
@overload
def clarkson_woodruff_transform(
    input_matrix: onp.ToJustComplex128_ND,
    sketch_size: onp.ToInt,
    rng: onp.random.ToRNG | None = None,
    *,
    seed: onp.random.ToRNG | None = None,
) -> onp.ArrayND[np.complex128]: ...
@overload
def clarkson_woodruff_transform(
    input_matrix: onp.ToJustComplexND,
    sketch_size: onp.ToInt,
    rng: onp.random.ToRNG | None = None,
    *,
    seed: onp.random.ToRNG | None = None,
) -> onp.ArrayND[np.complex128 | np.clongdouble]: ...
