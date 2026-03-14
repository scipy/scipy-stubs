# type-tests for `spatial/distance.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.spatial.distance import (
    braycurtis,
    canberra,
    cdist,
    chebyshev,
    cityblock,
    correlation,
    cosine,
    dice,
    directed_hausdorff,
    euclidean,
    hamming,
    is_valid_dm,
    is_valid_y,
    jaccard,
    jensenshannon,
    mahalanobis,
    minkowski,
    num_obs_dm,
    num_obs_y,
    pdist,
    rogerstanimoto,
    russellrao,
    seuclidean,
    sokalsneath,
    sqeuclidean,
    squareform,
    yule,
)

###

_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_c128_1d: onp.Array1D[np.complex128]

_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_c128_2d: onp.Array2D[np.complex128]

_f64_nd: onp.ArrayND[np.float64]

_py_i_1d: list[int]
_py_f_1d: list[float]
_py_c_1d: list[complex]

_py_i_2d: list[list[int]]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]

###

# cdist
assert_type(cdist(_f64_2d, _f64_2d), onp.Array2D[np.float64])
assert_type(cdist(_f64_2d, _f64_2d, "euclidean"), onp.Array2D[np.float64])
assert_type(cdist(_f64_2d, _f64_2d, "cosine"), onp.Array2D[np.float64])
assert_type(cdist(_py_f_2d, _py_f_2d), onp.Array2D[np.float64])
assert_type(cdist(_f64_2d, _f64_2d, metric=lambda u, v: 0.0), onp.Array2D[np.float64])

# pdist
assert_type(pdist(_f64_2d), onp.Array1D[np.float64])
assert_type(pdist(_f64_2d, "euclidean"), onp.Array1D[np.float64])
assert_type(pdist(_f64_2d, "cosine"), onp.Array1D[np.float64])
assert_type(pdist(_py_f_2d), onp.Array1D[np.float64])
assert_type(pdist(_f64_2d, metric=lambda u, v: 0.0), onp.Array1D[np.float64])

# squareform
assert_type(squareform(_py_i_1d), onp.Array2D[np.intp])
assert_type(squareform(_py_f_1d), onp.Array2D[np.float64])
assert_type(squareform(_py_c_1d), onp.Array2D[np.complex128])
assert_type(squareform(_f32_1d), onp.Array2D[np.float32])
assert_type(squareform(_py_i_2d), onp.Array1D[np.intp])
assert_type(squareform(_py_f_2d), onp.Array1D[np.float64])
assert_type(squareform(_py_c_2d), onp.Array1D[np.complex128])
assert_type(squareform(_f32_2d), onp.Array1D[np.float32])

# correlation
assert_type(correlation(_f64_1d, _f64_1d), np.float64)
assert_type(correlation(_f64_1d, _f64_1d, w=_f64_1d), np.float64)

# cosine
assert_type(cosine(_f64_1d, _f64_1d), np.float64)
assert_type(cosine(_f64_1d, _f64_1d, w=_f64_1d), np.float64)

# hamming
assert_type(hamming(_f64_1d, _f64_1d), np.float64)
assert_type(hamming(_f64_1d, _f64_1d, w=_f64_1d), np.float64)

# jaccard
assert_type(jaccard(_f64_1d, _f64_1d), np.float64)

# braycurtis
assert_type(braycurtis(_f64_1d, _f64_1d), np.float64)

# canberra
assert_type(canberra(_f64_1d, _f64_1d), np.float64)

# chebyshev
assert_type(chebyshev(_f64_1d, _f64_1d), np.float64)

# cityblock
assert_type(cityblock(_f64_1d, _f64_1d), np.float64)

# dice
assert_type(dice(_f64_1d, _f64_1d), float)

# euclidean
assert_type(euclidean(_f64_1d, _f64_1d), float)

# minkowski
assert_type(minkowski(_f64_1d, _f64_1d), float)

# rogerstanimoto
assert_type(rogerstanimoto(_f64_1d, _f64_1d), float)

# russellrao
assert_type(russellrao(_f64_1d, _f64_1d), float)

# seuclidean
assert_type(seuclidean(_f64_1d, _f64_1d, V=_f64_1d), float)

# yule
assert_type(yule(_f64_1d, _f64_1d), float)

# directed_hausdorff
assert_type(directed_hausdorff(_f64_2d, _f64_2d), tuple[float, int, int])

# mahalanobis
assert_type(mahalanobis(_f64_1d, _f64_1d, VI=_f64_2d), np.float64)

# sokalsneath
assert_type(sokalsneath(_f64_1d, _f64_1d), np.float64)
assert_type(sokalsneath(_f64_1d, _c128_1d), np.complex128)
assert_type(sokalsneath(_c128_1d, _f64_1d), np.complex128)
assert_type(sokalsneath(_c128_1d, _c128_1d), np.complex128)

# sqeuclidean
assert_type(sqeuclidean(_f64_1d, _f64_1d), np.float64)
assert_type(sqeuclidean(_f64_1d, _c128_1d), np.complex128)
assert_type(sqeuclidean(_c128_1d, _f64_1d), np.complex128)
assert_type(sqeuclidean(_c128_1d, _c128_1d), np.complex128)

# jensenshannon
assert_type(jensenshannon(_py_f_1d, _py_f_1d), np.float64)
assert_type(jensenshannon(_py_f_2d, _py_f_2d), onp.Array1D[np.float64])
assert_type(jensenshannon(_f64_1d, _f64_1d), np.float64)
assert_type(jensenshannon(_f64_2d, _f64_2d), onp.Array1D[np.float64])
assert_type(jensenshannon(_f64_nd, _f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(jensenshannon(_f64_1d, _f64_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(jensenshannon(_f64_2d, _f64_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(jensenshannon(_f64_nd, _f64_nd, keepdims=True), onp.ArrayND[np.float64])

# num_obs_dm
assert_type(num_obs_dm(_f64_2d), int)

# num_obs_y
assert_type(num_obs_y(_f64_1d), int)

# is_valid_dm
assert_type(is_valid_dm(_f64_2d), bool)

# is_valid_y
assert_type(is_valid_y(_f64_1d), bool)
