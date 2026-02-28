# type-tests for `spatial/distance.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc
from optype.test import assert_subtype

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

f32_1d: onp.Array1D[np.float32]
f32_2d: onp.Array2D[np.float32]

f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]

c128_1d: onp.Array1D[np.complex128]
c128_2d: onp.Array2D[np.complex128]

out_f32_1d: onp.Array1D[np.float32]
out_f64_1d: onp.Array1D[np.float64]

py_i_1d: list[int]
py_f_1d: list[float]
py_c_1d: list[complex]

py_i_2d: list[list[int]]
py_f_2d: list[list[float]]
py_c_2d: list[list[complex]]

###
# cdist

assert_type(cdist(f64_2d, f64_2d), onp.ArrayND[npc.floating])
assert_type(cdist(f64_2d, f64_2d, "euclidean"), onp.ArrayND[npc.floating])
assert_type(cdist(f64_2d, f64_2d, "cosine"), onp.ArrayND[npc.floating])
assert_type(cdist(py_f_2d, py_f_2d), onp.ArrayND[npc.floating])

assert_type(cdist(c128_2d, c128_2d), onp.ArrayND[npc.inexact])
assert_type(cdist(c128_2d, c128_2d, "euclidean"), onp.ArrayND[npc.inexact])

assert_type(cdist(f64_2d, f64_2d, out=out_f64_1d), onp.Array1D[np.float64])
assert_type(cdist(f64_2d, f64_2d, out=out_f32_1d), onp.Array1D[np.float32])
assert_type(cdist(c128_2d, c128_2d, out=out_f64_1d), onp.Array1D[np.float64])

assert_type(cdist(f64_2d, f64_2d, metric=lambda u, v: 0.0), onp.ArrayND[npc.floating])  # pyrefly:ignore[assert-type]
assert_type(cdist(c128_2d, c128_2d, metric=lambda u, v: 0.0), onp.ArrayND[npc.inexact])
assert_type(cdist(c128_2d, c128_2d, metric=lambda u, v: 0.0, out=out_f64_1d), onp.Array1D[np.float64])

###
# pdist

assert_type(pdist(f64_2d), onp.ArrayND[npc.floating])
assert_type(pdist(f64_2d, "euclidean"), onp.ArrayND[npc.floating])
assert_type(pdist(f64_2d, "cosine"), onp.ArrayND[npc.floating])
assert_type(pdist(py_f_2d), onp.ArrayND[npc.floating])

assert_type(pdist(c128_2d), onp.ArrayND[npc.inexact])
assert_type(pdist(c128_2d, "euclidean"), onp.ArrayND[npc.inexact])

assert_type(pdist(f64_2d, out=out_f64_1d), onp.Array1D[np.float64])
assert_type(pdist(f64_2d, out=out_f32_1d), onp.Array1D[np.float32])
assert_type(pdist(c128_2d, out=out_f64_1d), onp.Array1D[np.float64])

assert_type(pdist(f64_2d, metric=lambda u, v: 0.0), onp.ArrayND[npc.floating])  # pyrefly:ignore[assert-type]
assert_type(pdist(c128_2d, metric=lambda u, v: 0.0), onp.ArrayND[npc.inexact])
assert_type(pdist(c128_2d, metric=lambda u, v: 0.0, out=out_f64_1d), onp.Array1D[np.float64])

###
# squareform

# 1-d strict int -> 2-d int
assert_type(squareform(py_i_1d), onp.Array2D[np.intp])

# 1-d strict float -> 2-d float64
assert_type(squareform(py_f_1d), onp.Array2D[np.float64])

# 1-d strict complex -> 2-d complex128
assert_type(squareform(py_c_1d), onp.Array2D[np.complex128])

# 1-d typed float32 array -> matches float overload -> 2-d float64
assert_type(squareform(f32_1d), onp.Array2D[np.float64])

# 2-d strict int -> 1-d int
assert_type(squareform(py_i_2d), onp.Array1D[np.intp])

# 2-d strict float -> 1-d float64
assert_type(squareform(py_f_2d), onp.Array1D[np.float64])

# 2-d strict complex -> 1-d complex128
assert_type(squareform(py_c_2d), onp.Array1D[np.complex128])

# 2-d typed float32 array -> matches float overload -> 1-d float64
assert_type(squareform(f32_2d), onp.Array1D[np.float64])

###
# scalar distance functions

# correlation / cosine return exact np.float64
assert_type(correlation(f64_1d, f64_1d), np.float64)
assert_type(correlation(f64_1d, f64_1d, w=f64_1d), np.float64)
assert_type(cosine(f64_1d, f64_1d), np.float64)
assert_type(cosine(f64_1d, f64_1d, w=f64_1d), np.float64)

# hamming / jaccard return np.float64
assert_type(hamming(f64_1d, f64_1d), np.float64)
assert_type(hamming(f64_1d, f64_1d, w=f64_1d), np.float64)
assert_type(jaccard(f64_1d, f64_1d), np.float64)

# braycurtis / canberra return np.float64
assert_type(braycurtis(f64_1d, f64_1d), np.float64)
assert_type(canberra(f64_1d, f64_1d), np.float64)

# chebyshev / cityblock return a floating compat type
assert_subtype[npc.floating](chebyshev(f64_1d, f64_1d))
assert_subtype[npc.floating](cityblock(f64_1d, f64_1d))

# pure-Python builtins float
assert_type(dice(f64_1d, f64_1d), float)
assert_type(euclidean(f64_1d, f64_1d), float)
assert_type(minkowski(f64_1d, f64_1d), float)
assert_type(rogerstanimoto(f64_1d, f64_1d), float)
assert_type(russellrao(f64_1d, f64_1d), float)
assert_type(seuclidean(f64_1d, f64_1d, V=f64_1d), float)
assert_type(yule(f64_1d, f64_1d), float)

# directed_hausdorff
assert_type(directed_hausdorff(f64_2d, f64_2d), tuple[float, int, int])

###
# overloaded scalar functions

# mahalanobis: float input -> np.float64
assert_type(mahalanobis(f64_1d, f64_1d, VI=f64_2d), np.float64)

# sokalsneath: float input -> np.float64; complex input -> np.float64 | np.complex128
assert_type(sokalsneath(f64_1d, f64_1d), np.float64)
assert_type(sokalsneath(c128_1d, c128_1d), np.float64 | np.complex128)

# sqeuclidean: float input -> npc.floating; complex input -> npc.inexact
assert_subtype[npc.floating](sqeuclidean(f64_1d, f64_1d))
assert_subtype[npc.inexact](sqeuclidean(c128_1d, c128_1d))

###
# jensenshannon

# 1-d strict input, keepdims=False (default) -> scalar
assert_type(jensenshannon(py_f_1d, py_f_1d), np.float32 | np.float64)
assert_type(jensenshannon(f64_1d, f64_1d), np.float32 | np.float64)

# 1-d strict input, keepdims=True -> 1-d array
assert_type(jensenshannon(f64_1d, f64_1d, keepdims=True), onp.Array1D[np.float32 | np.float64])

# N-d input -> scalar or array
assert_subtype[np.float32 | np.float64 | onp.ArrayND[np.float32 | np.float64]](jensenshannon(f64_2d, f64_2d))

###
# utility functions

assert_type(num_obs_dm(f64_2d), int)
assert_type(num_obs_y(f64_1d), int)
assert_type(is_valid_dm(f64_2d), bool)
assert_type(is_valid_y(f64_1d), bool)
