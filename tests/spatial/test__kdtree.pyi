# type-tests for `spatial/_kdtree.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import coo_matrix, dok_matrix
from scipy.spatial import KDTree, Rectangle, cKDTree, distance_matrix, minkowski_distance, minkowski_distance_p

###

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_c128_1d: onp.Array1D[np.complex128]
_c128_2d: onp.Array2D[np.complex128]

###
# cKDTree

_ctree: cKDTree[None, None]

assert_type(cKDTree(_f64_2d), cKDTree[None, None])
assert_type(cKDTree(_f64_2d, boxsize=_f64_2d), cKDTree[onp.Array2D[np.float64], onp.Array1D[np.float64]])

assert_type(_ctree.data, onp.Array2D[np.float64])
assert_type(_ctree.leafsize, int)
assert_type(_ctree.m, int)
assert_type(_ctree.n, int)
assert_type(_ctree.size, int)
assert_type(_ctree.boxsize, None)

# cKDTree.query_ball_point

assert_type(_ctree.query_ball_point(_f64_1d, 1.0), list[int])
assert_type(_ctree.query_ball_point(_f64_1d, 1.0, 2.0, 0.0, None, None, True), np.intp)
assert_type(_ctree.query_ball_point(_f64_1d, 1.0, return_length=True), np.intp)
assert_type(_ctree.query_ball_point(_f64_2d, _f64_1d), onp.ArrayND[np.object_])
assert_type(_ctree.query_ball_point(_f64_2d, _f64_1d, 2.0, 0.0, None, None, True), onp.ArrayND[np.intp])
assert_type(_ctree.query_ball_point(_f64_2d, _f64_1d, return_length=True), onp.ArrayND[np.intp])
assert_type(_ctree.query_ball_point(_f64_2d, 1.0), list[int] | onp.ArrayND[np.object_])
assert_type(_ctree.query_ball_point(_f64_2d, 1.0, 2.0, 0.0, None, None, True), np.intp | onp.ArrayND[np.intp])
assert_type(_ctree.query_ball_point(_f64_2d, 1.0, return_length=True), np.intp | onp.ArrayND[np.intp])

# cKDTree.query_pairs

assert_type(_ctree.query_pairs(1.0), set[tuple[int, int]])
assert_type(_ctree.query_pairs(1.0, 2.0, 0.0, "ndarray"), onp.ArrayND[np.intp])
assert_type(_ctree.query_pairs(1.0, output_type="ndarray"), onp.ArrayND[np.intp])

# cKDTree.count_neighbors

assert_type(_ctree.count_neighbors(_ctree, 1.0), np.intp)
assert_type(_ctree.count_neighbors(_ctree, 1.0, 2.0, (_f64_1d, _f64_1d)), np.float64)
assert_type(_ctree.count_neighbors(_ctree, 1.0, weights=(_f64_1d, _f64_1d)), np.float64)
assert_type(_ctree.count_neighbors(_ctree, _f64_1d), np.intp | onp.Array1D[np.intp])
assert_type(_ctree.count_neighbors(_ctree, _f64_1d, 2.0, (_f64_1d, _f64_1d)), np.float64 | onp.Array1D[np.float64])
assert_type(_ctree.count_neighbors(_ctree, _f64_1d, weights=(_f64_1d, _f64_1d)), np.float64 | onp.Array1D[np.float64])

# cKDTree.sparse_distance_matrix

assert_type(_ctree.sparse_distance_matrix(_ctree, 1.0), dok_matrix[np.float64])
assert_type(_ctree.sparse_distance_matrix(_ctree, 1.0, output_type="coo_matrix"), coo_matrix[np.float64])
assert_type(_ctree.sparse_distance_matrix(_ctree, 1.0, output_type="dict"), dict[tuple[int, int], float])
assert_type(_ctree.sparse_distance_matrix(_ctree, 1.0, output_type="ndarray"), onp.ArrayND[np.void])

###
# Rectangle

_rect = Rectangle(maxes=[1.0, 2.0, 3.0], mins=[0.0, 0.0, 0.0])

assert_type(_rect.maxes, onp.Array1D[np.float64])
assert_type(_rect.mins, onp.Array1D[np.float64])
assert_type(_rect.volume(), np.float64)
assert_type(_rect.split(0, 0.5), tuple[Rectangle, Rectangle])
assert_type(_rect.min_distance_point(_f64_1d), onp.ArrayND[np.float64])
assert_type(_rect.max_distance_point(_f64_1d), onp.ArrayND[np.float64])
assert_type(_rect.min_distance_rectangle(_rect), onp.ArrayND[np.float64])
assert_type(_rect.max_distance_rectangle(_rect), onp.ArrayND[np.float64])

###
# KDTree

_tree: KDTree[None, None]

assert_type(_tree.data, onp.Array2D[np.float64])
assert_type(_tree.leafsize, int)
assert_type(_tree.m, int)
assert_type(_tree.n, int)
assert_type(_tree.maxes, onp.Array1D[np.float64])
assert_type(_tree.mins, onp.Array1D[np.float64])
assert_type(_tree.size, int)
assert_type(_tree.indices, onp.Array1D[np.intp])
assert_type(_tree.boxsize, None)

_tree_box: KDTree[onp.Array2D[np.float64], onp.Array1D[np.float64]]

assert_type(_tree_box.boxsize, onp.Array2D[np.float64])

# KDTree.query

assert_type(_tree.query(_f64_1d), tuple[float, np.intp] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.intp]])
assert_type(_tree.query(_f64_1d, k=3), tuple[float, np.intp] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.intp]])

# KDTree.query_ball_point

assert_type(_tree.query_ball_point(_f64_1d, r=1.0), list[int])
assert_type(_tree.query_ball_point(_f64_1d, r=1.0, return_length=True), np.intp)
assert_type(_tree.query_ball_point(_f64_2d, r=_f64_1d), onp.ArrayND[np.object_])
assert_type(_tree.query_ball_point(_f64_2d, r=_f64_1d, return_length=True), onp.ArrayND[np.intp])

# KDTree.query_ball_tree

assert_type(_tree.query_ball_tree(_tree, r=1.0), list[list[int]])

# KDTree.query_pairs

assert_type(_tree.query_pairs(1.0), set[tuple[int, int]])
assert_type(_tree.query_pairs(1.0, output_type="set"), set[tuple[int, int]])
assert_type(_tree.query_pairs(1.0, output_type="ndarray"), onp.ArrayND[np.intp])

# KDTree.count_neighbors

assert_type(_tree.count_neighbors(_tree, r=1.0), np.intp)
assert_type(_tree.count_neighbors(_tree, r=1.0, weights=(_f64_1d, _f64_1d)), np.float64)

# KDTree.sparse_distance_matrix

assert_type(_tree.sparse_distance_matrix(_tree, max_distance=1.0), dok_matrix[np.float64])
assert_type(_tree.sparse_distance_matrix(_tree, max_distance=1.0, output_type="dok_matrix"), dok_matrix[np.float64])
assert_type(_tree.sparse_distance_matrix(_tree, max_distance=1.0, output_type="coo_matrix"), coo_matrix[np.float64])
assert_type(_tree.sparse_distance_matrix(_tree, max_distance=1.0, output_type="dict"), dict[tuple[int, int], float])
assert_type(_tree.sparse_distance_matrix(_tree, max_distance=1.0, output_type="ndarray"), onp.ArrayND[np.void])

###
# minkowski_distance_p

assert_type(minkowski_distance_p(_f64_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(minkowski_distance_p(_c128_1d, _c128_1d), onp.ArrayND[np.float64 | np.complex128])

###
# minkowski_distance

assert_type(minkowski_distance(_f64_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(minkowski_distance(_c128_1d, _c128_1d), onp.ArrayND[np.float64 | np.complex128])

###
# distance_matrix

assert_type(distance_matrix(_f64_2d, _f64_2d), onp.Array2D[np.float64])
assert_type(distance_matrix(_c128_2d, _c128_2d), onp.Array2D[np.float64 | np.complex128])
