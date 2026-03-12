# type-tests for `spatial/_qhull.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.spatial import ConvexHull, Delaunay, HalfspaceIntersection, Voronoi, tsearch

###

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

###
# Delaunay

_tri: Delaunay

assert_type(_tri.furthest_site, bool)
assert_type(_tri.paraboloid_scale, float)
assert_type(_tri.paraboloid_shift, float)
assert_type(_tri.simplices, onp.Array2D[np.int32])
assert_type(_tri.neighbors, onp.Array2D[np.int32])
assert_type(_tri.equations, onp.Array2D[np.float64])
assert_type(_tri.coplanar, onp.Array2D[np.int32])
assert_type(_tri.good, onp.Array1D[np.int32])
assert_type(_tri.nsimplex, int)
assert_type(_tri.vertices, onp.Array2D[np.float64])
assert_type(_tri.ndim, int)
assert_type(_tri.npoints, int)
assert_type(_tri.min_bound, onp.Array1D[np.float64])
assert_type(_tri.max_bound, onp.Array1D[np.float64])
assert_type(_tri.points, onp.Array2D[np.float64])
assert_type(_tri.transform, onp.Array3D[np.float64])
assert_type(_tri.vertex_to_simplex, onp.Array1D[np.int32])
assert_type(_tri.vertex_neighbor_vertices, tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])
assert_type(_tri.convex_hull, onp.Array2D[np.int32])

assert_type(_tri.find_simplex(_f64_1d), onp.Array0D[np.int32])
assert_type(_tri.find_simplex(_f64_2d), onp.Array1D[np.int32])
assert_type(_tri.find_simplex(_f64_3d), onp.Array2D[np.int32])
assert_type(_tri.find_simplex(_f64_nd), onp.ArrayND[np.int32])

assert_type(_tri.plane_distance(_f64_1d), onp.Array1D[np.float64])
assert_type(_tri.plane_distance(_f64_2d), onp.Array2D[np.float64])
assert_type(_tri.plane_distance(_f64_3d), onp.Array3D[np.float64])
assert_type(_tri.plane_distance(_f64_nd), onp.ArrayND[np.float64])

assert_type(_tri.lift_points(_f64_1d), onp.Array1D[np.float64])
assert_type(_tri.lift_points(_f64_2d), onp.Array2D[np.float64])
assert_type(_tri.lift_points(_f64_3d), onp.Array3D[np.float64])
assert_type(_tri.lift_points(_f64_nd), onp.ArrayND[np.float64])

###
# ConvexHull

_hull: ConvexHull

assert_type(_hull.simplices, onp.Array2D[np.int32])
assert_type(_hull.neighbors, onp.Array2D[np.int32])
assert_type(_hull.equations, onp.Array2D[np.float64])
assert_type(_hull.coplanar, onp.Array2D[np.int32])
assert_type(_hull.good, onp.Array1D[np.bool_] | None)
assert_type(_hull.min_bound, onp.Array1D[np.float64])
assert_type(_hull.max_bound, onp.Array1D[np.float64])
assert_type(_hull.points, onp.Array2D[np.float64])
assert_type(_hull.vertices, onp.Array2D[np.int32])
assert_type(_hull.volume, float)
assert_type(_hull.area, float)
assert_type(_hull.nsimplex, int)
assert_type(_hull.ndim, int)
assert_type(_hull.npoints, int)

###
# Voronoi

_vor: Voronoi

assert_type(_vor.vertices, onp.Array2D[np.float64])
assert_type(_vor.ridge_points, onp.Array2D[np.int32])
assert_type(_vor.ridge_vertices, list[list[int]])
assert_type(_vor.regions, list[list[int]])
assert_type(_vor.point_region, onp.Array1D[np.intp])
assert_type(_vor.furthest_site, bool)
assert_type(_vor.points, onp.Array2D[np.float64])
assert_type(_vor.ridge_dict, dict[tuple[int, int], list[int]])

###
# HalfspaceIntersection

_hsi: HalfspaceIntersection

assert_type(_hsi.interior_point, onp.Array1D[np.float64])
assert_type(_hsi.intersections, onp.Array2D[np.float64])
assert_type(_hsi.dual_facets, list[list[int]])
assert_type(_hsi.dual_equations, onp.Array2D[np.float64])
assert_type(_hsi.dual_points, onp.Array2D[np.float64])
assert_type(_hsi.dual_volume, float)
assert_type(_hsi.dual_area, float)
assert_type(_hsi.ndim, int)
assert_type(_hsi.nineq, int)
assert_type(_hsi.halfspaces, onp.Array2D[np.float64])
assert_type(_hsi.dual_vertices, onp.Array1D[np.int32])

###
# tsearch

assert_type(tsearch(_tri, _f64_1d), onp.Array0D[np.int32])
assert_type(tsearch(_tri, _f64_2d), onp.Array1D[np.int32])
assert_type(tsearch(_tri, _f64_3d), onp.Array2D[np.int32])
assert_type(tsearch(_tri, _f64_nd), onp.ArrayND[np.int32])
