# type-tests for `spatial/_spherical_voronoi.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.spatial import SphericalVoronoi

_f64_2d: onp.Array2D[np.float64]
_sv: SphericalVoronoi

assert_type(SphericalVoronoi(_f64_2d), SphericalVoronoi)
assert_type(_sv.points, onp.Array2D[np.float64])
assert_type(_sv.center, onp.Array1D[np.float64])
assert_type(_sv.radius, float)
assert_type(_sv.sort_vertices_of_regions(), None)
assert_type(_sv.calculate_areas(), onp.Array1D[np.float64])
