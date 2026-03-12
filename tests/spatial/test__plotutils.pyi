# type-tests for `spatial/_plotutils.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scipy.spatial import ConvexHull, Delaunay, Voronoi, convex_hull_plot_2d, delaunay_plot_2d, voronoi_plot_2d

###

_f64_2d: onp.Array2D[np.float64]
_ax: Axes

###
# delaunay_plot_2d

_tri: Delaunay
assert_type(delaunay_plot_2d(_tri), Figure)
assert_type(delaunay_plot_2d(_tri, ax=_ax), Figure)

###
# convex_hull_plot_2d

_hull: ConvexHull
assert_type(convex_hull_plot_2d(_hull), Figure)
assert_type(convex_hull_plot_2d(_hull, ax=_ax), Figure)

###
# voronoi_plot_2d

_vor: Voronoi
assert_type(voronoi_plot_2d(_vor), Figure)
assert_type(voronoi_plot_2d(_vor, ax=_ax), Figure)
assert_type(voronoi_plot_2d(_vor, show_points=True, show_vertices=False), Figure)
