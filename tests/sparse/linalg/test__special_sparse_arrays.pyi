from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse.linalg import LaplacianNd

gs: tuple[int, int]

lap_i8 = LaplacianNd(gs)
assert_type(lap_i8, LaplacianNd[np.int8])

lap_f64 = LaplacianNd(gs, dtype=np.dtype(np.float64))
assert_type(lap_f64, LaplacianNd[np.float64])

lap_any = LaplacianNd(gs, dtype="float64")
assert_type(lap_any, LaplacianNd[Any])

# methods
lap: LaplacianNd[np.float64]
assert_type(lap.eigenvalues(), onp.Array1D[np.float64])
assert_type(lap.eigenvectors(), onp.Array2D[np.float64])
assert_type(lap.toarray(), onp.Array2D[np.float64])
