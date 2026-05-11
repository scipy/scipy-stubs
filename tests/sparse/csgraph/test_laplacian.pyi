from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.sparse.csgraph import laplacian

_f64_nd: npt.NDArray[np.float64]

fn = laplacian(_f64_nd, form="function")
op.test.assert_subtype[Callable[[onp.ToComplex2D], onp.Array2D[npc.number]]](fn)

fn2, diag = laplacian(_f64_nd, form="function", return_diag=True)
op.test.assert_subtype[Callable[[onp.ToComplex2D], onp.Array2D[npc.number]]](fn2)
op.test.assert_subtype[onp.Array1D[npc.number]](diag)
