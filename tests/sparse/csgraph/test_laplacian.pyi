# type-tests for `sparse/csgraph/_laplacian.pyi`

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg._interface import LinearOperator

###

_f64_nd: npt.NDArray[np.float64]

###

# laplacian

_fn = laplacian(_f64_nd, form="function")
op.test.assert_subtype[Callable[[onp.ToComplex2D], onp.Array2D[npc.number]]](_fn)

_fn1 = laplacian(_f64_nd, form="lo")
op.test.assert_subtype[LinearOperator[npc.number, tuple[int, int]]](_fn1)

_fn2, _diag = laplacian(_f64_nd, form="function", return_diag=True)
op.test.assert_subtype[Callable[[onp.ToComplex2D], onp.Array2D[npc.number]]](_fn2)
op.test.assert_subtype[onp.Array1D[npc.number]](_diag)
