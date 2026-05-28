from collections.abc import Mapping
from typing import Literal

import numpy as np
import optype.numpy as onp

from scipy.sparse._base import _spbase
from scipy.sparse.linalg import LinearOperator

__all__ = ["svds"]

###

type _Inexact = np.float32 | np.float64 | np.complex64 | np.complex128
type _ToMatrix[ScalarT: _Inexact] = onp.ArrayND[ScalarT] | LinearOperator[ScalarT] | _spbase

type _Which = Literal["LM", "SM"]
type _ReturnSingularVectors = Literal["u", "v"] | bool
type _Solver = Literal["arpack", "propack", "lobpcg"]

###

def svds[ScalarT: _Inexact](
    A: _ToMatrix[ScalarT],
    k: int = 6,
    ncv: int | None = None,
    tol: float = 0,
    which: _Which = "LM",
    v0: onp.ArrayND[ScalarT] | None = None,
    maxiter: int | None = None,
    return_singular_vectors: _ReturnSingularVectors = True,
    solver: _Solver = "arpack",
    rng: onp.random.ToRNG | None = None,
    options: Mapping[str, object] | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> tuple[onp.Array2D[ScalarT], onp.ArrayND[np.float32 | np.float64], onp.ArrayND[ScalarT]]: ...
