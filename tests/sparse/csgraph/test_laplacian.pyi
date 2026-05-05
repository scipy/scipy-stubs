from typing import reveal_type

import numpy as np
import numpy.typing as npt

from scipy.sparse.csgraph import laplacian

G: npt.NDArray[np.float64]
v: npt.NDArray[np.float64]

fn = laplacian(G, form="function")
reveal_type(fn)  # Callable

result = fn(v)
reveal_type(result)

fn2, diag = laplacian(G, form="function", return_diag=True)
reveal_type(fn2)
reveal_type(diag)
