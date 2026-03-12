import numpy as np
import optype.numpy as onp

__all__ = ["geometric_slerp"]

def geometric_slerp(
    start: onp.ToFloat1D, end: onp.ToFloat1D, t: onp.ToFloat | onp.ToFloat1D, tol: onp.ToFloat = 1e-07
) -> onp.Array2D[np.float64]: ...
