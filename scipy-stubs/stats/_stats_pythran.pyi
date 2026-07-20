from typing import Final, overload

import numpy as np
import optype.numpy as onp

###

# undocumented
__pythran__: Final[tuple[str, str]] = ...

# undocumented
@overload
def _Aij(A: onp.Array2D[np.int_], i: int, j: int) -> int: ...
@overload
def _Aij(A: onp.Array2D[np.float64], i: int, j: int) -> float: ...

# undocumented
@overload
def _Dij(A: onp.Array2D[np.int_], i: int, j: int) -> int: ...
@overload
def _Dij(A: onp.Array2D[np.float64], i: int, j: int) -> float: ...

# undocumented
@overload
def _concordant_pairs(A: onp.Array2D[np.int_]) -> int: ...
@overload
def _concordant_pairs(A: onp.Array2D[np.float64]) -> float: ...

# undocumented
@overload
def _discordant_pairs(A: onp.Array2D[np.int_]) -> int: ...
@overload
def _discordant_pairs(A: onp.Array2D[np.float64]) -> float: ...

# undocumented
@overload
def _a_ij_Aij_Dij2(A: onp.Array2D[np.int_]) -> int: ...
@overload
def _a_ij_Aij_Dij2(A: onp.Array2D[np.float64]) -> float: ...

# undocumented
def _compute_outer_prob_inside_method(m: int, n: int, g: int, h: int) -> float: ...

# undocumented
@overload
def siegelslopes(y: onp.Array1D[np.float64], x: onp.Array1D[np.float64], method: str) -> tuple[float, float]: ...
@overload
def siegelslopes(y: onp.Array1D[np.float32], x: onp.Array1D[np.float32], method: str) -> tuple[float, float]: ...
