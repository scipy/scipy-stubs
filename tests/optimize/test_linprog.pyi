from collections.abc import Sequence
from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import OptimizeResult, linprog, linprog_verbose_callback
from scipy.optimize._typing import Bound, MethodLinprog
from scipy.sparse import csr_array

c: list[float]

bound: Bound
bounds: Sequence[Bound]
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_csr: csr_array[np.float64]
_method: MethodLinprog

###

res = linprog(c, bounds=bounds)
assert_type(res.fun, float | None)
assert_type(res.x, np.ndarray[tuple[int], np.dtype[np.float64]] | None)
assert_type(res.success, bool)
assert_type(res.message, str)

_1: OptimizeResult[np.float64] = linprog(c, bounds=bound)
_2: OptimizeResult[np.float64] = linprog(c, bounds=bounds)
_3: OptimizeResult[np.float64] = linprog(c, bounds=_f64_1d)
_4: OptimizeResult[np.float64] = linprog(c, A_ub=_f64_csr, b_ub=_f64_1d, bounds=_f64_2d)
_5: OptimizeResult[np.float64] = linprog(c, A_eq=_f64_csr, b_eq=_f64_1d, bounds=_f64_2d, method="highs-ds")
_6: OptimizeResult[np.float64] = linprog(c, A_ub=_f64_csr, b_ub=_f64_1d, bounds=_f64_2d, method="highs-ipm")
_7: OptimizeResult[np.float64] = linprog(c, A_eq=_f64_csr, b_eq=_f64_1d, bounds=_f64_2d, method="interior-point")  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
_8: OptimizeResult[np.float64] = linprog(c, bounds=_f64_2d, method="revised simplex")  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
_9: OptimizeResult[np.float64] = linprog(c, bounds=_f64_2d, method="simplex")  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
_10: OptimizeResult[np.float64] = linprog(c, A_ub=_f64_csr, b_ub=_f64_1d, bounds=_f64_2d, method=_method)
_11: OptimizeResult[np.float64] = linprog(c, method="highs", options={"mip_rel_gap": 1e-3})

###
# linprog_verbose_callback

assert_type(linprog_verbose_callback(res), None)
