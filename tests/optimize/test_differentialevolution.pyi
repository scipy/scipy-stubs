from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import Bounds, NonlinearConstraint, differential_evolution
from scipy.optimize._differentialevolution import OptimizeResult

###

type _ResultUnconstrained = OptimizeResult[onp.Array1D[np.float64] | None]
type _ResultConstrained = OptimizeResult[list[onp.Array2D[np.float64]] | None]

def _obj(x: onp.Array1D[np.float64]) -> float: ...

_b: list[tuple[float, float]]
_nlc: NonlinearConstraint

###

assert_type(differential_evolution(_obj, bounds=([-5.0], [5.0])), _ResultUnconstrained)
assert_type(differential_evolution(_obj, bounds=[(-5.0, 5.0), (-2.0, 2.0)]), _ResultUnconstrained)
assert_type(differential_evolution(_obj, bounds=[[-5.0, 5.0], [-2.0, 2.0]]), _ResultUnconstrained)
assert_type(differential_evolution(_obj, _b, constraints=_nlc), _ResultConstrained)
assert_type(differential_evolution(_obj, _b, constraints=Bounds(0, 1)), _ResultConstrained)

_res: OptimizeResult

assert_type(_res.x, onp.Array1D[np.float64])
assert_type(_res.fun, float | np.float64)
assert_type(_res.population, onp.Array2D[np.float64])
assert_type(_res.jac, onp.Array1D[np.float64] | list[onp.Array2D[np.float64]] | None)
assert_type(differential_evolution(_obj, _b).jac, onp.Array1D[np.float64] | None)
assert_type(differential_evolution(_obj, _b, constraints=_nlc).jac, list[onp.Array2D[np.float64]] | None)
