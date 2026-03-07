from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.optimize import root

###

def _fun(x: onp.ArrayND[np.float64]) -> list[float]: ...
def _jac(x: onp.ArrayND[np.float64]) -> list[list[float]]: ...

###
# root

_fun_root = root(_fun, [1.0, 2.0])
assert_type(_fun_root.x, onp.ArrayND[npc.floating])
assert_type(_fun_root.success, bool)
assert_type(_fun_root.message, str)
assert_type(_fun_root.nfev, int)

assert_type(root(_fun, [1.0, 2.0], method="broyden1").x, onp.ArrayND[npc.floating])
assert_type(root(_fun, [1.0, 2.0], jac=_jac, method="hybr").x, onp.ArrayND[npc.floating])
