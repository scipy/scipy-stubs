from scipy._typing import Untyped

from scipy._lib._array_api import array_namespace as array_namespace, xp_atleast_nd as xp_atleast_nd
from ._constraints import old_bound_to_new as old_bound_to_new
from ._optimize import MemoizeJac as MemoizeJac, OptimizeResult as OptimizeResult

MSG_NONE: int
MSG_ITER: int
MSG_INFO: int
MSG_VERS: int
MSG_EXIT: int
MSG_ALL: Untyped
MSGS: Untyped
INFEASIBLE: int
LOCALMINIMUM: int
FCONVERGED: int
XCONVERGED: int
MAXFUN: int
LSFAIL: int
CONSTANT: int
NOPROGRESS: int
USERABORT: int
RCSTRINGS: Untyped

def fmin_tnc(
    func,
    x0,
    fprime: Untyped | None = None,
    args=(),
    approx_grad: int = 0,
    bounds: Untyped | None = None,
    epsilon: float = 1e-08,
    scale: Untyped | None = None,
    offset: Untyped | None = None,
    messages=...,
    maxCGit: int = -1,
    maxfun: Untyped | None = None,
    eta: int = -1,
    stepmx: int = 0,
    accuracy: int = 0,
    fmin: int = 0,
    ftol: int = -1,
    xtol: int = -1,
    pgtol: int = -1,
    rescale: int = -1,
    disp: Untyped | None = None,
    callback: Untyped | None = None,
) -> Untyped: ...
