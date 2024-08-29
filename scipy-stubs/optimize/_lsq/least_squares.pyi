from scipy._typing import Untyped

from scipy.optimize import OptimizeResult as OptimizeResult
from scipy.optimize._minimize import Bounds as Bounds
from scipy.optimize._numdiff import approx_derivative as approx_derivative, group_columns as group_columns
from scipy.sparse import issparse as issparse
from scipy.sparse.linalg import LinearOperator as LinearOperator
from .common import EPS as EPS, in_bounds as in_bounds, make_strictly_feasible as make_strictly_feasible
from .dogbox import dogbox as dogbox
from .trf import trf as trf

TERMINATION_MESSAGES: Untyped
FROM_MINPACK_TO_COMMON: Untyped

def call_minpack(fun, x0, jac, ftol, xtol, gtol, max_nfev, x_scale, diff_step) -> Untyped: ...
def prepare_bounds(bounds, n) -> Untyped: ...
def check_tolerance(ftol, xtol, gtol, method) -> Untyped: ...
def check_x_scale(x_scale, x0) -> Untyped: ...
def check_jac_sparsity(jac_sparsity, m, n) -> Untyped: ...
def huber(z, rho, cost_only): ...
def soft_l1(z, rho, cost_only): ...
def cauchy(z, rho, cost_only): ...
def arctan(z, rho, cost_only): ...

IMPLEMENTED_LOSSES: Untyped

def construct_loss_function(m, loss, f_scale) -> Untyped: ...
def least_squares(
    fun,
    x0,
    jac: str = "2-point",
    bounds=...,
    method: str = "trf",
    ftol: float = 1e-08,
    xtol: float = 1e-08,
    gtol: float = 1e-08,
    x_scale: float = 1.0,
    loss: str = "linear",
    f_scale: float = 1.0,
    diff_step: Untyped | None = None,
    tr_solver: Untyped | None = None,
    tr_options: Untyped | None = None,
    jac_sparsity: Untyped | None = None,
    max_nfev: Untyped | None = None,
    verbose: int = 0,
    args=(),
    kwargs: Untyped | None = None,
) -> Untyped: ...
