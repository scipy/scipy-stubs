from scipy._typing import Untyped

from scipy.linalg import LinAlgError as LinAlgError, cho_factor as cho_factor, cho_solve as cho_solve
from scipy.sparse import issparse as issparse
from scipy.sparse.linalg import LinearOperator as LinearOperator, aslinearoperator as aslinearoperator

EPS: Untyped

def intersect_trust_region(x, s, Delta) -> Untyped: ...
def solve_lsq_trust_region(
    n, m, uf, s, V, Delta, initial_alpha: Untyped | None = None, rtol: float = 0.01, max_iter: int = 10
) -> Untyped: ...
def solve_trust_region_2d(B, g, Delta) -> Untyped: ...
def update_tr_radius(Delta, actual_reduction, predicted_reduction, step_norm, bound_hit) -> Untyped: ...
def build_quadratic_1d(J, g, s, diag: Untyped | None = None, s0: Untyped | None = None) -> Untyped: ...
def minimize_quadratic_1d(a, b, lb, ub, c: int = 0) -> Untyped: ...
def evaluate_quadratic(J, g, s, diag: Untyped | None = None) -> Untyped: ...
def in_bounds(x, lb, ub) -> Untyped: ...
def step_size_to_bound(x, s, lb, ub) -> Untyped: ...
def find_active_constraints(x, lb, ub, rtol: float = 1e-10) -> Untyped: ...
def make_strictly_feasible(x, lb, ub, rstep: float = 1e-10) -> Untyped: ...
def CL_scaling_vector(x, g, lb, ub) -> Untyped: ...
def reflective_transformation(y, lb, ub) -> Untyped: ...
def print_header_nonlinear(): ...
def print_iteration_nonlinear(iteration, nfev, cost, cost_reduction, step_norm, optimality): ...
def print_header_linear(): ...
def print_iteration_linear(iteration, cost, cost_reduction, step_norm, optimality): ...
def compute_grad(J, f) -> Untyped: ...
def compute_jac_scale(J, scale_inv_old: Untyped | None = None) -> Untyped: ...
def left_multiplied_operator(J, d) -> Untyped: ...
def right_multiplied_operator(J, d) -> Untyped: ...
def regularized_lsq_operator(J, diag) -> Untyped: ...
def right_multiply(J, d, copy: bool = True) -> Untyped: ...
def left_multiply(J, d, copy: bool = True) -> Untyped: ...
def check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol) -> Untyped: ...
def scale_for_robust_loss_function(J, f, rho) -> Untyped: ...
