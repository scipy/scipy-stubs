from scipy._typing import Untyped

from scipy.optimize import OptimizeResult as OptimizeResult
from .common import print_header_linear as print_header_linear, print_iteration_linear as print_iteration_linear

def compute_kkt_optimality(g, on_bound) -> Untyped: ...
def bvls(A, b, x_lsq, lb, ub, tol, max_iter, verbose, rcond: Untyped | None = None) -> Untyped: ...
