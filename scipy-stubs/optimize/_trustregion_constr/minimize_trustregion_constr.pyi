from collections.abc import Callable, Iterable
from typing import Concatenate, Final, Literal, Protocol, TypeAlias, TypedDict, type_check_only

import numpy as np
import optype.numpy as onp
from scipy.optimize._constraints import Bounds, LinearConstraint, NonlinearConstraint, PreparedConstraint
from scipy.optimize._optimize import OptimizeResult as _OptimizeResult
from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator
from .canonical_constraint import CanonicalConstraint

_StopCond: TypeAlias = Literal[0, 1, 2, 3, 4]

_Float: TypeAlias = float | int | bool | np.float64
_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float64]
_Sparse: TypeAlias = spmatrix | sparray
_Matrix: TypeAlias = _Float2D | _Sparse

_HessPFunc: TypeAlias = Callable[Concatenate[_Float1D, _Float1D, ...], _Float1D]
_ObjectiveHessFunc: TypeAlias = Callable[[_Float1D], _Matrix]
_ConstraintsHessFunc: TypeAlias = Callable[[_Float1D, _Float1D, _Float1D], _Matrix]

@type_check_only
class _CGInfo(TypedDict):
    niter: int | bool
    stop_cond: _StopCond
    hits_boundary: bool

@type_check_only
class _Objective(Protocol):
    f: _Float
    g: _Float1D
    nfev: int | bool
    njev: int | bool
    nhev: int | bool

###

TERMINATION_MESSAGES: dict[Literal[0, 1, 2, 3], str]  # undocumented

class OptimizeResult(_OptimizeResult):
    x: _Float1D
    optimality: _Float
    const_violation: _Float
    fun: _Float
    grad: _Float1D
    lagrangian_grad: _Float1D
    nit: int | bool
    nfev: int | bool
    njev: int | bool
    nhev: int | bool
    cg_niter: int | bool
    method: Literal["equality_constrained_sqp", "tr_interior_point"]
    constr: list[_Float]
    jac: list[_Matrix]
    v: list[_FloatND]
    constr_nfev: list[int | bool]
    constr_njev: list[int | bool]
    constr_nhev: list[int | bool]
    tr_radius: _Float
    constr_penalty: _Float
    barrier_tolerance: _Float
    barrier_parameter: _Float
    execution_time: _Float
    message: str
    status: Literal[0, 1, 2, 3]
    cg_stop_cond: _StopCond

# undocumented
class HessianLinearOperator:
    n: Final[int | bool]
    hessp: Final[_HessPFunc]

    def __init__(self, /, hessp: _HessPFunc, n: int | bool) -> None: ...
    def __call__(self, /, x: _Float1D, *args: object) -> LinearOperator: ...

# undocumented
class LagrangianHessian:
    n: Final[int | bool]
    objective_hess: Final[_ObjectiveHessFunc]
    constraints_hess: Final[_ConstraintsHessFunc]

    def __init__(self, /, n: int | bool, objective_hess: _ObjectiveHessFunc, constraints_hess: _ConstraintsHessFunc) -> None: ...
    def __call__(self, /, x: _Float1D, v_eq: _Float1D, v_ineq: _Float1D | None = None) -> LinearOperator: ...

# undocumented
def update_state_sqp(
    state: OptimizeResult,
    x: _Float1D,
    last_iteration_failed: bool,
    objective: _Objective,
    prepared_constraints: Iterable[CanonicalConstraint | PreparedConstraint],
    start_time: float | int | bool,
    tr_radius: _Float,
    constr_penalty: _Float,
    cg_info: _CGInfo,
) -> OptimizeResult: ...

# undocumented
def update_state_ip(
    state: OptimizeResult,
    x: _Float1D,
    last_iteration_failed: bool,
    objective: _Objective,
    prepared_constraints: Iterable[CanonicalConstraint | PreparedConstraint],
    start_time: float | int | bool,
    tr_radius: _Float,
    constr_penalty: _Float,
    cg_info: _CGInfo,
    barrier_parameter: _Float,
    barrier_tolerance: _Float,
) -> OptimizeResult: ...

#
def _minimize_trustregion_constr(
    fun: Callable[Concatenate[_Float1D, ...], _Float],
    x0: onp.ToFloat1D,
    args: tuple[object, ...],
    grad: Callable[Concatenate[_Float1D, ...], onp.ToFloat1D] | None,
    hess: Callable[Concatenate[_Float1D, ...], onp.ToFloat2D | _Sparse | LinearOperator] | None,
    hessp: _HessPFunc | None,
    bounds: Bounds | None,
    constraints: LinearConstraint | NonlinearConstraint | None,
    xtol: float | int | bool = 1e-8,
    gtol: float | int | bool = 1e-8,
    barrier_tol: float | int | bool = 1e-8,
    sparse_jacobian: bool | None = None,
    callback: Callable[[OptimizeResult], None] | None = None,
    maxiter: int | bool = 1000,
    verbose: Literal[0, 1, 2] = 0,
    finite_diff_rel_step: onp.ToFloat1D | None = None,
    initial_constr_penalty: float | int | bool = 1.0,
    initial_tr_radius: float | int | bool = 1.0,
    initial_barrier_parameter: float | int | bool = 0.1,
    initial_barrier_tolerance: float | int | bool = 0.1,
    factorization_method: str | None = None,
    disp: bool = False,
) -> OptimizeResult: ...
