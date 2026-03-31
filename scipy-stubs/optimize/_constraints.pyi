import types
from collections.abc import Callable, Iterable, Sequence
from typing import (
    Any,
    Concatenate,
    Final,
    Generic,
    Literal,
    Never,
    NotRequired,
    SupportsIndex,
    TypeAlias,
    TypedDict,
    overload,
    type_check_only,
)
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.optimize._differentiable_functions import LinearVectorFunction, VectorFunction
from scipy.optimize._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse._typing import _Sparse2D
from scipy.sparse.linalg import LinearOperator

###

_T = TypeVar("_T")
_InexactT = TypeVar("_InexactT", bound=npc.inexact)
_NumberT = TypeVar("_NumberT", bound=npc.number)
_NumberT_co = TypeVar("_NumberT_co", bound=npc.number, default=np.float64 | Any, covariant=True)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int, *tuple[int, ...]], default=_AnyShape, covariant=True)

_Tuple2: TypeAlias = tuple[_T, _T]
_MethodJac: TypeAlias = Literal["2-point", "3-point", "cs"]
_ToFloat2D: TypeAlias = onp.ToFloat2D | _Sparse2D[npc.floating | npc.integer]

_AnyShape: TypeAlias = tuple[Any, ...]
# workaround for mypy & pyright's failure to conform to the overload typing specification
_JustAnyShape: TypeAlias = tuple[Never, Never, Never]

@type_check_only
class _OldConstraint(TypedDict):
    type: Literal["eq", "ineq"]
    fun: Callable[Concatenate[onp.Array1D[np.float64], ...], onp.ToFloat1D]
    jac: NotRequired[Callable[Concatenate[onp.Array1D[np.float64], ...], _ToFloat2D]]
    args: NotRequired[tuple[object, ...]]

@type_check_only
class _BaseConstraint(Generic[_ShapeT_co]):
    keep_feasible: onp.ArrayND[np.bool_, _ShapeT_co]

@type_check_only
class _Constraint(_BaseConstraint[_ShapeT_co], Generic[_ShapeT_co, _NumberT_co]):
    lb: onp.ArrayND[_NumberT_co, _ShapeT_co]
    ub: onp.ArrayND[_NumberT_co, _ShapeT_co]

###

class Bounds(_Constraint[_ShapeT_co, _NumberT_co], Generic[_ShapeT_co, _NumberT_co]):
    @classmethod
    def __class_getitem__(cls, arg: object, /) -> types.GenericAlias: ...

    #
    @overload
    def __init__(
        self: Bounds[tuple[int], np.int_],
        /,
        lb: int | Sequence[int],
        ub: int | Sequence[int],
        keep_feasible: bool | onp.ToBoolStrict1D = False,
    ) -> None: ...
    @overload
    def __init__(
        self: Bounds[_AnyShape, np.int_],
        /,
        lb: onp.SequenceND[int],
        ub: onp.SequenceND[int],
        keep_feasible: bool | onp.ToBoolStrict1D = False,
    ) -> None: ...
    @overload
    def __init__(
        self: Bounds[tuple[int], np.float64],
        /,
        lb: op.JustFloat | list[float] = ...,  # = np.inf
        ub: float | Sequence[float] = ...,  # = -np.inf
        keep_feasible: bool | onp.ToBoolStrict1D = False,
    ) -> None: ...
    @overload
    def __init__(
        self: Bounds[tuple[int], np.float64],
        /,
        lb: float | Sequence[float] = ...,  # = -np.inf
        ub: op.JustFloat | list[float] = ...,  # = np.inf
        keep_feasible: bool | onp.ToBoolStrict1D = False,
    ) -> None: ...
    @overload
    def __init__(
        self: Bounds[_AnyShape, np.float64],
        /,
        lb: onp.SequenceND[list[float]],
        ub: float | onp.SequenceND[float] = ...,  # = -np.inf
        keep_feasible: bool | onp.ToBoolStrict1D = False,
    ) -> None: ...
    @overload
    def __init__(
        self: Bounds[_AnyShape, np.float64],
        /,
        lb: float | onp.SequenceND[float],
        ub: onp.SequenceND[list[float]],
        keep_feasible: bool | onp.ToBoolStrict1D = False,
    ) -> None: ...
    @overload  # ?d, Nd
    def __init__(
        self: Bounds[_AnyShape, _NumberT],
        /,
        lb: onp.ArrayND[_NumberT, _JustAnyShape],
        ub: onp.ArrayND[_NumberT],
        keep_feasible: bool | onp.ToBoolStrict1D = False,
    ) -> None: ...
    @overload  # Nd, ?d
    def __init__(
        self: Bounds[_AnyShape, _NumberT],
        /,
        lb: onp.ArrayND[_NumberT],
        ub: onp.ArrayND[_NumberT, _JustAnyShape],
        keep_feasible: bool | onp.ToBoolStrict1D = False,
    ) -> None: ...
    @overload  # 1d, 1d
    def __init__(
        self: Bounds[tuple[int], _NumberT],
        /,
        lb: _NumberT | onp.Array1D[_NumberT],
        ub: _NumberT | onp.Array1D[_NumberT],
        keep_feasible: bool | onp.ToBoolStrict1D = False,
    ) -> None: ...
    @overload  # 2d, <=2d
    def __init__(
        self: Bounds[tuple[int, int], _NumberT],
        /,
        lb: onp.Array2D[_NumberT],
        ub: onp.Array2D[_NumberT] | onp.Array1D[_NumberT],
        keep_feasible: bool | onp.ToBoolStrict1D | onp.ToBoolStrict2D = False,
    ) -> None: ...
    @overload  # <=2d, 2d
    def __init__(
        self: Bounds[tuple[int, int], _NumberT],
        /,
        lb: onp.Array2D[_NumberT] | onp.Array1D[_NumberT],
        ub: onp.Array2D[_NumberT],
        keep_feasible: bool | onp.ToBoolStrict1D | onp.ToBoolStrict2D = False,
    ) -> None: ...
    @overload  # Nd
    def __init__(
        self,
        /,
        lb: onp.ToFloat | onp.ToFloatND = ...,  # = -np.inf
        ub: onp.ToFloat | onp.ToFloatND = ...,  # = np.inf
        keep_feasible: bool | onp.ToBoolND = False,
    ) -> None: ...

    #
    @overload  # known scalar type
    def residual(
        self: Bounds[_AnyShape, _NumberT], /, x: onp.ToInt | onp.ToInt1D
    ) -> _Tuple2[onp.ArrayND[_NumberT, _ShapeT_co]]: ...
    @overload  # known inexact scalar type
    def residual(
        self: Bounds[_AnyShape, npc.integer],
        /,
        x: onp.CanArray[tuple[()] | tuple[int], np.dtype[_InexactT]] | Sequence[_InexactT],
    ) -> _Tuple2[onp.ArrayND[_InexactT, _ShapeT_co]]: ...
    @overload  # c64 scalar type
    def residual(
        self: Bounds[_AnyShape, npc.integer | np.float64], /, x: onp.ToJustFloat64 | onp.ToJustFloat64_1D
    ) -> _Tuple2[onp.ArrayND[np.float64, _ShapeT_co]]: ...
    @overload  # known floating type
    def residual(
        self: Bounds[_AnyShape, npc.inexact64 | npc.inexact80], /, x: onp.ToFloat64 | onp.ToFloat64_1D
    ) -> _Tuple2[onp.ArrayND[_NumberT_co, _ShapeT_co]]: ...
    @overload  # c128 scalar type
    def residual(
        self: Bounds[_AnyShape, npc.integer | np.float16 | npc.inexact32 | npc.inexact64],
        /,
        x: onp.ToJustComplex128 | onp.ToJustComplex128_1D,
    ) -> _Tuple2[onp.ArrayND[np.complex128, _ShapeT_co]]: ...
    @overload  # known complex type
    def residual(
        self: Bounds[_AnyShape, np.complex128 | npc.complexfloating160], /, x: onp.ToComplex128 | onp.ToComplex128_1D
    ) -> _Tuple2[onp.ArrayND[_NumberT_co, _ShapeT_co]]: ...

class LinearConstraint(_Constraint[tuple[int], np.float64]):
    A: Final[onp.Array2D[np.float64] | _Sparse2D[np.float64]]

    def __init__(
        self,
        /,
        A: _ToFloat2D,
        lb: onp.ToFloat | onp.ToFloat1D = ...,
        ub: onp.ToFloat | onp.ToFloat1D = ...,
        keep_feasible: bool | onp.ToBool1D = False,
    ) -> None: ...
    def residual(self, /, x: onp.ToFloat1D) -> _Tuple2[onp.Array1D[np.float64]]: ...

class NonlinearConstraint(_Constraint[tuple[int], np.float64]):
    fun: Final[Callable[[onp.Array1D[np.float64]], onp.ToFloat1D]]
    finite_diff_rel_step: Final[onp.ToFloat | onp.ToFloat1D | None]
    finite_diff_jac_sparsity: Final[_ToFloat2D | None]
    jac: Final[Callable[[onp.Array1D[np.float64]], _ToFloat2D] | _MethodJac]
    hess: Final[Callable[[onp.Array1D[np.float64]], _ToFloat2D | LinearOperator] | _MethodJac | HessianUpdateStrategy | None]

    def __init__(
        self,
        /,
        fun: Callable[[onp.Array1D[np.float64]], onp.ToFloat1D],
        lb: onp.ToFloat | onp.ToFloat1D,
        ub: onp.ToFloat | onp.ToFloat1D,
        jac: Callable[[onp.Array1D[np.float64]], _ToFloat2D] | _MethodJac = "2-point",
        hess: Callable[[onp.Array1D[np.float64]], _ToFloat2D | LinearOperator] | _MethodJac | HessianUpdateStrategy | None = None,
        keep_feasible: bool | onp.ToBool1D = False,
        finite_diff_rel_step: onp.ToFloat | onp.ToFloat1D | None = None,
        finite_diff_jac_sparsity: _ToFloat2D | None = None,
    ) -> None: ...

class PreparedConstraint(_BaseConstraint[_ShapeT_co], Generic[_ShapeT_co]):  # undocumented
    fun: Final[VectorFunction | LinearVectorFunction]
    bounds: Final[_Tuple2[onp.Array1D[np.float64]]]

    lb: onp.ArrayND[np.float64, _ShapeT_co]
    ub: onp.ArrayND[np.float64, _ShapeT_co]

    @classmethod
    def __class_getitem__(cls, arg: object, /) -> types.GenericAlias: ...
    def __init__(
        self,
        /,
        constraint: _Constraint[_ShapeT_co],
        x0: onp.ToFloat1D,
        sparse_jacobian: bool | None = None,
        finite_diff_bounds: _Tuple2[onp.ToFloat | onp.ToFloat2D] = ...,
    ) -> None: ...
    def violation(self, /, x: onp.ToFloat1D) -> onp.Array1D[np.float64]: ...

def new_bounds_to_old(lb: onp.ToFloat1D, ub: onp.ToFloat1D, n: SupportsIndex) -> list[_Tuple2[float]]: ...  # undocumented
def old_bound_to_new(bounds: Iterable[_Tuple2[float]]) -> _Tuple2[onp.Array1D[np.float64]]: ...  # undocumented
def strict_bounds(
    lb: onp.ToFloat1D, ub: onp.ToFloat1D, keep_feasible: onp.ToBool1D, n_vars: SupportsIndex
) -> _Tuple2[onp.Array1D[np.float64]]: ...  # undocumented
def new_constraint_to_old(con: _BaseConstraint, x0: onp.ToFloatND) -> list[_OldConstraint]: ...  # undocumented
def old_constraint_to_new(ic: int, con: _OldConstraint) -> NonlinearConstraint: ...  # undocumented
