from collections.abc import Callable
from typing import Any, ClassVar, Final, Generic, Literal, Protocol, TypeAlias, TypedDict, type_check_only
from typing_extensions import Self, TypeVar, TypeVarTuple, Unpack, override

import numpy as np
import optype.numpy as onp

__all__ = ["complex_ode", "ode"]

_SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.inexact[Any], default=np.float64 | np.complex128)
_Ts = TypeVarTuple("_Ts", default=Unpack[tuple[()]])

@type_check_only
class _IntegratorParams(TypedDict, total=False):
    with_jacobian: bool
    rtol: float | int | bool
    atol: float | int | bool
    lband: float | int | bool | None
    uband: float | int | bool | None
    order: int | bool
    nsteps: int | bool
    max_step: float | int | bool
    min_step: float | int | bool
    first_step: float | int | bool
    ixpr: int | bool
    max_hnil: int | bool
    max_order_ns: int | bool
    max_order_s: int | bool
    method: Literal["adams", "bds"] | None
    safety: float | int | bool
    ifactor: float | int | bool
    dfactor: float | int | bool
    beta: float | int | bool
    verbosity: int | bool

@type_check_only
class _ODEFuncF(Protocol[Unpack[_Ts]]):
    def __call__(
        self,
        t: float | int | bool,
        y: float | int | bool | onp.ArrayND[np.float64],
        /,
        *args: Unpack[_Ts],
    ) -> float | int | bool | onp.ArrayND[np.floating[Any]]: ...

@type_check_only
class _ODEFuncC(Protocol[Unpack[_Ts]]):
    def __call__(
        self,
        t: float | int | bool,
        y: complex | float | int | bool | onp.ArrayND[np.complex128],
        /,
        *args: Unpack[_Ts],
    ) -> complex | float | int | bool | onp.ArrayND[np.complexfloating[Any, Any]]: ...

_SolOutFunc: TypeAlias = Callable[[float | int | bool, onp.Array1D[np.inexact[Any]]], Literal[0, -1]]

###

class ode(Generic[Unpack[_Ts]]):
    stiff: int | bool
    f: _ODEFuncF[Unpack[_Ts]]
    f_params: tuple[()] | tuple[Unpack[_Ts]]
    jac: _ODEFuncF[Unpack[_Ts]] | None
    jac_params: tuple[()] | tuple[Unpack[_Ts]]
    t: float | int | bool
    def __init__(self, /, f: _ODEFuncF[Unpack[_Ts]], jac: _ODEFuncF[Unpack[_Ts]] | None = None) -> None: ...
    @property
    def y(self, /) -> float | int | bool: ...
    def integrate(self, /, t: float | int | bool, step: bool = False, relax: bool = False) -> float | int | bool: ...
    def set_initial_value(self, /, y: onp.ToComplex | onp.ToComplexND, t: float | int | bool = 0.0) -> Self: ...
    def set_integrator(self, /, name: str, **integrator_params: Unpack[_IntegratorParams]) -> Self: ...
    def set_f_params(self, /, *args: Unpack[_Ts]) -> Self: ...
    def set_jac_params(self, /, *args: Unpack[_Ts]) -> Self: ...
    def set_solout(self, /, solout: _SolOutFunc) -> None: ...
    def get_return_code(self, /) -> Literal[-7, -6, -5, -4, -3, -2, -1, 1, 2]: ...
    def successful(self, /) -> bool: ...

class complex_ode(ode[Unpack[_Ts]], Generic[Unpack[_Ts]]):
    cf: _ODEFuncC[Unpack[_Ts]]
    cjac: _ODEFuncC[Unpack[_Ts]] | None
    tmp: onp.Array1D[np.float64]
    def __init__(self, /, f: _ODEFuncC[Unpack[_Ts]], jac: _ODEFuncC[Unpack[_Ts]] | None = None) -> None: ...
    @property
    @override
    def y(self, /) -> complex: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def integrate(self, /, t: float | int | bool, step: bool = False, relax: bool = False) -> complex: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

def find_integrator(name: str) -> type[IntegratorBase] | None: ...

class IntegratorConcurrencyError(RuntimeError):
    def __init__(self, /, name: str) -> None: ...

class IntegratorBase(Generic[_SCT_co]):
    runner: ClassVar[Callable[..., tuple[Any, ...]] | None]  # fortran function or unavailable
    supports_run_relax: ClassVar[Literal[0, 1] | None] = None
    supports_step: ClassVar[Literal[0, 1] | None] = None
    supports_solout: ClassVar[bool] = ...
    scalar: ClassVar[type] = ...

    handle: int | bool
    success: Literal[0, 1] | bool | None = None
    integrator_classes: list[type[IntegratorBase]]
    istate: int | bool | None = None

    def acquire_new_handle(self, /) -> None: ...
    def check_handle(self, /) -> None: ...
    def reset(self, /, n: int | bool, has_jac: bool) -> None: ...
    def run(
        self,
        /,
        f: Callable[..., _SCT_co],
        jac: Callable[..., onp.ArrayND[_SCT_co]] | None,
        y0: complex | float | int | bool,
        t0: float | int | bool,
        t1: float | int | bool,
        f_params: tuple[object, ...],
        jac_params: tuple[object, ...],
    ) -> tuple[_SCT_co, float | int | bool]: ...
    def step(
        self,
        /,
        f: Callable[..., _SCT_co],
        jac: Callable[..., onp.ArrayND[_SCT_co]],
        y0: complex | float | int | bool,
        t0: float | int | bool,
        t1: float | int | bool,
        f_params: tuple[object, ...],
        jac_params: tuple[object, ...],
    ) -> tuple[_SCT_co, float | int | bool]: ...
    def run_relax(
        self,
        /,
        f: Callable[..., _SCT_co],
        jac: Callable[..., onp.ArrayND[_SCT_co]],
        y0: complex | float | int | bool,
        t0: float | int | bool,
        t1: float | int | bool,
        f_params: tuple[object, ...],
        jac_params: tuple[object, ...],
    ) -> tuple[_SCT_co, float | int | bool]: ...

class vode(IntegratorBase[_SCT_co], Generic[_SCT_co]):
    messages: ClassVar[dict[int | bool, str]] = ...

    active_global_handle: int | bool
    meth: int | bool
    with_jacobian: bool
    rtol: float | int | bool
    atol: float | int | bool
    mu: float | int | bool
    ml: float | int | bool
    order: int | bool
    nsteps: int | bool
    max_step: float | int | bool
    min_step: float | int | bool
    first_step: float | int | bool
    initialized: bool
    rwork: onp.Array1D[np.float64]
    iwork: onp.Array1D[np.int32]
    call_args: list[float | int | bool | onp.ArrayND[np.float64] | onp.ArrayND[np.int32]]

    def __init__(
        self,
        /,
        method: Literal["adams", "bdf"] = "adams",
        with_jacobian: bool = False,
        rtol: float | int | bool = 1e-06,
        atol: float | int | bool = 1e-12,
        lband: float | int | bool | None = None,
        uband: float | int | bool | None = None,
        order: int | bool = 12,
        nsteps: int | bool = 500,
        max_step: float | int | bool = 0.0,
        min_step: float | int | bool = 0.0,
        first_step: float | int | bool = 0.0,
    ) -> None: ...

class zvode(vode[np.complex128]):
    active_global_handle: int | bool
    zwork: onp.Array1D[np.complex128]
    call_args: list[float | int | bool | onp.ArrayND[np.complex128] | onp.ArrayND[np.float64] | onp.ArrayND[np.int32]]  # type: ignore[assignment] # pyright: ignore[reportIncompatibleVariableOverride]
    initialized: bool

class dopri5(IntegratorBase[np.float64]):
    name: ClassVar[str] = "dopri5"
    messages: ClassVar[dict[int | bool, str]] = ...

    rtol: Final[float | int | bool]
    atol: Final[float | int | bool]
    nsteps: Final[int | bool]
    max_step: Final[float | int | bool]
    first_step: Final[float | int | bool]
    safety: Final[float | int | bool]
    ifactor: Final[float | int | bool]
    dfactor: Final[float | int | bool]
    beta: Final[float | int | bool]
    verbosity: Final[int | bool]
    solout: Callable[[float | int | bool, onp.Array1D[np.inexact[Any]]], Literal[0, -1]] | None
    solout_cmplx: bool
    iout: int | bool
    work: onp.Array1D[np.float64]
    iwork: onp.Array1D[np.int32]
    call_args: list[float | int | bool | Callable[..., Literal[0, -1, 1]] | onp.ArrayND[np.float64] | onp.ArrayND[np.int32]]

    def __init__(
        self,
        /,
        rtol: float | int | bool = 1e-06,
        atol: float | int | bool = 1e-12,
        nsteps: int | bool = 500,
        max_step: float | int | bool = 0.0,
        first_step: float | int | bool = 0.0,
        safety: float | int | bool = 0.9,
        ifactor: float | int | bool = 10.0,
        dfactor: float | int | bool = 0.2,
        beta: float | int | bool = 0.0,
        method: None = None,  # unused
        verbosity: int | bool = -1,
    ) -> None: ...
    def set_solout(self, /, solout: _SolOutFunc | None, complex: bool = False) -> None: ...
    def _solout(
        self,
        /,
        nr: int | bool,  # unused
        xold: object,  # unused
        x: float | int | bool,
        y: onp.Array1D[np.floating[Any]],
        nd: int | bool,  # unused
        icomp: int | bool,  # unused
        con: object,  # unused
    ) -> Literal[0, -1, 1]: ...

class dop853(dopri5):
    name: ClassVar[str] = "dop853"
    def __init__(
        self,
        /,
        rtol: float | int | bool = 1e-06,
        atol: float | int | bool = 1e-12,
        nsteps: int | bool = 500,
        max_step: float | int | bool = 0.0,
        first_step: float | int | bool = 0.0,
        safety: float | int | bool = 0.9,
        ifactor: float | int | bool = 6.0,
        dfactor: float | int | bool = 0.3,
        beta: float | int | bool = 0.0,
        method: None = None,  # ignored
        verbosity: int | bool = -1,
    ) -> None: ...

class lsoda(IntegratorBase[np.float64]):
    active_global_handle: ClassVar[int | bool] = 0
    messages: ClassVar[dict[int | bool, str]] = ...

    with_jacobian: Final[bool]
    rtol: Final[float | int | bool]
    atol: Final[float | int | bool]
    mu: Final[float | int | bool | None]
    ml: Final[float | int | bool | None]
    max_order_ns: Final[int | bool]
    max_order_s: Final[int | bool]
    nsteps: Final[int | bool]
    max_step: Final[float | int | bool]
    min_step: Final[float | int | bool]
    first_step: Final[float | int | bool]
    ixpr: Final[int | bool]
    max_hnil: Final[int | bool]
    initialized: Final[bool]
    rwork: onp.Array1D[np.float64]
    iwork: onp.Array1D[np.int32]
    call_args: list[float | int | bool | onp.Array1D[np.float64] | onp.Array1D[np.int32]]
    def __init__(
        self,
        /,
        with_jacobian: bool = False,
        rtol: float | int | bool = 1e-06,
        atol: float | int | bool = 1e-12,
        lband: float | int | bool | None = None,
        uband: float | int | bool | None = None,
        nsteps: int | bool = 500,
        max_step: float | int | bool = 0.0,
        min_step: float | int | bool = 0.0,
        first_step: float | int | bool = 0.0,
        ixpr: int | bool = 0,
        max_hnil: int | bool = 0,
        max_order_ns: int | bool = 12,
        max_order_s: int | bool = 5,
        method: None = None,  # ignored
    ) -> None: ...
