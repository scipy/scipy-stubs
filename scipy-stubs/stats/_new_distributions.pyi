from typing import Any, ClassVar, Generic, TypeAlias, overload
from typing_extensions import Never, TypeVar, Unpack

import numpy as np
import optype.numpy as onp
from ._distribution_infrastructure import ContinuousDistribution, _DistOpts, _RealDomain, _RealParameter

__all__ = ["Normal", "Uniform"]

###

_Float: TypeAlias = np.floating[Any]

_NT = TypeVar("_NT", default=int | bool)
_0D: TypeAlias = tuple[()]  # noqa: PYI042
_1D: TypeAlias = tuple[_NT]  # noqa: PYI042
_2D: TypeAlias = tuple[_NT, _NT]  # noqa: PYI042
_3D: TypeAlias = tuple[_NT, _NT, _NT]  # noqa: PYI042
_ND: TypeAlias = tuple[_NT, ...]

_ToFloat_1D: TypeAlias = onp.ToFloatStrict1D | onp.ToFloat
_ToFloat_2D: TypeAlias = onp.ToFloatStrict2D | _ToFloat_1D
_ToFloat_3D: TypeAlias = onp.ToFloatStrict3D | _ToFloat_2D
_ToFloat_ND: TypeAlias = onp.ToFloatND | onp.ToFloat

_FloatT = TypeVar("_FloatT", bound=_Float, default=_Float)
_FloatT_co = TypeVar("_FloatT_co", bound=_Float, default=_Float, covariant=True)

_ShapeT = TypeVar("_ShapeT", bound=_ND, default=_ND)
_ShapeT_co = TypeVar("_ShapeT_co", bound=_ND, default=_ND, covariant=True)

###

class Normal(ContinuousDistribution[_FloatT_co, _ShapeT_co], Generic[_ShapeT_co, _FloatT_co]):
    _mu_domain: ClassVar[_RealDomain] = ...
    _mu_param: ClassVar[_RealParameter] = ...
    _sigma_domain: ClassVar[_RealDomain] = ...
    _sigma_param: ClassVar[_RealParameter] = ...
    _x_support: ClassVar[_RealDomain] = ...
    _x_param: ClassVar[_RealParameter] = ...
    _normalization: ClassVar[np.float64] = ...
    _log_normalization: ClassVar[np.float64] = ...

    @property
    def mu(self, /) -> _FloatT_co | onp.Array[_ShapeT_co, _FloatT_co]: ...
    @property
    def sigma(self, /) -> _FloatT_co | onp.Array[_ShapeT_co, _FloatT_co]: ...

    # TODO(jorenham): __new__

    #
    @overload  # default
    def __init__(self: Normal[_0D, np.float64], /, **kw: Unpack[_DistOpts]) -> None: ...
    @overload  # mu: N-d <known shape, dtype>
    def __init__(  # pyright: ignore[reportOverlappingOverload]
        self: Normal[_ShapeT, _FloatT],
        /,
        *,
        mu: onp.CanArrayND[_FloatT, _ShapeT],
        sigma: onp.CanArrayND[_FloatT | np.integer[Any] | np.bool_, _ShapeT] | onp.ToInt,
        **kw: Unpack[_DistOpts],
    ) -> None: ...
    @overload  # sigma: N-d <known shape, dtype>
    def __init__(  # pyright: ignore[reportOverlappingOverload]
        self: Normal[_ShapeT, _FloatT],
        /,
        *,
        mu: onp.CanArrayND[_FloatT | np.integer[Any] | np.bool_, _ShapeT] | onp.ToInt,
        sigma: onp.CanArrayND[_FloatT, _ShapeT],
        **kw: Unpack[_DistOpts],
    ) -> None: ...
    @overload  # mu, sigma: 0-d float | int | bool
    def __init__(  # pyright: ignore[reportOverlappingOverload]
        self: Normal[_0D, np.float64],
        /,
        *,
        mu: float | int | bool,
        sigma: float | int | bool | onp.ToInt,
        **kw: Unpack[_DistOpts],
    ) -> None: ...
    @overload  # mu, sigma: 0-d float | int | bool
    def __init__(  # pyright: ignore[reportOverlappingOverload]
        self: Normal[_0D, np.float64],
        /,
        *,
        mu: float | int | bool | onp.ToInt,
        sigma: float | int | bool,
        **kw: Unpack[_DistOpts],
    ) -> None: ...
    @overload  # mu: 0-d <known dtype>, sigma: 0-d
    def __init__(self: Normal[_0D, _FloatT], /, *, mu: _FloatT, sigma: _FloatT | onp.ToInt, **kw: Unpack[_DistOpts]) -> None: ...  # pyright: ignore[reportOverlappingOverload]
    @overload  # a, sigma: 0-d <known dtype>
    def __init__(self: Normal[_0D, _FloatT], /, *, mu: _FloatT | onp.ToInt, sigma: _FloatT, **kw: Unpack[_DistOpts]) -> None: ...  # pyright: ignore[reportOverlappingOverload]
    @overload  # a, sigma: 0-d
    def __init__(self: Normal[_0D], /, *, mu: onp.ToFloat = 0.0, sigma: onp.ToFloat = 1.0, **kw: Unpack[_DistOpts]) -> None: ...  # pyright: ignore[reportOverlappingOverload]
    @overload  # mu: 1-d
    def __init__(self: Normal[_1D], /, *, mu: onp.ToFloatStrict1D, sigma: _ToFloat_1D = 1.0, **kw: Unpack[_DistOpts]) -> None: ...  # pyright: ignore[reportOverlappingOverload]
    @overload  # sigma: 1-d
    def __init__(self: Normal[_1D], /, *, mu: _ToFloat_1D = 0.0, sigma: onp.ToFloatStrict1D, **kw: Unpack[_DistOpts]) -> None: ...  # pyright: ignore[reportOverlappingOverload]
    @overload  # mu: 2-d
    def __init__(self: Normal[_2D], /, *, mu: onp.ToFloatStrict2D, sigma: _ToFloat_2D = 1.0, **kw: Unpack[_DistOpts]) -> None: ...  # pyright: ignore[reportOverlappingOverload]
    @overload  # sigma: 2-d
    def __init__(self: Normal[_2D], /, *, mu: _ToFloat_2D = 0.0, sigma: onp.ToFloatStrict2D, **kw: Unpack[_DistOpts]) -> None: ...  # pyright: ignore[reportOverlappingOverload]
    @overload  # mu: 3-d
    def __init__(self: Normal[_2D], /, *, mu: onp.ToFloatStrict3D, sigma: _ToFloat_3D = 1.0, **kw: Unpack[_DistOpts]) -> None: ...  # pyright: ignore[reportOverlappingOverload]
    @overload  # sigma: 3-d
    def __init__(self: Normal[_3D], /, *, mu: _ToFloat_3D = 0.0, sigma: onp.ToFloatStrict3D, **kw: Unpack[_DistOpts]) -> None: ...  # pyright: ignore[reportOverlappingOverload]
    @overload  # mu: >=1-d
    def __init__(  # pyright: ignore[reportOverlappingOverload]
        self: Normal[onp.AtLeast1D],
        /,
        *,
        mu: onp.ToFloatND,
        sigma: _ToFloat_ND = 1.0,
        **kw: Unpack[_DistOpts],
    ) -> None: ...
    @overload  # sigma: >=1-d
    def __init__(  # pyright: ignore[reportOverlappingOverload]
        self: Normal[onp.AtLeast1D],
        /,
        *,
        mu: _ToFloat_ND = 0.0,
        sigma: onp.ToFloatND,
        **kw: Unpack[_DistOpts],
    ) -> None: ...

class StandardNormal(Normal[tuple[()], np.float64]):  # undocumented
    mu: ClassVar[np.float64] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
    sigma: ClassVar[np.float64] = ...  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, /, **kw: Unpack[_DistOpts]) -> None: ...

class Uniform(ContinuousDistribution[_FloatT_co, _ShapeT_co], Generic[_ShapeT_co, _FloatT_co]):
    _a_domain: ClassVar[_RealDomain] = ...
    _a_param: ClassVar[_RealParameter] = ...
    _b_domain: ClassVar[_RealDomain] = ...
    _b_param: ClassVar[_RealParameter] = ...
    _x_support: ClassVar[_RealDomain] = ...
    _x_param: ClassVar[_RealParameter] = ...

    @property
    def a(self, /) -> _FloatT_co | onp.Array[_ShapeT_co, _FloatT_co]: ...
    @property
    def b(self, /) -> _FloatT_co | onp.Array[_ShapeT_co, _FloatT_co]: ...
    @property
    def ab(self, /) -> _FloatT_co | onp.Array[_ShapeT_co, _FloatT_co]: ...  # b - a

    # NOTE: `a` and `b` are both required; the defaults are just there to confuse you or something...
    @overload  # a: 0-d float | int | bool, b: 0-d
    def __init__(
        self: Uniform[_0D, np.float64], /, *, a: float | int | bool, b: float | int | bool | onp.ToInt, **kw: Unpack[_DistOpts]
    ) -> None: ...
    @overload  # a, b: 0-d float | int | bool
    def __init__(
        self: Uniform[_0D, np.float64], /, *, a: float | int | bool | onp.ToInt, b: float | int | bool, **kw: Unpack[_DistOpts]
    ) -> None: ...
    @overload  # a: 0-d <known dtype>, b: 0-d
    def __init__(self: Uniform[_0D, _FloatT], /, *, a: _FloatT, b: _FloatT | onp.ToInt, **kw: Unpack[_DistOpts]) -> None: ...
    @overload  # a, b: 0-d <known dtype>
    def __init__(self: Uniform[_0D, _FloatT], /, *, a: _FloatT | onp.ToInt, b: _FloatT, **kw: Unpack[_DistOpts]) -> None: ...
    @overload  # a, b: 0-d
    def __init__(self: Uniform[_0D], /, *, a: onp.ToFloat, b: onp.ToFloat, **kw: Unpack[_DistOpts]) -> None: ...
    @overload  # a: 1-d
    def __init__(self: Uniform[_1D], /, *, a: onp.ToFloatStrict1D, b: _ToFloat_1D, **kw: Unpack[_DistOpts]) -> None: ...
    @overload  # b: 1-d
    def __init__(self: Uniform[_1D], /, *, a: _ToFloat_1D, b: onp.ToFloatStrict1D, **kw: Unpack[_DistOpts]) -> None: ...
    @overload  # a: 2-d
    def __init__(self: Uniform[_2D], /, *, a: onp.ToFloatStrict2D, b: _ToFloat_2D, **kw: Unpack[_DistOpts]) -> None: ...
    @overload  # b: 2-d
    def __init__(self: Uniform[_2D], /, *, a: _ToFloat_2D, b: onp.ToFloatStrict2D, **kw: Unpack[_DistOpts]) -> None: ...
    @overload  # a: 3-d
    def __init__(self: Uniform[_2D], /, *, a: onp.ToFloatStrict3D, b: _ToFloat_3D, **kw: Unpack[_DistOpts]) -> None: ...
    @overload  # b: 3-d
    def __init__(self: Uniform[_3D], /, *, a: _ToFloat_3D, b: onp.ToFloatStrict3D, **kw: Unpack[_DistOpts]) -> None: ...
    @overload  # a: >=1-d
    def __init__(self: Uniform[onp.AtLeast1D], /, *, a: onp.ToFloatND, b: _ToFloat_ND, **kw: Unpack[_DistOpts]) -> None: ...
    @overload  # b: >=1-d
    def __init__(self: Uniform[onp.AtLeast1D], /, *, a: _ToFloat_ND, b: onp.ToFloatND, **kw: Unpack[_DistOpts]) -> None: ...
    @overload  # a: None -> ValueError
    def __init__(self, /, *, a: None = None, b: _ToFloat_ND | None = None, **kw: Unpack[_DistOpts]) -> Never: ...
    @overload  # b: None -> ValueError
    def __init__(self, /, *, a: _ToFloat_ND | None = None, b: None = None, **kw: Unpack[_DistOpts]) -> Never: ...
