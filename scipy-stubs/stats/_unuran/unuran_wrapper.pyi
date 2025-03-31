from collections.abc import Callable
from typing import NamedTuple, Protocol, overload, type_check_only

import numpy as np
import optype.numpy as onp
import scipy.stats as stats
from scipy._typing import ToRNG

__all__ = ["DiscreteAliasUrn", "NumericalInversePolynomial", "TransformedDensityRejection", "UNURANError"]

@type_check_only
class _HasSupport(Protocol):
    @property
    def support(self, /) -> tuple[float | int | bool, float | int | bool]: ...

@type_check_only
class _HasPMF(_HasSupport, Protocol):
    @property
    def pmf(self, /) -> Callable[..., float | int | bool]: ...

@type_check_only
class _HasPDF(_HasSupport, Protocol):
    @property
    def pdf(self, /) -> Callable[..., float | int | bool]: ...

@type_check_only
class _HasCDF(_HasPDF, Protocol):
    @property
    def cdf(self, /) -> Callable[..., float | int | bool]: ...

@type_check_only
class _TDRDist(_HasPDF, Protocol):
    @property
    def dpdf(self, /) -> Callable[..., float | int | bool]: ...

@type_check_only
class _PINVDist(_HasCDF, Protocol):
    @property
    def logpdf(self, /) -> Callable[..., float | int | bool]: ...

@type_check_only
class _PPFMethodMixin:
    @overload
    def ppf(self, /, u: onp.ToFloat) -> float | int | bool: ...
    @overload
    def ppf(self, /, u: onp.ToFloatND) -> onp.ArrayND[np.float64]: ...

class UNURANError(RuntimeError): ...

class UError(NamedTuple):
    max_error: float | int | bool
    mean_absolute_error: float | int | bool

class Method:
    @overload
    def rvs(self, /, size: None = None, random_state: ToRNG = None) -> float | int | bool | int | bool: ...
    @overload
    def rvs(self, /, size: int | bool | tuple[int | bool, ...]) -> onp.ArrayND[np.float64 | np.int_]: ...
    def set_random_state(self, /, random_state: ToRNG = None) -> None: ...

class TransformedDensityRejection(Method):
    def __init__(
        self,
        /,
        dist: _TDRDist,
        *,
        mode: float | int | bool | None = ...,
        center: float | int | bool | None = ...,
        domain: tuple[float | int | bool, float | int | bool] | None = ...,
        c: float | int | bool = ...,
        construction_points: onp.ToFloatND = ...,
        use_dars: bool = ...,
        max_squeeze_hat_ratio: float | int | bool = ...,
        random_state: ToRNG = ...,
    ) -> None: ...
    @property
    def hat_area(self, /) -> float | int | bool: ...
    @property
    def squeeze_hat_ratio(self, /) -> float | int | bool: ...
    @property
    def squeeze_area(self, /) -> float | int | bool: ...
    @overload
    def ppf_hat(self, /, u: onp.ToFloat) -> float | int | bool: ...
    @overload
    def ppf_hat(self, /, u: onp.ToScalar | onp.ToArrayND) -> float | int | bool | onp.ArrayND[np.float64]: ...

class SimpleRatioUniforms(Method):
    def __init__(
        self,
        /,
        dist: _HasPDF,
        *,
        mode: float | int | bool | None = ...,
        pdf_area: float | int | bool = ...,
        domain: tuple[float | int | bool, float | int | bool] | None = ...,
        cdf_at_mode: float | int | bool = ...,
        random_state: ToRNG = ...,
    ) -> None: ...

class NumericalInversePolynomial(_PPFMethodMixin, Method):
    def __init__(
        self,
        /,
        dist: _PINVDist,
        *,
        mode: float | int | bool | None = ...,
        center: float | int | bool | None = ...,
        domain: tuple[float | int | bool, float | int | bool] | None = ...,
        order: int | bool = ...,
        u_resolution: float | int | bool = ...,
        random_state: ToRNG = ...,
    ) -> None: ...
    @property
    def intervals(self, /) -> int | bool: ...
    @overload
    def cdf(self, /, x: onp.ToFloat) -> float | int | bool: ...
    @overload
    def cdf(self, /, x: onp.ToFloat | onp.ToFloatND) -> float | int | bool | onp.ArrayND[np.float64]: ...
    def u_error(self, /, sample_size: int | bool = ...) -> UError: ...
    def qrvs(
        self,
        /,
        size: int | bool | tuple[int | bool, ...] | None = ...,
        d: int | bool | None = ...,
        qmc_engine: stats.qmc.QMCEngine | None = ...,
    ) -> float | int | bool | onp.ArrayND[np.float64]: ...

class NumericalInverseHermite(_PPFMethodMixin, Method):
    def __init__(
        self,
        /,
        dist: _HasCDF,
        *,
        domain: tuple[float | int | bool, float | int | bool] | None = ...,
        order: int | bool = ...,
        u_resolution: float | int | bool = ...,
        construction_points: onp.ToFloatND | None = ...,
        max_intervals: int | bool = ...,
        random_state: ToRNG = ...,
    ) -> None: ...
    @property
    def intervals(self, /) -> int | bool: ...
    @property
    def midpoint_error(self, /) -> float | int | bool: ...
    def u_error(self, /, sample_size: int | bool = ...) -> UError: ...
    def qrvs(
        self,
        /,
        size: int | bool | tuple[int | bool, ...] | None = ...,
        d: int | bool | None = ...,
        qmc_engine: stats.qmc.QMCEngine | None = ...,
    ) -> float | int | bool | onp.ArrayND[np.float64]: ...

class DiscreteAliasUrn(Method):
    def __init__(
        self,
        /,
        dist: onp.ToFloat | onp.ToFloatND | _HasPMF,
        *,
        domain: tuple[float | int | bool, float | int | bool] | None = ...,
        urn_factor: float | int | bool = ...,
        random_state: ToRNG = ...,
    ) -> None: ...

class DiscreteGuideTable(_PPFMethodMixin, Method):
    def __init__(
        self,
        /,
        dist: onp.ToFloat | onp.ToFloatND | _HasPMF,
        *,
        domain: tuple[float | int | bool, float | int | bool] | None = ...,
        guide_factor: float | int | bool = ...,
        random_state: ToRNG = ...,
    ) -> None: ...
