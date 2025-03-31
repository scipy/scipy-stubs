from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Final, Generic, Literal, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import AnyShape, ToRNG
from ._distn_infrastructure import rv_frozen
from .qmc import QMCEngine

__all__ = ["FastGeneratorInversion", "RatioUniforms"]

_RT_co = TypeVar("_RT_co", covariant=True, bound=onp.ToFloat, default=onp.ToFloat)

###

PINV_CONFIG: Final[dict[str, dict[str, Callable[..., onp.ToFloat]]]]

class CustomDistPINV(Generic[_RT_co]):  # undocumented
    def __init__(self, /, pdf: Callable[Concatenate[onp.ToFloat, ...], _RT_co], args: Sequence[object]) -> None: ...
    def pdf(self, /, x: onp.ToFloat) -> _RT_co: ...

class RatioUniforms:
    def __init__(
        self,
        /,
        pdf: Callable[Concatenate[onp.ToFloat, ...], onp.ToFloat],
        *,
        umax: onp.ToFloat,
        vmin: onp.ToFloat,
        vmax: onp.ToFloat,
        c: onp.ToFloat = 0,
        random_state: ToRNG = None,
    ) -> None: ...
    def rvs(self, /, size: AnyShape = 1) -> np.float64 | onp.ArrayND[np.float64]: ...

class FastGeneratorInversion:
    def __init__(
        self,
        /,
        dist: rv_frozen[Any, float | int | bool | np.float64],
        *,
        domain: tuple[onp.ToFloat, onp.ToFloat] | None = None,
        ignore_shape_range: bool = False,
        random_state: ToRNG = None,
    ) -> None: ...
    @property
    def random_state(self, /) -> np.random.Generator: ...
    @random_state.setter
    def random_state(self, random_state: ToRNG, /) -> None: ...
    @property
    def loc(self, /) -> float | int | bool | np.float64: ...
    @loc.setter
    def loc(self, loc: onp.ToFloat, /) -> None: ...
    @property
    def scale(self, /) -> float | int | bool | np.float64: ...
    @scale.setter
    def scale(self, scale: onp.ToFloat, /) -> None: ...
    @overload
    def rvs(self, /, size: None = None) -> np.float64: ...
    @overload
    def rvs(self, /, size: AnyShape) -> onp.ArrayND[np.float64]: ...
    @overload
    def qrvs(
        self,
        /,
        size: tuple[Literal[1]] | None = None,
        d: int | bool | None = None,
        qmc_engine: QMCEngine | None = None,
    ) -> np.float64: ...
    @overload
    def qrvs(
        self,
        /,
        size: AnyShape,
        d: int | bool | None = None,
        qmc_engine: QMCEngine | None = None,
    ) -> np.float64 | onp.ArrayND[np.float64]: ...
    @overload
    def ppf(self, /, q: onp.ToFloat) -> np.float64: ...
    @overload
    def ppf(self, /, q: onp.ToFloatND) -> onp.ArrayND[np.float64]: ...
    def evaluate_error(
        self,
        /,
        size: int | bool = 100_000,
        random_state: ToRNG = None,
        x_error: bool = False,
    ) -> tuple[np.float64, np.float64]: ...
    def support(self, /) -> tuple[float | int | bool, float | int | bool] | tuple[np.float64, np.float64]: ...

def argus_pdf(x: onp.ToFloat, chi: onp.ToFloat) -> float | int | bool: ...  # undocumented
def argus_gamma_trf(x: onp.ToFloat, chi: onp.ToFloat) -> np.float64: ...  # undocumented
def argus_gamma_inv_trf(x: onp.ToFloat, chi: onp.ToFloat) -> onp.ToFloat: ...  # undocumented
def betaprime_pdf(x: onp.ToFloat, a: onp.ToFloat, b: onp.ToFloat) -> float | int | bool | np.float64: ...  # undocumented
def beta_valid_params(a: op.CanFloat, b: op.CanFloat) -> bool: ...  # undocumented
def gamma_pdf(x: onp.ToFloat, a: op.CanFloat) -> float | int | bool: ...  # undocumented
def invgamma_pdf(x: onp.ToFloat, a: op.CanFloat) -> float | int | bool: ...  # undocumented
def burr_pdf(x: onp.ToFloat, cc: op.CanFloat, dd: op.CanFloat) -> np.float64 | Literal[0]: ...  # undocumented
def burr12_pdf(x: onp.ToFloat, cc: onp.ToFloat, dd: onp.ToFloat) -> float | int | bool: ...  # undocumented
def chi_pdf(x: onp.ToFloat, a: onp.ToFloat) -> float | int | bool: ...  # undocumented
def chi2_pdf(x: onp.ToFloat, df: onp.ToFloat) -> float | int | bool: ...  # undocumented
def alpha_pdf(x: onp.ToFloat, a: onp.ToFloat) -> float | int | bool: ...  # undocumented
def bradford_pdf(x: onp.ToFloat, c: onp.ToFloat) -> onp.ToFloat: ...  # undocumented
def crystalball_pdf(x: onp.ToFloat, b: onp.ToFloat, m: onp.ToFloat) -> float | int | bool: ...  # undocumented
def weibull_min_pdf(x: onp.ToFloat, c: onp.ToFloat) -> onp.ToFloat: ...  # undocumented
def weibull_max_pdf(x: onp.ToFloat, c: onp.ToFloat) -> onp.ToFloat: ...  # undocumented
def invweibull_pdf(x: onp.ToFloat, c: onp.ToFloat) -> onp.ToFloat: ...  # undocumented
def wald_pdf(x: onp.ToFloat) -> float | int | bool: ...  # undocumented
def geninvgauss_mode(p: op.CanFloat, b: onp.ToFloat) -> onp.ToFloat: ...  # undocumented
def geninvgauss_pdf(x: onp.ToFloat, p: onp.ToFloat, b: onp.ToFloat) -> float | int | bool: ...  # undocumented
def invgauss_mode(mu: onp.ToFloat) -> float | int | bool: ...  # undocumented
def invgauss_pdf(x: onp.ToFloat, mu: onp.ToFloat) -> float | int | bool: ...  # undocumented
def powerlaw_pdf(x: onp.ToFloat, a: onp.ToFloat) -> onp.ToFloat: ...  # undocumented
