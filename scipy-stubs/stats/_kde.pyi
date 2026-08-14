from collections.abc import Callable
from types import GenericAlias
from typing import Any, Final, Generic, Literal, Self, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["gaussian_kde"]

###

type _ToFloatMax1D = onp.ToFloat | onp.ToFloat1D
type _ToFloatMax2D = _ToFloatMax1D | onp.ToFloat2D

type _BWMethod = Literal["scott", "silverman"] | onp.ToFloat | Callable[[gaussian_kde], onp.ToFloat]

_ScalarT = TypeVar("_ScalarT", bound=npc.number | np.bool)
_ScalarT_co = TypeVar("_ScalarT_co", bound=npc.number | np.bool, default=np.float64, covariant=True)

###

class gaussian_kde(Generic[_ScalarT_co]):
    dataset: onp.Array2D[_ScalarT_co]  # readonly
    covariance: Final[onp.Array2D[np.float64 | Any]]  # usually float64, sometimes longdouble
    factor: Final[np.float64]
    d: Final[int]
    n: Final[int]

    @classmethod
    def __class_getitem__(cls, arg: object | type, /) -> GenericAlias: ...

    #
    @property
    def weights(self, /) -> onp.Array1D[np.float64]: ...
    @property
    def inv_cov(self, /) -> onp.Array2D[np.float64]: ...
    @property
    def neff(self, /) -> np.float64: ...

    #
    @overload  # <known scalar-type>
    def __init__(
        self: gaussian_kde[_ScalarT],
        /,
        dataset: onp.ToArray1D[_ScalarT, _ScalarT] | onp.ToArray2D[_ScalarT, _ScalarT],
        bw_method: _BWMethod | None = None,
        weights: _ToFloatMax1D | None = None,
    ) -> None: ...
    @overload  # ~bool
    def __init__(
        self: gaussian_kde[np.bool],
        /,
        dataset: onp.ToJustBool1D | onp.ToJustBool2D,
        bw_method: _BWMethod | None = None,
        weights: _ToFloatMax1D | None = None,
    ) -> None: ...
    @overload  # ~int
    def __init__(
        self: gaussian_kde[np.int64],
        /,
        dataset: onp.ToJustInt64_1D | onp.ToJustInt64_2D,
        bw_method: _BWMethod | None = None,
        weights: _ToFloatMax1D | None = None,
    ) -> None: ...
    @overload  # ~float
    def __init__(
        self: gaussian_kde[np.float64],
        /,
        dataset: onp.ToJustFloat64_1D | onp.ToJustFloat64_2D,
        bw_method: _BWMethod | None = None,
        weights: _ToFloatMax1D | None = None,
    ) -> None: ...
    @overload  # fallback
    def __init__(
        self: gaussian_kde[np.float64 | Any],
        /,
        dataset: _ToFloatMax2D,
        bw_method: _BWMethod | None = None,
        weights: _ToFloatMax1D | None = None,
    ) -> None: ...

    #
    def __call__(self, /, points: _ToFloatMax2D) -> onp.Array1D[np.float64]: ...
    def evaluate(self, /, points: _ToFloatMax2D) -> onp.Array1D[np.float64]: ...
    def pdf(self, /, x: _ToFloatMax2D) -> onp.Array1D[np.float64]: ...
    def logpdf(self, /, x: _ToFloatMax2D) -> onp.Array1D[np.float64]: ...

    #
    def integrate_gaussian(self, /, mean: _ToFloatMax1D, cov: onp.ToFloat | onp.ToFloat2D) -> np.float64: ...
    def integrate_box_1d(self, /, low: onp.ToFloat, high: onp.ToFloat) -> np.float64: ...
    def integrate_box(
        self,
        /,
        low_bounds: _ToFloatMax1D,
        high_bounds: _ToFloatMax1D,
        maxpts: int | None = None,
        *,
        rng: onp.random.ToRNG | None = None,
    ) -> np.float64: ...
    def integrate_kde(self, /, other: Self) -> np.float64: ...

    #
    def resample(self, /, size: int | None = None, seed: onp.random.ToRNG | None = None) -> onp.Array2D[np.float64]: ...

    #
    def scotts_factor(self, /) -> np.float64: ...
    def silverman_factor(self, /) -> np.float64: ...
    def covariance_factor(self, /) -> np.float64: ...

    #
    def set_bandwidth(self, /, bw_method: _BWMethod | None = None) -> None: ...

    #
    def marginal(self, /, dimensions: onp.ToInt | onp.ToInt1D) -> Self: ...
