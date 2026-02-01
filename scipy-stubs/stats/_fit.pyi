from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Concatenate, Generic, Literal, Protocol, TypeAlias, TypeVar, overload, type_check_only

import numpy as np
import optype.numpy as onp

from ._distn_infrastructure import rv_continuous, rv_continuous_frozen, rv_discrete
from scipy.optimize import OptimizeResult

_Params: TypeAlias = Mapping[str, onp.ToFloat]
_Bounds: TypeAlias = Mapping[str, tuple[onp.ToFloat, onp.ToFloat]] | Sequence[tuple[onp.ToFloat, onp.ToFloat]]

_GOFStatName: TypeAlias = Literal["ad", "ks", "cvm", "filliben"]
_GOFStatFunc: TypeAlias = Callable[[rv_continuous_frozen, onp.ArrayND[np.float64]], float | np.float32 | np.float64]
_FitMethod: TypeAlias = Literal["mle", "mse"]
_PlotType: TypeAlias = Literal["hist", "qq", "pp", "cdf"]

# Define fresh local TypeVars to ensure they are "Free" for the Generic class
_ST = TypeVar("_ST", bound=onp.ArrayND[np.float64])
_DT = TypeVar("_DT", bound=onp.ArrayND[np.float64])

# TODO(jorenham): make more specific
_Optimizer: TypeAlias = Callable[Concatenate[Callable[..., Any], ...], OptimizeResult]

@type_check_only
class _PXF1n(Protocol):
    def __call__(self, x: onp.ToFloat, arg0: onp.ToFloat, /, *args: onp.ToFloat) -> np.float64: ...

@type_check_only
class _PXF2n(Protocol):
    def __call__(self, x: onp.ToFloat, arg0: onp.ToFloat, arg1: onp.ToFloat, /, *args: onp.ToFloat) -> np.float64: ...

_PXFT_co = TypeVar("_PXFT_co", bound=Callable[Concatenate[onp.ToFloat, ...], np.float64], default=_PXF1n | _PXF2n, covariant=True)
_AxesT = TypeVar("_AxesT", default=Any)

###

class FitResult(Generic[_PXFT_co]):
    pxf: _PXFT_co
    params: tuple[onp.ToFloat, *tuple[onp.ToFloat, ...]]
    success: bool | None
    message: str | None
    discrete: bool

    @overload
    def __init__(
        self: FitResult[_PXF1n], /, dist: rv_discrete, data: onp.ToFloatND, discrete: bool, res: OptimizeResult
    ) -> None: ...
    @overload
    def __init__(
        self: FitResult[_PXF2n], /, dist: rv_continuous, data: onp.ToFloatND, discrete: bool, res: OptimizeResult
    ) -> None: ...
    def nllf(self, /, params: tuple[onp.ToFloat, ...] | None = None, data: onp.ToFloatND | None = None) -> np.float64: ...
    def plot(self, /, ax: _AxesT | None = None, *, plot_type: _PlotType = "hist") -> _AxesT: ...

@dataclass
class GoodnessOfFitResult(Generic[_ST, _DT]):
    fit_result: FitResult[_PXF2n]
    statistic: _ST
    pvalue: _ST
    null_distribution: _DT

@overload
def fit(
    dist: rv_discrete,
    data: onp.Array1D[np.float64],  # Strict 1D Array
    bounds: _Bounds | None = None,
    *,
    guess: _Params | onp.ToFloat1D | None = None,
    method: _FitMethod = "mle",
    optimizer: _Optimizer = ...,
) -> FitResult[_PXF1n]: ...
@overload
def fit(
    dist: rv_continuous,
    data: onp.ArrayND[np.float64],  # Strict ND Array (2D, 3D, etc)
    bounds: _Bounds | None = None,
    *,
    guess: _Params | onp.ToFloat1D | None = None,
    method: _FitMethod = "mle",
    optimizer: _Optimizer = ...,
) -> FitResult[_PXF2n]: ...
@overload
def goodness_of_fit(
    dist: rv_continuous,
    data: onp.Array1D[np.float64],  # Strict 1D Array
    *,
    known_params: _Params | None = None,
    fit_params: _Params | None = None,
    guessed_params: _Params | None = None,
    statistic: _GOFStatName | _GOFStatFunc = "ad",
    n_mc_samples: onp.ToJustInt = 9_999,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> GoodnessOfFitResult[np.float64, onp.Array1D[np.float64]]: ...
@overload
def goodness_of_fit(
    dist: rv_continuous,
    data: onp.ArrayND[np.float64],  # Strict ND Array
    *,
    known_params: _Params | None = None,
    fit_params: _Params | None = None,
    guessed_params: _Params | None = None,
    statistic: _GOFStatName | _GOFStatFunc = "ad",
    n_mc_samples: onp.ToJustInt = 9_999,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> GoodnessOfFitResult[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]: ...
# Force git update
# Force git sync
