from typing import Literal, TypeAlias, overload

import numpy as np
import optype as op
import optype.numpy as onp

from scipy._typing import Falsy, NanPolicy, Truthy

_QuantileMethod: TypeAlias = Literal[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",  # default
    "median_unbiased",
    "normal_unbiased",
]

###

@overload  # this mypy error is a false positive
def quantile(  # type: ignore[overload-overlap]
    x: onp.ToFloatStrict1D,
    p: onp.ToJustFloat,
    *,
    method: _QuantileMethod = "linear",
    axis: Literal[-1, 0] | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: op.CanBool | None = None,
) -> np.float64: ...
@overload
def quantile(
    x: onp.ToFloatND,
    p: onp.ToJustFloat,
    *,
    method: _QuantileMethod = "linear",
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: Falsy | None = None,
) -> np.float64: ...
@overload
def quantile(
    x: onp.ToFloatND,
    p: onp.ToJustFloatND,
    *,
    method: _QuantileMethod = "linear",
    axis: op.CanIndex = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: op.CanBool | None = False,
) -> onp.ArrayND[np.float64]: ...
@overload
def quantile(
    x: onp.ToFloatND,
    p: onp.ToJustFloat | onp.ToJustFloatND,
    *,
    method: _QuantileMethod = "linear",
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: Truthy,
) -> onp.ArrayND[np.float64]: ...
