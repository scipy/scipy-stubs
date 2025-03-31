from collections.abc import Sequence
from typing import Any, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from numpy._typing import _ArrayLike, _NestedSequence
from scipy._typing import AnyBool, Falsy, NanPolicy, Truthy

_SCT_fc = TypeVar("_SCT_fc", bound=np.inexact[Any])

###

# NOTE: This demonstrates the ridiculous complexity that's required to properly annotate this simple function with "array-likes".
# NOTE: Shape-typing hasn't even been included, as that would require even more overloads.

# sequences of `builtins.float | int | bool`, that implicitly (and inevitably) also cover `builtins.int | bool` and `builtins.bool`
@overload
def variation(
    a: Sequence[float | int | bool],
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: onp.ToInt = 0,
    *,
    keepdims: Falsy = False,
) -> np.float64: ...
@overload
def variation(
    a: onp.ToIntND | _NestedSequence[float | int | bool],
    axis: None,
    nan_policy: NanPolicy = "propagate",
    ddof: onp.ToInt = 0,
    *,
    keepdims: Falsy = False,
) -> np.float64: ...
@overload
def variation(
    a: onp.ToIntND | _NestedSequence[float | int | bool],
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: onp.ToInt = 0,
    *,
    keepdims: Truthy,
) -> onp.ArrayND[np.float64]: ...
@overload
def variation(
    a: onp.ToIntND | _NestedSequence[float | int | bool],
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: onp.ToInt = 0,
    *,
    keepdims: AnyBool = False,
) -> np.float64 | onp.ArrayND[np.float64]: ...

# array-like input with known floating- or complex | float | int | bool-floating dtypes
@overload
def variation(
    a: _ArrayLike[_SCT_fc],
    axis: None,
    nan_policy: NanPolicy = "propagate",
    ddof: onp.ToInt = 0,
    *,
    keepdims: Falsy = False,
) -> _SCT_fc: ...
@overload
def variation(
    a: _ArrayLike[_SCT_fc],
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: onp.ToInt = 0,
    *,
    keepdims: Truthy,
) -> onp.ArrayND[_SCT_fc]: ...
@overload
def variation(
    a: _ArrayLike[_SCT_fc],
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: onp.ToInt = 0,
    *,
    keepdims: AnyBool = False,
) -> _SCT_fc | onp.ArrayND[_SCT_fc]: ...

# sequences of `builtin.complex | float | int | bool`, which behave as if `float | int | bool <: complex | float | int | bool` and therefore "overlaps" with the `builtins.float | int | bool`
# overloads, hence the `complex128 | float64` returns
@overload
def variation(
    a: Sequence[complex | float | int | bool],
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: onp.ToInt = 0,
    *,
    keepdims: Falsy = False,
) -> np.complex128 | np.float64: ...
@overload
def variation(
    a: _NestedSequence[complex | float | int | bool],
    axis: None,
    nan_policy: NanPolicy = "propagate",
    ddof: onp.ToInt = 0,
    *,
    keepdims: Falsy = False,
) -> np.complex128 | np.float64: ...
@overload
def variation(
    a: _NestedSequence[complex | float | int | bool],
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: onp.ToInt = 0,
    *,
    keepdims: Truthy,
) -> onp.ArrayND[np.complex128 | np.float64]: ...
@overload
def variation(
    a: _NestedSequence[complex | float | int | bool],
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: onp.ToInt = 0,
    *,
    keepdims: AnyBool = False,
) -> np.complex128 | np.float64 | onp.ArrayND[np.complex128 | np.float64]: ...

# catch-all in case of broad gradual types
@overload
def variation(
    a: onp.ToComplexND,
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: onp.ToInt = 0,
    *,
    keepdims: AnyBool = False,
) -> np.inexact[Any] | onp.ArrayND[np.inexact[Any]]: ...
