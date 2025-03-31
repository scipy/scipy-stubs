from types import ModuleType
from typing import Any, Literal, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
from numpy._typing import _ArrayLike
from scipy._typing import AnyShape

_SCT = TypeVar("_SCT", bound=np.inexact[Any])

def next_fast_len(target: op.CanIndex, real: op.CanBool = False) -> int | bool: ...
def prev_fast_len(target: op.CanIndex, real: op.CanBool = False) -> int | bool: ...

#
@overload  # xp: None -> np.fft.fftfreq
def fftfreq(
    n: int | bool | np.integer[Any],
    d: onp.ToFloat = 1.0,
    *,
    xp: None = None,
    device: Literal["cpu"] | None = None,
) -> onp.Array1D[np.float64]: ...
@overload  # xp: ModuleType -> xp.fft.fftfreq
def fftfreq(n: int | bool, d: float | int | bool = 1.0, *, xp: ModuleType, device: object | None = None) -> Any: ...  # noqa: ANN401

#
@overload  # np.fft.rfftfreq
def rfftfreq(
    n: int | bool | np.integer[Any],
    d: onp.ToFloat = 1.0,
    *,
    xp: None = None,
    device: Literal["cpu"] | None = None,
) -> onp.Array1D[np.float64]: ...
@overload  # xp.fft.fftfreq
def rfftfreq(n: int | bool, d: float | int | bool = 1.0, *, xp: ModuleType, device: object | None = None) -> Any: ...  # noqa: ANN401

#
@overload
def fftshift(x: onp.ToIntND | onp.SequenceND[float | int | bool], axes: AnyShape | None = None) -> onp.ArrayND[np.float64]: ...
@overload
def fftshift(
    x: onp.SequenceND[complex | float | int | bool], axes: AnyShape | None = None
) -> onp.ArrayND[np.complex128 | np.float64]: ...
@overload
def fftshift(x: _ArrayLike[_SCT], axes: AnyShape | None = None) -> onp.ArrayND[_SCT]: ...
@overload
def fftshift(x: onp.ToComplexND, axes: AnyShape | None = None) -> onp.ArrayND[np.inexact[Any]]: ...

#
@overload
def ifftshift(x: onp.ToIntND | onp.SequenceND[float | int | bool], axes: AnyShape | None = None) -> onp.ArrayND[np.float64]: ...
@overload
def ifftshift(
    x: onp.SequenceND[complex | float | int | bool], axes: AnyShape | None = None
) -> onp.ArrayND[np.complex128 | np.float64]: ...
@overload
def ifftshift(x: _ArrayLike[_SCT], axes: AnyShape | None = None) -> onp.ArrayND[_SCT]: ...
@overload
def ifftshift(x: onp.ToComplexND, axes: AnyShape | None = None) -> onp.ArrayND[np.inexact[Any]]: ...
