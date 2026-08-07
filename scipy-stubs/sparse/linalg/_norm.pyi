from typing import Any, Literal, SupportsIndex, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.sparse._base import _spbase

__all__ = ["norm"]

###

type _Ord = Literal["fro", 0, 1, 2, -1] | float

type _AsInexact64 = npc.inexact64 | npc.integer | np.bool
type _ToInexact64 = _AsInexact64 | npc.floating32

type _Axis2 = tuple[SupportsIndex, SupportsIndex]

###

@overload  # +f64 | ~c128, ord=None|"fro", axis=None|2d
def norm(x: _spbase[_AsInexact64], ord: Literal["fro"] | None = None, axis: _Axis2 | None = None) -> np.float64: ...
@overload  # ~f32 | ~c64, ord=None|"fro", axis=None|2d
def norm(x: _spbase[npc.inexact32], ord: Literal["fro"] | None = None, axis: _Axis2 | None = None) -> np.float32: ...
@overload  # +f64 | ~c128, axis=None|2d
def norm(x: _spbase[_AsInexact64], ord: _Ord | None = None, axis: _Axis2 | None = None) -> np.float64 | Any: ...
@overload  # ~c64, axis=None|2d
def norm(x: _spbase[npc.complexfloating64], ord: _Ord | None = None, axis: _Axis2 | None = None) -> np.float32 | Any: ...
@overload  # fallback, axis=None|2d
def norm(x: _spbase, ord: _Ord | None = None, axis: _Axis2 | None = None) -> np.float64 | Any: ...
@overload  # +f32 | +f64 | ~c128, ord=None, axis: int (positional)
def norm(x: _spbase[_ToInexact64], ord: None, axis: SupportsIndex) -> onp.Array1D[np.float64]: ...
@overload  # ~c64, ord=None, axis: int (positional)
def norm(x: _spbase[npc.complexfloating64], ord: None, axis: SupportsIndex) -> onp.Array1D[np.float32]: ...
@overload  # +f32 | +f64 | ~c128, axis: int (positional)
def norm(x: _spbase[_ToInexact64], ord: _Ord | None, axis: SupportsIndex) -> onp.Array1D[np.float64 | Any]: ...
@overload  # ~c64, axis: int (positional)
def norm(x: _spbase[npc.complexfloating64], ord: _Ord | None, axis: SupportsIndex) -> onp.Array1D[np.float32 | Any]: ...
@overload  # fallback, axis: int (positional)
def norm(x: _spbase, ord: _Ord | None, axis: SupportsIndex) -> onp.Array1D[np.float64 | Any]: ...
@overload  # +f32 | +f64 | ~c128, ord=None, axis: int (keyword)
def norm(x: _spbase[_ToInexact64], ord: None = None, *, axis: SupportsIndex) -> onp.Array1D[np.float64]: ...
@overload  # ~c64, ord=None, axis: int (keyword)
def norm(x: _spbase[npc.complexfloating64], ord: None = None, *, axis: SupportsIndex) -> onp.Array1D[np.float32]: ...
@overload  # +f32 | +f64 | ~c128, axis: int (keyword)
def norm(x: _spbase[_ToInexact64], ord: _Ord | None = None, *, axis: SupportsIndex) -> onp.Array1D[np.float64 | Any]: ...
@overload  # ~c64, axis: int (keyword)
def norm(x: _spbase[npc.complexfloating64], ord: _Ord | None = None, *, axis: SupportsIndex) -> onp.Array1D[np.float32 | Any]: ...
@overload  # fallback, axis: int (keyword)
def norm(x: _spbase, ord: _Ord | None = None, *, axis: SupportsIndex) -> onp.Array1D[np.float64 | Any]: ...
