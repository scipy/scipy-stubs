from collections.abc import Sequence
from typing import Any, Literal, Never, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc
import optype.typing as opt

__all__ = ["cossin"]

###

type _Tuple2[T] = tuple[T, T]
type _ToBlocks[T] = T | Sequence[T]

# workaround for https://github.com/microsoft/pyright/issues/10232
type _JustAnyShape = tuple[Never, Never, Never, Never]

###

@overload  # +f64, ?d
def cossin(
    X: onp.ArrayND[npc.floating | npc.integer | np.bool, _JustAnyShape],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    separate: Literal[False] = False,
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any]]: ...
@overload  # +f64, 2d
def cossin(
    X: _ToBlocks[onp.ToFloatStrict2D],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    separate: Literal[False] = False,
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> tuple[onp.Array2D[np.float64 | Any], onp.Array2D[np.float64 | Any], onp.Array2D[np.float64 | Any]]: ...
@overload  # +f64, nd
def cossin(
    X: _ToBlocks[onp.ToFloatND],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    separate: Literal[False] = False,
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any]]: ...
@overload  # +f64, ?d, separate=True
def cossin(
    X: onp.ArrayND[npc.floating | npc.integer | np.bool, _JustAnyShape],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    *,
    separate: Literal[True],
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.float64 | Any]], onp.ArrayND[np.float64 | Any], _Tuple2[onp.ArrayND[np.float64 | Any]]]: ...
@overload  # +f64, 2d, separate=True
def cossin(
    X: _ToBlocks[onp.ToFloatStrict2D],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    *,
    separate: Literal[True],
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> tuple[_Tuple2[onp.Array2D[np.float64 | Any]], onp.Array1D[np.float64 | Any], _Tuple2[onp.Array2D[np.float64 | Any]]]: ...
@overload  # +f64, nd, separate=True
def cossin(
    X: _ToBlocks[onp.ToFloatND],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    *,
    separate: Literal[True],
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.float64 | Any]], onp.ArrayND[np.float64 | Any], _Tuple2[onp.ArrayND[np.float64 | Any]]]: ...
@overload  # ~c128, ?d
def cossin(
    X: onp.ArrayND[npc.complexfloating, _JustAnyShape],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    separate: Literal[False] = False,
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> tuple[onp.ArrayND[np.complex128 | Any], onp.ArrayND[np.float64 | Any], onp.ArrayND[np.complex128 | Any]]: ...
@overload  # ~c128, 2d
def cossin(
    X: _ToBlocks[onp.ToJustComplexStrict2D],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    separate: Literal[False] = False,
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> tuple[onp.Array2D[np.complex128 | Any], onp.Array2D[np.float64 | Any], onp.Array2D[np.complex128 | Any]]: ...
@overload  # ~c128, nd
def cossin(
    X: _ToBlocks[onp.ToJustComplexND],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    separate: Literal[False] = False,
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> tuple[onp.ArrayND[np.complex128 | Any], onp.ArrayND[np.float64 | Any], onp.ArrayND[np.complex128 | Any]]: ...
@overload  # ~c128, ?d, separate=True
def cossin(
    X: onp.ArrayND[npc.complexfloating, _JustAnyShape],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    *,
    separate: Literal[True],
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> tuple[
    _Tuple2[onp.ArrayND[np.complex128 | Any]], onp.ArrayND[np.float64 | Any], _Tuple2[onp.ArrayND[np.complex128 | Any]]
]: ...
@overload  # ~c128, 2d, separate=True
def cossin(
    X: _ToBlocks[onp.ToJustComplexStrict2D],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    *,
    separate: Literal[True],
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> tuple[
    _Tuple2[onp.Array2D[np.complex128 | Any]], onp.Array1D[np.float64 | Any], _Tuple2[onp.Array2D[np.complex128 | Any]]
]: ...
@overload  # ~c128, nd, separate=True
def cossin(
    X: _ToBlocks[onp.ToJustComplexND],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    *,
    separate: Literal[True],
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> tuple[
    _Tuple2[onp.ArrayND[np.complex128 | Any]], onp.ArrayND[np.float64 | Any], _Tuple2[onp.ArrayND[np.complex128 | Any]]
]: ...
