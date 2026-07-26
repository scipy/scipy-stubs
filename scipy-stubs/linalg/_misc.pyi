from typing import Any, Literal, Never, SupportsIndex, overload
from typing_extensions import deprecated

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc
from numpy.linalg import LinAlgError

__all__ = ["LinAlgError", "LinAlgWarning", "bandwidth", "norm"]

###

type _Inf = float
type _Order = Literal["fro", "nuc", 0, 1, -1, 2, -2] | _Inf
type _Axis = SupportsIndex | tuple[SupportsIndex, SupportsIndex]
type _SubScalar = npc.inexact64 | npc.integer
type _BoolF16Inexact80 = np.bool | np.float16 | npc.inexact80

type _ToIntND = onp.ToArrayND[Never, npc.integer]
type _AsInexact32ND = onp.ToArrayND[Never, npc.inexact32]
type _AsInexact64ND = onp.ToArrayND[complex, npc.inexact64]
type _AsBoolF16Inexact80ND = onp.ToArrayND[Never, _BoolF16Inexact80]

# workaround for a strange bug in pyright's overlapping overload detection with `numpy<2.1`
type _WorkaroundForPyright = tuple[int] | tuple[Any, ...]

###

class LinAlgWarning(RuntimeWarning): ...

# NOTE: false positives on numpy<2.1
# pyright: reportOverlappingOverload=false

# NOTE: the `inexact{32,64,80}` groups are disjoint
# mypy: disable-error-code=overload-overlap

@overload  # 0d +inexact64
def norm(
    a: complex | _SubScalar, ord: _Order | None = None, axis: None = None, keepdims: bool = False, check_finite: bool = True
) -> np.float64: ...
@overload  # 0d ~inexact32
def norm(
    a: npc.inexact32, ord: _Order | None = None, axis: None = None, keepdims: bool = False, check_finite: bool = True
) -> np.float32: ...
@overload  # 0d ~bool | ~f16 | ~f80 | ~c160
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def norm(
    a: _BoolF16Inexact80, ord: _Order | None = None, axis: None = None, keepdims: bool = False, check_finite: bool = True
) -> np.float64 | Any: ...
@overload  # Nd ~integer
def norm(
    a: _ToIntND, ord: _Order | None = None, axis: None = None, keepdims: Literal[False] = False, check_finite: bool = True
) -> np.float64: ...
@overload  # Nd +inexact64
def norm(
    a: _AsInexact64ND, ord: _Order | None = None, axis: None = None, keepdims: Literal[False] = False, check_finite: bool = True
) -> float: ...
@overload  # Nd ~inexact32
def norm(
    a: _AsInexact32ND, ord: _Order | None = None, axis: None = None, keepdims: Literal[False] = False, check_finite: bool = True
) -> float | np.float32: ...
@overload  # Nd ~bool | ~f16 | ~f80 | ~c160
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def norm(
    a: _AsBoolF16Inexact80ND,
    ord: _Order | None = None,
    axis: None = None,
    keepdims: Literal[False] = False,
    check_finite: bool = True,
) -> np.float64 | Any: ...
@overload  # Nd +inexact64, keepdims: True, shape known
def norm[ShapeT: tuple[int, ...]](
    a: onp.ArrayND[_SubScalar, ShapeT],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Literal[True],
    check_finite: bool = True,
) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd ~inexact32, keepdims: True, shape known
def norm[ShapeT: tuple[int, ...]](
    a: onp.ArrayND[npc.inexact32, ShapeT],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Literal[True],
    check_finite: bool = True,
) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # Nd ~bool | ~f16 | ~f80 | ~c160, keepdims: True, shape known
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def norm[ShapeT: tuple[int, ...]](
    a: onp.ArrayND[_BoolF16Inexact80, ShapeT],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Literal[True],
    check_finite: bool = True,
) -> onp.ArrayND[np.float64 | Any, ShapeT]: ...
@overload  # Nd +inexact64, keepdims: True
def norm(
    a: onp.ToArrayND[complex, _SubScalar],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Literal[True],
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # Nd ~inexact32, keepdims: True
def norm(
    a: _AsInexact32ND, ord: _Order | None = None, axis: _Axis | None = None, *, keepdims: Literal[True], check_finite: bool = True
) -> onp.ArrayND[np.float32]: ...
@overload  # Nd ~bool | ~f16 | ~f80 | ~c160, keepdims: True
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def norm(
    a: _AsBoolF16Inexact80ND,
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Literal[True],
    check_finite: bool = True,
) -> onp.ArrayND[np.float64 | Any]: ...
@overload  # catch-all, keepdims: True
def norm(
    a: onp.ToComplexND,
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Literal[True],
    check_finite: bool = True,
) -> onp.ArrayND[np.float64 | Any, _WorkaroundForPyright]: ...
@overload  # catch-all
def norm(
    a: onp.ToArrayND, ord: _Order | None = None, axis: _Axis | None = None, keepdims: bool = False, check_finite: bool = True
) -> np.float64 | Any: ...

#
def _datacopied(arr: onp.ArrayND[Any], original: onp.CanArrayND[Any]) -> bool: ...  # undocumented

#
@overload  # pyright workaround
def bandwidth(a: onp.ArrayND[npc.number, tuple[Never, Never, Never, Never]]) -> tuple[np.int64 | Any, np.int64 | Any]: ...
@overload
def bandwidth(a: onp.ToComplexStrict2D) -> tuple[np.int64, np.int64]: ...
@overload
def bandwidth(a: onp.ToComplexStrict3D) -> tuple[onp.Array1D[np.int64], onp.Array1D[np.int64]]: ...
@overload
def bandwidth(a: onp.ToComplexND) -> tuple[onp.ArrayND[np.int64] | Any, onp.ArrayND[np.int64] | Any]: ...
