from typing import Any, Literal, SupportsIndex, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy._typing import AnyShape
from scipy.fft._typing import DCTType

__all__ = ["dct", "dctn", "dst", "dstn", "idct", "idctn", "idst", "idstn"]

_DTypeT = TypeVar("_DTypeT", bound=np.dtype[np.float32 | np.float64 | npc.floating80 | npc.complexfloating])
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

_NormKind: TypeAlias = Literal["ortho"] | None

# workaround for a strange bug in pyright's overlapping overload detection with `numpy<2.1`
_WorkaroundForPyright: TypeAlias = tuple[int] | tuple[Any, ...]

_FloatND: TypeAlias = onp.ArrayND[np.float32 | np.float64 | np.longdouble, _WorkaroundForPyright]

###

@overload
def dctn(
    x: onp.CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def dctn(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def dctn(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def dctn(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.float64]: ...
@overload
def dctn(
    x: onp.SequenceND[list[complex]] | list[complex],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload
def dctn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> _FloatND: ...

#
@overload
def idctn(
    x: onp.CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def idctn(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def idctn(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def idctn(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.float64]: ...
@overload
def idctn(
    x: onp.SequenceND[list[complex]] | list[complex],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload
def idctn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> _FloatND: ...

#
@overload
def dstn(
    x: onp.CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def dstn(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def dstn(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def dstn(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.float64]: ...
@overload
def dstn(
    x: onp.SequenceND[list[complex]] | list[complex],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload
def dstn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> _FloatND: ...

#
@overload
def idstn(
    x: onp.CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def idstn(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def idstn(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def idstn(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.float64]: ...
@overload
def idstn(
    x: onp.SequenceND[list[complex]] | list[complex],
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload
def idstn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> _FloatND: ...

#
@overload
def dct(
    x: onp.CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def dct(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def dct(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def dct(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.float64]: ...
@overload
def dct(
    x: onp.SequenceND[list[complex]] | list[complex],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload
def dct(
    x: onp.ToFloatND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> _FloatND: ...

#
@overload
def idct(
    x: onp.CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def idct(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def idct(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def idct(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.float64]: ...
@overload
def idct(
    x: onp.SequenceND[list[complex]] | list[complex],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload
def idct(
    x: onp.ToFloatND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> _FloatND: ...

#
@overload
def dst(
    x: onp.CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def dst(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def dst(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def dst(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.float64]: ...
@overload
def dst(
    x: onp.SequenceND[list[complex]] | list[complex],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload
def dst(
    x: onp.ToFloatND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> _FloatND: ...

#
@overload
def idst(
    x: onp.CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def idst(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def idst(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def idst(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.float64]: ...
@overload
def idst(
    x: onp.SequenceND[list[complex]] | list[complex],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload
def idst(
    x: onp.ToFloatND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: SupportsIndex = -1,
    norm: _NormKind = None,
    overwrite_x: bool = False,
) -> _FloatND: ...
