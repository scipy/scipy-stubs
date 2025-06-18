from typing import TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy._typing import AnyShape, DCTType, NormalizationMode

__all__ = ["dct", "dctn", "dst", "dstn", "idct", "idctn", "idst", "idstn"]

_DTypeT = TypeVar("_DTypeT", bound=np.dtype[np.float32 | np.float64 | np.longdouble | npc.complexfloating])
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

_ToIntOrND: TypeAlias = onp.ToInt | onp.ToIntND
_FloatND: TypeAlias = onp.ArrayND[np.float32 | np.float64 | np.longdouble]  # doesn't include `numpy.float16`

###

# NOTE: These have (almost) identical signatures, so be sure to keep them in sync.

@overload
def dctn(
    x: onp.CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def dctn(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def dctn(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def dctn(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload
def dctn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
@overload
def idctn(
    x: onp.CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def idctn(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def idctn(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def idctn(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload
def idctn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
@overload
def dstn(
    x: onp.CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def dstn(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def dstn(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def dstn(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload
def dstn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
@overload
def idstn(
    x: onp.CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def idstn(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def idstn(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def idstn(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload
def idstn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
@overload
def dct(
    x: onp.CanArrayND[np.integer, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def dct(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def dct(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def dct(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload
def dct(
    x: onp.ToFloatND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
@overload
def idct(
    x: onp.CanArrayND[np.integer, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def idct(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def idct(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def idct(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload
def idct(
    x: onp.ToFloatND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
@overload
def dst(
    x: onp.CanArrayND[np.integer, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def dst(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def dst(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def dst(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload
def dst(
    x: onp.ToFloatND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
@overload
def idst(
    x: onp.CanArrayND[np.integer, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def idst(
    x: onp.CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def idst(
    x: onp.CanArray[_ShapeT, _DTypeT],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> np.ndarray[_ShapeT, _DTypeT]: ...
@overload
def idst(
    x: onp.SequenceND[float],
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload
def idst(
    x: onp.ToFloatND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...
