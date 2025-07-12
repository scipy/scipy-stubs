from typing import TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from ._realtransforms import dct, dctn, dst, dstn, idct, idst
from ._typing import DCTType, NormalizationMode
from scipy._typing import (
    AnyShape,
    CanArrayND,  # path-dependent Pyright bug workaround
)

__all__ = ["dct", "dctn", "dst", "dstn", "idct", "idctn", "idst", "idstn"]

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])
_DTypeT = TypeVar("_DTypeT", bound=np.dtype[np.float32 | np.float64 | np.longdouble | npc.complexfloating])

_ToIntOrND: TypeAlias = onp.ToInt | onp.ToIntND

###

#
@overload
def idctn(
    x: CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[np.float64, _ShapeT]: ...
@overload
def idctn(
    x: CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[np.float32, _ShapeT]: ...
@overload
def idctn(
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
def idctn(
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
def idctn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[npc.floating]: ...

#
@overload
def idstn(
    x: CanArrayND[npc.integer, _ShapeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[np.float64, _ShapeT]: ...
@overload
def idstn(
    x: CanArrayND[np.float16, _ShapeT],
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[np.float32, _ShapeT]: ...
@overload
def idstn(
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
def idstn(
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
def idstn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    s: _ToIntOrND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> onp.ArrayND[npc.floating]: ...
