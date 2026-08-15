# turned off for this file: mypy wrongly flags these disjoint dtype tiers as overlapping
# mypy: disable-error-code="overload-overlap"
from typing import Any, Literal, overload

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["polar"]

###

type _Tuple2[T] = tuple[T, T]
type _Side = Literal["left", "right"]

type _ASF64 = npc.floating64 | npc.floating80 | npc.integer64 | npc.integer32
type _AsC128 = npc.complexfloating128 | npc.complexfloating160

###

@overload  # 2d ~f32
def polar(a: onp.ToFloat32Strict2D, side: _Side = "right") -> _Tuple2[onp.Array2D[np.float32]]: ...
@overload  # nd ~f32
def polar(a: onp.ToFloat32_ND, side: _Side = "right") -> _Tuple2[onp.ArrayND[np.float32]]: ...
@overload  # 2d +f64
def polar(a: onp.ToArrayStrict2D[float, _ASF64], side: _Side = "right") -> _Tuple2[onp.Array2D[np.float64]]: ...
@overload  # nd +f64
def polar(a: onp.ToArrayND[float, _ASF64], side: _Side = "right") -> _Tuple2[onp.ArrayND[np.float64]]: ...
@overload  # 2d ~c64
def polar(a: onp.ToJustComplex64Strict2D, side: _Side = "right") -> _Tuple2[onp.Array2D[np.complex64]]: ...
@overload  # nd ~c64
def polar(a: onp.ToJustComplex64_ND, side: _Side = "right") -> _Tuple2[onp.ArrayND[np.complex64]]: ...
@overload  # 2d ~c128
def polar(a: onp.ToArrayStrict2D[op.JustComplex, _AsC128], side: _Side = "right") -> _Tuple2[onp.Array2D[np.complex128]]: ...
@overload  # nd ~c128
def polar(a: onp.ToArrayND[op.JustComplex, _AsC128], side: _Side = "right") -> _Tuple2[onp.ArrayND[np.complex128]]: ...
@overload  # nd +complex
def polar(a: onp.ToComplexND, side: _Side = "right") -> _Tuple2[onp.ArrayND[Any]]: ...
