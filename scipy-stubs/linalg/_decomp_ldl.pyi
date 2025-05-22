from typing import Any, TypeAlias, overload

import numpy as np
import optype.numpy as onp

__all__ = ["ldl"]

_ISize1D: TypeAlias = onp.Array1D[np.intp]
_ISizeND: TypeAlias = onp.ArrayND[np.intp]
_Float2D: TypeAlias = onp.Array2D[np.floating[Any]]
_FloatND: TypeAlias = onp.ArrayND[np.floating[Any]]
_Complex2D: TypeAlias = onp.Array2D[np.complexfloating[Any]]
_ComplexND: TypeAlias = onp.ArrayND[np.complexfloating[Any]]
_InexactND: TypeAlias = onp.ArrayND[np.inexact[Any]]

###

@overload
def ldl(
    A: onp.ToFloatStrict2D,
    lower: onp.ToBool = True,
    hermitian: onp.ToBool = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Float2D, _Float2D, _ISize1D]: ...
@overload
def ldl(
    A: onp.ToFloatND,
    lower: onp.ToBool = True,
    hermitian: onp.ToBool = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> tuple[_FloatND, _FloatND, _ISizeND]: ...
@overload
def ldl(
    A: onp.ToJustComplexStrict2D,
    lower: onp.ToBool = True,
    hermitian: onp.ToBool = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Complex2D, _Complex2D, _ISize1D]: ...
@overload
def ldl(
    A: onp.ToJustComplexND,
    lower: onp.ToBool = True,
    hermitian: onp.ToBool = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> tuple[_ComplexND, _ComplexND, _ISizeND]: ...
@overload
def ldl(
    A: onp.ToComplexND,
    lower: onp.ToBool = True,
    hermitian: onp.ToBool = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> tuple[_InexactND, _InexactND, _ISizeND]: ...
