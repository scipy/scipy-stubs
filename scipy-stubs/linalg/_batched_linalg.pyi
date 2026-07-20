# defined in scipy/linalg/src/_batched_linalg_module.cc

from typing import Literal, final, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

###

type _Int = np.int32 | np.int64
type _Float = np.float32 | np.float64
type _Complex = np.complex64 | np.complex128
type _Inexact = npc.inexact32 | npc.inexact64
type _Numeric = np.bool | npc.integer | _Inexact
type _ErrList = list[dict[str, float]]

###

@final
class error(Exception): ...  # undocumented

#
@overload
def _bandwidth(a: onp.Array2D[_Numeric], /) -> tuple[np.int64, np.int64]: ...
@overload
def _bandwidth(a: onp.Array[onp.AtLeast3D, _Numeric], /) -> tuple[onp.ArrayND[np.int64], onp.ArrayND[np.int64]]: ...
@overload
def _bandwidth(
    a: onp.ArrayND[_Numeric, tuple[int, ...]], /
) -> tuple[np.int64, np.int64] | tuple[onp.ArrayND[np.int64], onp.ArrayND[np.int64]]: ...

#
def _cholesky[ScalarT: _Inexact](
    a: onp.ArrayND[ScalarT], lower: bool, overwrite_a: bool, clean: bool, /
) -> tuple[onp.ArrayND[ScalarT], _ErrList]: ...

#
def _det[ScalarT: _Inexact](a: onp.ArrayND[ScalarT], overwrite_a: bool, /) -> onp.ArrayND[ScalarT]: ...

#
def _eig[ScalarT: _Inexact](
    a: onp.ArrayND[ScalarT],
    compute_vl: bool,
    compute_vr: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    b: onp.ArrayND[ScalarT] = ...,
    /,
) -> tuple[
    onp.ArrayND[_Complex], onp.ArrayND[ScalarT] | None, onp.ArrayND[_Complex] | None, onp.ArrayND[_Complex] | None, _ErrList
]: ...

#
def _inv[ScalarT: _Inexact](
    a: onp.ArrayND[ScalarT], structure: int, overwrite_a: bool, lower: bool, /
) -> tuple[onp.ArrayND[ScalarT], _ErrList]: ...

#
def _lstsq[ScalarT: _Inexact](
    a: onp.ArrayND[ScalarT],
    b: onp.ArrayND[_Inexact],
    rcond: float,
    lapack_driver: Literal["gelsd", "gelss", "gelsy"],
    overwrite_a: bool,
    overwrite_b: bool,
    /,
) -> tuple[onp.ArrayND[ScalarT], np.int64 | onp.ArrayND[np.int64], onp.ArrayND[_Float] | None, _ErrList]: ...

#
def _lu[ScalarT: _Inexact](
    a: onp.ArrayND[ScalarT], permute_l: bool, overwrite_a: bool, /
) -> tuple[onp.ArrayND[_Int], onp.ArrayND[ScalarT], onp.ArrayND[ScalarT]]: ...

#
def _qr[ScalarT: _Inexact](
    a: onp.ArrayND[ScalarT], overwrite_a: bool, mode: Literal[1, 11, 21, 31], pivoting: bool, /
) -> tuple[
    onp.ArrayND[ScalarT] | None, onp.ArrayND[ScalarT], onp.ArrayND[ScalarT] | None, onp.ArrayND[_Int] | None, _ErrList
]: ...

#
def _solve[ScalarT: _Inexact](
    a: onp.ArrayND[ScalarT],
    b: onp.ArrayND[_Inexact],
    structure: int,
    lower: bool,
    transposed: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    /,
) -> tuple[onp.ArrayND[ScalarT], _ErrList]: ...

#
@overload
def _svd[ScalarT: _Inexact](
    a: onp.ArrayND[ScalarT],
    lapack_driver: Literal["gesdd", "gesvd"],
    compute_uv: Literal[True],
    full_matrices: bool,
    overwrite_a: bool,
    /,
) -> tuple[onp.ArrayND[ScalarT], onp.ArrayND[_Float], onp.ArrayND[ScalarT], _ErrList]: ...
@overload
def _svd(
    a: onp.ArrayND[_Inexact],
    lapack_driver: Literal["gesdd", "gesvd"],
    compute_uv: Literal[False],
    full_matrices: bool,
    overwrite_a: bool,
    /,
) -> tuple[onp.ArrayND[_Float], _ErrList]: ...
