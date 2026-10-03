import types
from collections.abc import Callable, Mapping
from typing import Any, Final, Generic, Literal, SupportsIndex, final
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.sparse import csc_array, csc_matrix, csr_matrix

###

type _Int1D = onp.Array1D[np.int32]
type _Inexact2D = onp.Array2D[np.float32 | np.float64 | np.complex64 | np.complex128]

type _Trans = Literal["N", "T", "H"]

_InexactT_co = TypeVar("_InexactT_co", bound=np.float32 | np.float64 | np.complex64 | np.complex128, default=Any, covariant=True)

###

@final
class SuperLU(Generic[_InexactT_co]):
    shape: Final[tuple[int, int]]
    nnz: Final[int]
    perm_r: Final[onp.Array1D[np.int32]]
    perm_c: Final[onp.Array1D[np.int32]]
    L: csc_array[_InexactT_co]  # readonly
    U: csc_array[_InexactT_co]  # readonly

    @classmethod
    def __class_getitem__(cls, arg: object, /) -> types.GenericAlias: ...

    #
    def solve[ShapeT: tuple[int] | tuple[int, int]](
        self, /, rhs: onp.ArrayND[npc.number, ShapeT], trans: _Trans = "N"
    ) -> onp.ArrayND[_InexactT_co, ShapeT]: ...

def gssv(
    N: SupportsIndex,
    nnz: SupportsIndex,
    nzvals: _Inexact2D,
    colind: _Int1D,
    rowptr: _Int1D,
    B: _Inexact2D,
    csc: onp.ToBool = 0,
    options: Mapping[str, object] = ...,
) -> tuple[csc_matrix | csr_matrix, int]: ...

#
def gstrf(
    N: SupportsIndex,
    nnz: SupportsIndex,
    nzvals: _Inexact2D,
    colind: _Int1D,
    rowptr: _Int1D,
    csc_construct_func: type[csc_array] | Callable[..., csc_array],
    ilu: onp.ToBool = 0,
    options: Mapping[str, object] = ...,
) -> SuperLU: ...

#
def gstrs(
    trans: Literal["N", "T"],
    L_n: SupportsIndex,
    L_nnz: SupportsIndex,
    L_nzvals: _Inexact2D,
    L_rowind: _Int1D,
    L_colptr: _Int1D,
    U_n: SupportsIndex,
    U_nnz: SupportsIndex,
    U_nzvals: _Inexact2D,
    U_rowind: _Int1D,
    U_colptr: _Int1D,
    B: _Inexact2D,
) -> tuple[onp.ArrayND[np.float64 | np.complex128], int]: ...
