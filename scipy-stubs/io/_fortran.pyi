from typing import Literal, Self, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy._typing import ExitMixin
from scipy.io._typing import FileLike

__all__ = ["FortranEOFError", "FortranFile", "FortranFormattingError"]

###

type _FileModeRW = Literal["r", "w"]

###

class FortranEOFError(TypeError, OSError): ...
class FortranFormattingError(TypeError, OSError): ...

class FortranFile(ExitMixin):
    def __init__(
        self,
        /,
        filename: FileLike[bytes],
        mode: _FileModeRW = "r",
        header_dtype: onp.AnyDType = np.uint32,  # noqa: PYI011
    ) -> None: ...

    #
    def __enter__(self, /) -> Self: ...
    def close(self, /) -> None: ...

    #
    def write_record(self, /, *items: onp.ToArrayND) -> None: ...
    def read_record(self, /, *dtypes: onp.ToDType, dtype: onp.ToDType | None = None) -> onp.Array1D[np.void]: ...

    #
    @overload
    def read_ints(self, /) -> onp.Array1D[np.int32]: ...
    @overload
    def read_ints[IntegerT: npc.integer](self, /, dtype: onp.ToDType[IntegerT]) -> onp.Array1D[IntegerT]: ...
    @overload
    def read_ints(self, /, dtype: str | type) -> onp.Array1D: ...

    #
    @overload
    def read_reals(self, /) -> onp.Array1D[np.float64]: ...
    @overload
    def read_reals[FloatingT: npc.floating](self, /, dtype: onp.ToDType[FloatingT]) -> onp.Array1D[FloatingT]: ...
    @overload
    def read_reals(self, /, dtype: str | type) -> onp.Array1D: ...
