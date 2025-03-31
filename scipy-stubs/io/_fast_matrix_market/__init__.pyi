import io
from typing import Any, Final, Literal, TypeAlias, overload, type_check_only
from typing_extensions import TypedDict, Unpack, override

import numpy as np
import optype.numpy as onp
from scipy._typing import Falsy, FileLike, FileName, Truthy
from scipy.sparse import coo_array, coo_matrix
from scipy.sparse._base import _spbase

__all__ = ["mminfo", "mmread", "mmwrite"]

_Format: TypeAlias = Literal["coordinate", "array"]
_Field: TypeAlias = Literal["real", "complex | float | int | bool", "pattern", "integer"]
_Symmetry: TypeAlias = Literal["general", "symmetric", "skew-symmetric", "hermitian"]

@type_check_only
class _TextToBytesWrapperKwargs(TypedDict, total=False):
    buffer_size: int | bool

###

PARALLELISM: Final = 0
ALWAYS_FIND_SYMMETRY: Final = False

class _TextToBytesWrapper(io.BufferedReader):
    encoding: Final[str]
    errors: Final[str]

    def __init__(
        self,
        /,
        text_io_buffer: io.TextIOBase,
        encoding: str | None = None,
        errors: str | None = None,
        **kwargs: Unpack[_TextToBytesWrapperKwargs],
    ) -> None: ...
    @override
    def read(self, /, size: int | bool | None = -1) -> bytes: ...
    @override
    def read1(self, /, size: int | bool = -1) -> bytes: ...
    @override
    def peek(self, /, size: int | bool = -1) -> bytes: ...
    @override
    def seek(self, /, offset: int | bool, whence: int | bool = 0) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

@overload
def mmread(source: FileLike[bytes], *, spmatrix: Truthy = True) -> onp.ArrayND[np.number[Any]] | coo_array: ...
@overload
def mmread(source: FileLike[bytes], *, spmatrix: Falsy) -> onp.ArrayND[np.number[Any]] | coo_matrix: ...

#
def mmwrite(
    target: FileName,
    a: onp.CanArray | list[object] | tuple[object, ...] | _spbase,
    comment: str | None = None,
    field: _Field | None = None,
    precision: int | bool | None = None,
    symmetry: _Symmetry | Literal["AUTO"] = "AUTO",
) -> None: ...

#
def mminfo(source: FileName) -> tuple[int | bool, int | bool, int | bool, _Format, _Field, _Symmetry]: ...
