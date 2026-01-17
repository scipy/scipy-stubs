import io
from typing import Any, Final, Literal, TypeAlias, overload, type_check_only
from typing_extensions import TypedDict, Unpack, override

import optype.numpy as onp

from scipy.io._typing import FileLike
from scipy.sparse import coo_array, coo_matrix, sparray, spmatrix

__all__ = ["mminfo", "mmread", "mmwrite"]

_Format: TypeAlias = Literal["coordinate", "array"]
_Field: TypeAlias = Literal["real", "complex", "pattern", "integer"]
_Symmetry: TypeAlias = Literal["general", "symmetric", "skew-symmetric", "hermitian"]
_Info: TypeAlias = tuple[int, int, int, _Format, _Field, _Symmetry]

@type_check_only
class _TextToBytesWrapperKwargs(TypedDict, total=False):
    buffer_size: int

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
    def read(self, /, size: int | None = -1) -> bytes: ...
    @override
    def read1(self, /, size: int = -1) -> bytes: ...
    @override
    def peek(self, /, size: int = -1) -> bytes: ...
    @override
    def seek(self, /, offset: int, whence: int = 0) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]

#
@overload
def mmread(source: FileLike[bytes], *, spmatrix: onp.ToTrue = True) -> onp.Array2D | coo_matrix: ...
@overload
def mmread(source: FileLike[bytes], *, spmatrix: onp.ToFalse) -> onp.Array2D | coo_array[Any, tuple[int, int]]: ...

#
def mmwrite(
    target: FileLike[bytes],
    a: spmatrix | sparray | onp.ToArrayND,
    comment: str = "",
    field: _Field | None = None,
    precision: int | None = None,
    symmetry: _Symmetry | None = None,
) -> None: ...

#
def mminfo(source: FileLike[bytes]) -> _Info: ...
