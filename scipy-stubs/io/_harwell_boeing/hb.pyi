from typing import IO, Any, Final, Literal, TypeAlias, overload, type_check_only
from typing_extensions import LiteralString, Protocol, Self

import numpy as np
import optype.typing as opt
from scipy._typing import Falsy, FileLike, Truthy
from scipy.sparse import csc_array, csc_matrix
from scipy.sparse._base import _spbase

__all__ = ["hb_read", "hb_write"]

_ValueType: TypeAlias = Literal["real", "complex | float | int | bool", "pattern", "integer"]
_Structure: TypeAlias = Literal["symmetric", "unsymmetric", "hermitian", "skewsymmetric", "rectangular"]
_Storage: TypeAlias = Literal["assembled", "elemental"]

_Real: TypeAlias = np.integer[Any] | np.float32 | np.float64

@type_check_only
class _HasWidthAndRepeat(Protocol):
    @property
    def width(self, /) -> int | bool: ...
    @property
    def repeat(self, /) -> int | bool | None: ...

###

class MalformedHeader(Exception): ...
class LineOverflow(Warning): ...

class HBInfo:
    title: Final[str]
    key: Final[str]
    total_nlines: Final[int | bool]
    pointer_nlines: Final[int | bool]
    indices_nlines: Final[int | bool]
    values_nlines: Final[int | bool]
    pointer_format: Final[int | bool]
    indices_format: Final[int | bool]
    values_format: Final[int | bool]
    pointer_dtype: Final[int | bool]
    indices_dtype: Final[int | bool]
    values_dtype: Final[int | bool]
    pointer_nbytes_full: Final[int | bool]
    indices_nbytes_full: Final[int | bool]
    values_nbytes_full: Final[int | bool]
    nrows: Final[int | bool]
    ncols: Final[int | bool]
    nnon_zeros: Final[int | bool]
    nelementals: Final[int | bool]
    mxtype: HBMatrixType

    @classmethod
    def from_data(
        cls,
        m: _spbase,
        title: str = "Default title",
        key: str = "0",
        mxtype: HBMatrixType | None = None,
        fmt: None = None,
    ) -> Self: ...
    @classmethod
    def from_file(cls, fid: IO[str]) -> Self: ...

    #
    def __init__(
        self,
        /,
        title: str,
        key: str,
        total_nlines: int | bool,
        pointer_nlines: int | bool,
        indices_nlines: int | bool,
        values_nlines: int | bool,
        mxtype: HBMatrixType,
        nrows: int | bool,
        ncols: int | bool,
        nnon_zeros: int | bool,
        pointer_format_str: str,
        indices_format_str: str,
        values_format_str: str,
        right_hand_sides_nlines: int | bool = 0,
        nelementals: int | bool = 0,
    ) -> None: ...
    def dump(self, /) -> str: ...

class HBMatrixType:
    value_type: Final[_ValueType]
    structure: Final[_Structure]
    storage: Final[_Storage]

    @property
    def fortran_format(self, /) -> LiteralString: ...
    @classmethod
    def from_fortran(cls, fmt: str) -> Self: ...

    #
    def __init__(self, /, value_type: _ValueType, structure: _Structure, storage: _Storage = "assembled") -> None: ...

class HBFile:
    @property
    def title(self, /) -> str: ...
    @property
    def key(self, /) -> str: ...
    @property
    def type(self, /) -> _ValueType: ...
    @property
    def structure(self, /) -> _Structure: ...
    @property
    def storage(self, /) -> _Storage: ...

    #
    def __init__(self, /, file: IO[str], hb_info: HBMatrixType | None = None) -> None: ...
    def read_matrix(self, /) -> csc_array[_Real]: ...
    def write_matrix(self, /, m: _spbase) -> None: ...

def _nbytes_full(fmt: _HasWidthAndRepeat, nlines: int | bool) -> int | bool: ...
def _expect_int(value: opt.AnyInt, msg: str | None = None) -> int | bool: ...
def _read_hb_data(content: IO[str], header: HBInfo) -> csc_array[_Real]: ...
def _write_data(m: _spbase, fid: IO[str], header: HBInfo) -> None: ...

#
@overload
def hb_read(path_or_open_file: FileLike[str], *, spmatrix: Truthy = True) -> csc_array[_Real]: ...
@overload
def hb_read(path_or_open_file: FileLike[str], *, spmatrix: Falsy) -> csc_matrix[_Real]: ...

#
def hb_write(path_or_open_file: FileLike[str], m: _spbase, hb_info: HBInfo | None = None) -> None: ...
