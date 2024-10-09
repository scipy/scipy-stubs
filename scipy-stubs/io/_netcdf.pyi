from collections.abc import Sequence
from types import TracebackType
from typing import IO, Final
from typing_extensions import Self

import numpy.typing as npt
from scipy._typing import FileLike, FileModeRWA, Untyped

__all__ = ["netcdf_file", "netcdf_variable"]

IS_PYPY: Final[bool] = ...

ABSENT: Final[bytes] = ...
ZERO: Final[bytes] = ...
NC_BYTE: Final[bytes] = ...
NC_CHAR: Final[bytes] = ...
NC_SHORT: Final[bytes] = ...
NC_INT: Final[bytes] = ...
NC_FLOAT: Final[bytes] = ...
NC_DOUBLE: Final[bytes] = ...
NC_DIMENSION: Final[bytes] = ...
NC_VARIABLE: Final[bytes] = ...
NC_ATTRIBUTE: Final[bytes] = ...
FILL_BYTE: Final[bytes] = ...
FILL_CHAR: Final[bytes] = ...
FILL_SHORT: Final[bytes] = ...
FILL_INT: Final[bytes] = ...
FILL_FLOAT: Final[bytes] = ...
FILL_DOUBLE: Final[bytes] = ...

TYPEMAP: Final[dict[bytes, tuple[str, int]]]
FILLMAP: Final[dict[bytes, bytes]]
REVERSE: Final[dict[tuple[str, int], bytes]]

class netcdf_file:
    fp: IO[bytes]
    filename: str
    use_mmap: bool
    mode: FileModeRWA
    version_byte: int
    maskandscale: Untyped
    dimensions: dict[str, int]
    variables: dict[str, netcdf_variable]
    def __init__(
        self,
        /,
        filename: FileLike[bytes],
        mode: FileModeRWA = "r",
        mmap: bool | None = None,
        version: int = 1,
        maskandscale: bool = False,
    ) -> None: ...
    def __del__(self, /) -> None: ...
    def __enter__(self, /) -> Self: ...
    def __exit__(
        self,
        /,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def close(self, /) -> None: ...
    def createDimension(self, /, name: str, length: int) -> None: ...
    def createVariable(self, /, name: str, type: npt.DTypeLike, dimensions: Sequence[str]) -> netcdf_variable: ...
    def flush(self, /) -> None: ...
    def sync(self, /) -> None: ...

class netcdf_variable:
    data: npt.ArrayLike
    dimensions: Sequence[str]
    maskandscale: bool
    @property
    def isrec(self, /) -> Untyped: ...
    @property
    def shape(self, /) -> Untyped: ...
    def __init__(
        self,
        /,
        data: npt.ArrayLike,
        typecode: str,
        size: int,
        shape: Sequence[int],
        dimensions: Sequence[str],
        attributes: dict[str, object] | None = None,
        maskandscale: bool = False,
    ) -> None: ...
    def __getitem__(self, /, index: object) -> object: ...
    def __setitem__(self, /, index: object, data: npt.ArrayLike) -> None: ...
    def getValue(self, /) -> object: ...
    def assignValue(self, /, value: object) -> None: ...
    def typecode(self, /) -> str: ...
    def itemsize(self, /) -> int: ...

NetCDFFile = netcdf_file
NetCDFVariable = netcdf_variable
