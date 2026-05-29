from collections.abc import Mapping
from typing import Any, Literal, TypedDict, Unpack, overload, type_check_only
from typing_extensions import deprecated

import optype as op

from ._miobase import MatFileReader
from scipy.io._typing import ByteOrder, FileName
from scipy.sparse import coo_array, coo_matrix, csc_array, csc_matrix

__all__ = ["loadmat", "savemat", "whosmat"]

type _NoValueType = op.JustObject
type _DataClass = Literal[
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "single",
    "double",
    "cell",
    "struct",
    "object",
    "char",
    "sparse",
    "function",
    "opaque",
    "logical",
    "unknown",
]

@type_check_only
class _ReaderKwargs(TypedDict, total=False):
    byte_order: ByteOrder | None
    mat_dtype: bool
    squeeze_me: bool
    chars_as_strings: bool
    matlab_compatible: bool
    struct_as_record: bool
    verify_compressed_data_integrity: bool
    simplify_cells: bool
    variable_names: list[str] | tuple[str, ...] | None

###

def mat_reader_factory(
    file_name: FileName, appendmat: bool = True, **kwargs: Unpack[_ReaderKwargs]
) -> tuple[MatFileReader, bool]: ...

#
@overload
@deprecated("The default value for `spmatrix` is changing to False in v1.20.")
def loadmat(
    file_name: FileName,
    mdict: Mapping[str, object] | None = None,
    appendmat: bool = True,
    *,
    spmatrix: _NoValueType = ...,
    **kwargs: Unpack[_ReaderKwargs],
) -> dict[str, csc_matrix[Any] | coo_matrix[Any]]: ...
@overload
def loadmat(
    file_name: FileName,
    mdict: Mapping[str, object] | None = None,
    appendmat: bool = True,
    *,
    spmatrix: Literal[True],
    **kwargs: Unpack[_ReaderKwargs],
) -> dict[str, csc_matrix[Any] | coo_matrix[Any]]: ...
@overload
def loadmat(
    file_name: FileName,
    mdict: Mapping[str, object] | None = None,
    appendmat: bool = True,
    *,
    spmatrix: Literal[False],
    **kwargs: Unpack[_ReaderKwargs],
) -> dict[str, csc_array[Any] | coo_array[Any, tuple[int, int]]]: ...

#
def savemat(
    file_name: FileName,
    mdict: Mapping[str, object],
    appendmat: bool = True,
    format: Literal["5", "4"] = "5",
    long_field_names: bool = False,
    do_compression: bool = False,
    oned_as: Literal["row", "column"] = "row",
) -> None: ...

#
def whosmat(
    file_name: FileName, appendmat: bool = True, **kwargs: Unpack[_ReaderKwargs]
) -> list[tuple[str, tuple[int, ...], _DataClass]]: ...
