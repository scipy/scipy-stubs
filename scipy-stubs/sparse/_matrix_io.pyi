from typing import Any, Final, Literal, Protocol, TypedDict, final, type_check_only

import optype as op

from ._bsr import bsr_array, bsr_matrix
from ._coo import coo_array, coo_matrix
from ._csc import csc_array, csc_matrix
from ._csr import csr_array, csr_matrix
from ._data import _data_matrix
from ._dia import dia_array, dia_matrix

__all__ = ["load_npz", "save_npz"]

@final
@type_check_only
class _CanReadAndSeekBytes(Protocol):
    def read(self, length: int = ..., /) -> bytes: ...
    def seek(self, offset: int, whence: int, /) -> object: ...

@final
@type_check_only
class _PickleKwargs(TypedDict):
    allow_pickle: Literal[False]

###

PICKLE_KWARGS: Final[_PickleKwargs] = ...

def load_npz(
    file: op.io.ToPath | _CanReadAndSeekBytes,
) -> (
    bsr_array[Any]
    | coo_array[Any]
    | csc_array[Any]
    | csr_array[Any, tuple[Any, ...]]
    | dia_array[Any]
    | bsr_matrix[Any]
    | coo_matrix[Any]
    | csc_matrix[Any]
    | csr_matrix[Any]
    | dia_matrix[Any]
): ...
def save_npz(file: op.io.ToPath[str] | op.io.CanWrite[bytes], matrix: _data_matrix, compressed: bool = True) -> None: ...
