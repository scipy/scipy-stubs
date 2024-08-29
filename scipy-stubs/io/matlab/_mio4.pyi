from scipy._typing import Untyped

from ._mio_utils import chars_to_strings as chars_to_strings, squeeze_element as squeeze_element
from ._miobase import (
    MatFileReader as MatFileReader,
    arr_dtype_number as arr_dtype_number,
    arr_to_chars as arr_to_chars,
    convert_dtypes as convert_dtypes,
    docfiller as docfiller,
    matdims as matdims,
    read_dtype as read_dtype,
)

SYS_LITTLE_ENDIAN: Untyped
miDOUBLE: int
miSINGLE: int
miINT32: int
miINT16: int
miUINT16: int
miUINT8: int
mdtypes_template: Untyped
np_to_mtypes: Untyped
mxFULL_CLASS: int
mxCHAR_CLASS: int
mxSPARSE_CLASS: int
order_codes: Untyped
mclass_info: Untyped

class VarHeader4:
    is_logical: bool
    is_global: bool
    name: Untyped
    dtype: Untyped
    mclass: Untyped
    dims: Untyped
    is_complex: Untyped
    def __init__(self, name, dtype, mclass, dims, is_complex) -> None: ...

class VarReader4:
    file_reader: Untyped
    mat_stream: Untyped
    dtypes: Untyped
    chars_as_strings: Untyped
    squeeze_me: Untyped
    def __init__(self, file_reader) -> None: ...
    def read_header(self) -> Untyped: ...
    def array_from_header(self, hdr, process: bool = True) -> Untyped: ...
    def read_sub_array(self, hdr, copy: bool = True) -> Untyped: ...
    def read_full_array(self, hdr) -> Untyped: ...
    def read_char_array(self, hdr) -> Untyped: ...
    def read_sparse_array(self, hdr) -> Untyped: ...
    def shape_from_header(self, hdr) -> Untyped: ...

class MatFile4Reader(MatFileReader):
    def __init__(self, mat_stream, *args, **kwargs) -> None: ...
    def guess_byte_order(self) -> Untyped: ...
    dtypes: Untyped
    def initialize_read(self): ...
    def read_var_header(self) -> Untyped: ...
    def read_var_array(self, header, process: bool = True) -> Untyped: ...
    def get_variables(self, variable_names: Untyped | None = None) -> Untyped: ...
    def list_variables(self) -> Untyped: ...

def arr_to_2d(arr, oned_as: str = "row") -> Untyped: ...

class VarWriter4:
    file_stream: Untyped
    oned_as: Untyped
    def __init__(self, file_writer) -> None: ...
    def write_bytes(self, arr): ...
    def write_string(self, s): ...
    def write_header(self, name, shape, P=..., T=..., imagf: int = 0): ...
    def write(self, arr, name): ...
    def write_numeric(self, arr, name): ...
    def write_char(self, arr, name): ...
    def write_sparse(self, arr, name): ...

class MatFile4Writer:
    file_stream: Untyped
    oned_as: Untyped
    def __init__(self, file_stream, oned_as: Untyped | None = None): ...
    def put_variables(self, mdict, write_header: Untyped | None = None): ...
