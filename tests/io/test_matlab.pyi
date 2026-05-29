from typing import Any, Literal, assert_type

from scipy.io import loadmat, savemat, whosmat
from scipy.sparse import coo_array, coo_matrix, csc_array, csc_matrix

###

# loadmat
assert_type(loadmat("file.mat"), dict[str, csc_matrix | coo_matrix])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(loadmat("file.mat", mdict={}), dict[str, csc_matrix | coo_matrix])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(loadmat("file.mat", appendmat=False), dict[str, csc_matrix | coo_matrix])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(loadmat("file.mat", spmatrix=True), dict[str, csc_matrix | coo_matrix])
assert_type(loadmat("file.mat", spmatrix=False), dict[str, csc_array | coo_array[Any, tuple[int, int]]])
assert_type(loadmat("file.mat", byte_order="native"), dict[str, csc_matrix | coo_matrix])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(loadmat("file.mat", mat_dtype=True), dict[str, csc_matrix | coo_matrix])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(loadmat("file.mat", squeeze_me=False), dict[str, csc_matrix | coo_matrix])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(loadmat("file.mat", chars_as_strings=False), dict[str, csc_matrix | coo_matrix])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(loadmat("file.mat", matlab_compatible=True), dict[str, csc_matrix | coo_matrix])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(loadmat("file.mat", struct_as_record=True), dict[str, csc_matrix | coo_matrix])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(loadmat("file.mat", verify_compressed_data_integrity=True), dict[str, csc_matrix | coo_matrix])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(loadmat("file.mat", simplify_cells=True), dict[str, csc_matrix | coo_matrix])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(loadmat("file.mat", variable_names=("a", "b")), dict[str, csc_matrix | coo_matrix])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

# savemat
assert_type(savemat("file.mat", {"": ""}), None)
assert_type(savemat("file.mat", {"": ""}, appendmat=False), None)
assert_type(savemat("file.mat", {"": ""}, format="4"), None)
assert_type(savemat("file.mat", {"": ""}, long_field_names=True), None)
assert_type(savemat("file.mat", {"": ""}, do_compression=True), None)
assert_type(savemat("file.mat", {"": ""}, oned_as="column"), None)

# whosmat
assert_type(
    whosmat("file.mat"),
    list[
        tuple[
            str,
            tuple[int, ...],
            Literal[
                "int8",
                "uint8",
                "int16",
                "uint16",
                "int32",
                "uint32",
                "int64",
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
            ],
        ]
    ],
)
