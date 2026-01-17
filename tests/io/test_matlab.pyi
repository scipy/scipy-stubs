from typing import Any, assert_type

from scipy.io import loadmat, savemat

###

# loadmat
assert_type(loadmat("file.mat"), dict[str, Any])
assert_type(loadmat("file.mat", mdict={}), dict[str, Any])
assert_type(loadmat("file.mat", appendmat=False), dict[str, Any])
assert_type(loadmat("file.mat", spmatrix=False), dict[str, Any])
assert_type(loadmat("file.mat", byte_order="native"), dict[str, Any])
assert_type(loadmat("file.mat", mat_dtype=True), dict[str, Any])
assert_type(loadmat("file.mat", squeeze_me=False), dict[str, Any])
assert_type(loadmat("file.mat", chars_as_strings=False), dict[str, Any])
assert_type(loadmat("file.mat", matlab_compatible=True), dict[str, Any])
assert_type(loadmat("file.mat", struct_as_record=True), dict[str, Any])
assert_type(loadmat("file.mat", verify_compressed_data_integrity=True), dict[str, Any])
assert_type(loadmat("file.mat", simplify_cells=True), dict[str, Any])
assert_type(loadmat("file.mat", variable_names=("a", "b")), dict[str, Any])

# savemat
assert_type(savemat("file.mat", {"": ""}), None)
assert_type(savemat("file.mat", {"": ""}, appendmat=False), None)
assert_type(savemat("file.mat", {"": ""}, format="4"), None)
assert_type(savemat("file.mat", {"": ""}, long_field_names=True), None)
assert_type(savemat("file.mat", {"": ""}, do_compression=True), None)
assert_type(savemat("file.mat", {"": ""}, oned_as="column"), None)
