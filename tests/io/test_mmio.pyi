from typing import Literal, assert_type

from scipy.io import mminfo

###

# mminfo
assert_type(
    mminfo("file.mtx"),
    tuple[
        int,
        int,
        int,
        Literal["coordinate", "array"],
        Literal["real", "complex", "pattern", "integer"],
        Literal["general", "symmetric", "skew-symmetric", "hermitian"],
    ],
)
