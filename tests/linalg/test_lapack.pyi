# type-tests for `linalg/lapack.pyi`

from typing import assert_type

from scipy.linalg import get_lapack_funcs
from scipy.linalg.blas import _FortranFunction

###
# get_lapack_funcs

assert_type(get_lapack_funcs("getrf"), list[_FortranFunction] | _FortranFunction)
assert_type(get_lapack_funcs(["getrf", "getrs"]), list[_FortranFunction] | _FortranFunction)
