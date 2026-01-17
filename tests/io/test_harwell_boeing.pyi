from typing import assert_type

import optype.numpy.compat as npc

from scipy.io import hb_read, hb_write
from scipy.sparse import csc_array, csc_matrix

###

_csc_matrix: csc_matrix[npc.floating]
_csc_array: csc_array[npc.floating]

###

# hb_read
assert_type(hb_read("file.hb"), csc_matrix)
assert_type(hb_read("file.hb", spmatrix=True), csc_matrix)
assert_type(hb_read("file.hb", spmatrix=False), csc_array)

# hb_write
assert_type(hb_write("file.hb", _csc_matrix), None)
assert_type(hb_write("file.hb", _csc_array), None)
