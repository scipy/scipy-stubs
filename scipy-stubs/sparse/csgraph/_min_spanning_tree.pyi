from typing import Any, Final, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.sparse import csr_array, csr_matrix, sparray, spmatrix
from scipy.sparse._base import _spbase

###

DTYPE: Final[type[np.float64]] = ...
ITYPE: Final[type[np.int32]] = ...

@overload
def minimum_spanning_tree(csgraph: spmatrix[npc.integer | npc.floating], overwrite: bool = False) -> csr_matrix[np.float64]: ...
@overload
def minimum_spanning_tree(
    csgraph: onp.ToFloat2D | sparray[npc.integer | npc.floating, tuple[int, int]], overwrite: bool = False
) -> csr_array[np.float64, tuple[int, int]]: ...
@overload
def minimum_spanning_tree(
    csgraph: _spbase[npc.integer | npc.floating, tuple[int, int]], overwrite: bool = False
) -> csr_array[np.float64, tuple[int, int]] | Any: ...
