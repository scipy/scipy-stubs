from typing import Any, Final, Literal

import numpy as np
import optype.numpy.compat as npc

from scipy.sparse import csr_array, csr_matrix

###

DTYPE: Final[type[np.float64]] = ...
ITYPE: Final[type[np.int32]] = ...

class MaximumFlowResult:
    flow_value: Final[np.int_]
    flow: Final[csr_array[np.int32] | Any]

    def __init__(self, /, flow_value: np.int_, flow: csr_array[np.int32]) -> None: ...

#
def maximum_flow(
    csgraph: csr_array[npc.integer] | csr_matrix[npc.integer],
    source: int,
    sink: int,
    *,
    method: Literal["edmonds_karp", "dinic"] = "dinic",
) -> MaximumFlowResult: ...

#
def _add_reverse_edges(a: csr_array[np.int32]) -> csr_array[np.int32]: ...  # undocumented
def _make_edge_pointers(a: csr_array[npc.integer]) -> csr_array[np.int32]: ...  # undocumented
def _make_tails(a: csr_array[npc.integer]) -> csr_array[np.int32]: ...  # undocumented
