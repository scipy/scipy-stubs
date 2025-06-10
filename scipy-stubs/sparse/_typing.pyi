# NOTE(scipy-stubs): This ia a module only exists `if typing.TYPE_CHECKING: ...`, and has no stable API.

from typing import Literal, TypeAlias
from typing_extensions import TypeAliasType

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ("Index1D", "Numeric", "SPFormat", "ToShape1D", "ToShape1D", "ToShape2D", "ToShapeMin1D", "ToShapeMin3D")

###

# NOTE: For convenience, this does no explicitly disallow `float16`, which is not supported by SciPy.
Numeric = TypeAliasType("Numeric", np.bool_ | npc.number)

Index1D: TypeAlias = onp.Array1D[np.int32 | np.int64]

ToShape1D: TypeAlias = tuple[op.CanIndex]  # ndim == 1
ToShape2D: TypeAlias = tuple[op.CanIndex, op.CanIndex]  # ndim == 2
ToShapeMin1D: TypeAlias = tuple[op.CanIndex, *tuple[op.CanIndex, ...]]  # ndim >= 1
ToShapeMin3D: TypeAlias = tuple[op.CanIndex, op.CanIndex, op.CanIndex, *tuple[op.CanIndex, ...]]  # ndim >= 2

SPFormat: TypeAlias = Literal["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]
