# NOTE(scipy-stubs): This module only exists `if typing.TYPE_CHECKING: ...`, and has no stable API.

from typing import Literal, Protocol, SupportsIndex, TypeAlias, TypeVar, final, type_check_only
from typing_extensions import TypeAliasType

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = (
    "Index1D",
    "Numeric",
    "SPFormat",
    "ToShape1D",
    "ToShape2D",
    "ToShapeMax2D",
    "ToShapeMin1D",
    "ToShapeMin3D",
    "_CanStack",
    "_CanStackAs",
)

###

# NOTE: For convenience, this does no explicitly disallow `float16`, which is not supported by SciPy.
Numeric = TypeAliasType("Numeric", np.bool_ | npc.number)

Index1D: TypeAlias = onp.Array1D[np.int32]

ToShape1D: TypeAlias = tuple[SupportsIndex]  # ndim == 1
ToShape2D: TypeAlias = tuple[SupportsIndex, SupportsIndex]  # ndim == 2
ToShapeMax2D: TypeAlias = ToShape1D | ToShape2D  # ndim <= 2
ToShapeMin1D: TypeAlias = tuple[SupportsIndex, *tuple[SupportsIndex, ...]]  # ndim >= 1
ToShapeMin3D: TypeAlias = tuple[SupportsIndex, SupportsIndex, SupportsIndex, *tuple[SupportsIndex, ...]]  # ndim >= 2

SPFormat: TypeAlias = Literal["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]

###
# Interfaces for emulated dependent associated types

_AssocT_co = TypeVar("_AssocT_co", covariant=True)
_ScalarT_contra = TypeVar("_ScalarT_contra", bound=Numeric, contravariant=True)

@final
@type_check_only
class _CanStack(Protocol[_AssocT_co]):
    @type_check_only
    def __assoc_stacked__(self, /) -> _AssocT_co: ...

@final
@type_check_only
class _CanStackAs(Protocol[_ScalarT_contra, _AssocT_co]):
    @type_check_only
    def __assoc_stacked_as__(self, sctype: _ScalarT_contra, /) -> _AssocT_co: ...
