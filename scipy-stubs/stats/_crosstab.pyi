from collections.abc import Sequence
from typing import Any, Generic, Literal, Self, overload
from typing_extensions import TypeVar, override

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._typing import BaseBunch
from scipy.sparse import coo_matrix

###

_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_ScalarT_co = TypeVar("_ScalarT_co", bound=np.generic, default=Any, covariant=True)
_CountT_co = TypeVar(
    "_CountT_co",
    bound=onp.ArrayND[npc.integer] | coo_matrix[npc.integer],
    default=onp.ArrayND[np.intp] | coo_matrix[np.intp],
    covariant=True,
)

###

class CrosstabResult(BaseBunch[Any, Any], Generic[_ScalarT_co, _CountT_co]):
    @property
    def elements(self, /) -> tuple[onp.Array1D[_ScalarT_co], *tuple[onp.Array1D[_ScalarT_co], ...]]: ...
    @property
    @override
    def count(self, /) -> _CountT_co: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override]

    #
    def __new__(_cls, elements: tuple[onp.ArrayND[_ScalarT_co], ...], count: _CountT_co) -> Self: ...
    def __init__(self, /, elements: tuple[onp.ArrayND[_ScalarT_co], ...], count: _CountT_co) -> None: ...

#
@overload  # T@generic
def crosstab(
    arg0: onp.ToArrayND[_ScalarT, _ScalarT],
    /,
    *args: onp.ToArrayND[_ScalarT, _ScalarT],
    levels: onp.ToArrayND[_ScalarT, _ScalarT] | None = None,
    sparse: Literal[False] = False,
) -> CrosstabResult[_ScalarT, onp.ArrayND[np.intp]]: ...
@overload  # T@generic, sparse=True
def crosstab(
    arg0: onp.ToArrayND[_ScalarT, _ScalarT],
    arg1: onp.ToArrayND[_ScalarT, _ScalarT],
    /,
    *,
    levels: onp.ToArrayND[_ScalarT, _ScalarT] | None = None,
    sparse: Literal[True],
) -> CrosstabResult[_ScalarT, coo_matrix[np.intp]]: ...
@overload  # bool
def crosstab(  # type: ignore[overload-overlap]
    arg0: list[bool], /, *args: Sequence[bool], levels: Sequence[bool] | None = None, sparse: Literal[False] = False
) -> CrosstabResult[np.bool_, onp.ArrayND[np.intp]]: ...
@overload  # bool, sparse=True
def crosstab(  # type: ignore[overload-overlap]
    arg0: list[bool], arg1: Sequence[bool], /, *, levels: Sequence[bool] | None = None, sparse: Literal[True]
) -> CrosstabResult[np.bool_, coo_matrix[np.intp]]: ...
@overload  # int
def crosstab(
    arg0: list[int], /, *args: Sequence[int], levels: Sequence[int] | None = None, sparse: Literal[False] = False
) -> CrosstabResult[np.int_, onp.ArrayND[np.intp]]: ...
@overload  # int, sparse=True
def crosstab(
    arg0: list[int], arg1: Sequence[int], /, *, levels: Sequence[int] | None = None, sparse: Literal[True]
) -> CrosstabResult[np.int_, coo_matrix[np.intp]]: ...
@overload  # float
def crosstab(
    arg0: list[float], /, *args: Sequence[float], levels: Sequence[float] | None = None, sparse: Literal[False] = False
) -> CrosstabResult[np.float64, onp.ArrayND[np.intp]]: ...
@overload  # float, sparse=True
def crosstab(
    arg0: list[float], arg1: Sequence[float], /, *, levels: Sequence[float] | None = None, sparse: Literal[True]
) -> CrosstabResult[np.float64, coo_matrix[np.intp]]: ...
@overload  # complex
def crosstab(
    arg0: list[complex], /, *args: Sequence[complex], levels: Sequence[complex] | None = None, sparse: Literal[False] = False
) -> CrosstabResult[np.complex128, onp.ArrayND[np.intp]]: ...
@overload  # complex, sparse=True
def crosstab(
    arg0: list[complex], arg1: Sequence[complex], /, *, levels: Sequence[complex] | None = None, sparse: Literal[True]
) -> CrosstabResult[np.complex128, coo_matrix[np.intp]]: ...
@overload  # bytes
def crosstab(
    arg0: Sequence[bytes], /, *args: Sequence[bytes], levels: Sequence[bytes] | None = None, sparse: Literal[False] = False
) -> CrosstabResult[np.bytes_, onp.ArrayND[np.intp]]: ...
@overload  # bytes, sparse=True
def crosstab(
    arg0: Sequence[bytes], arg1: Sequence[bytes], /, *, levels: Sequence[bytes] | None = None, sparse: Literal[True]
) -> CrosstabResult[np.bytes_, coo_matrix[np.intp]]: ...
@overload  # str
def crosstab(
    arg0: Sequence[str], /, *args: Sequence[str], levels: Sequence[str] | None = None, sparse: Literal[False] = False
) -> CrosstabResult[np.str_, onp.ArrayND[np.intp]]: ...
@overload  # str, sparse=True
def crosstab(
    arg0: Sequence[str], arg1: Sequence[str], /, *, levels: Sequence[str] | None = None, sparse: Literal[True]
) -> CrosstabResult[np.str_, coo_matrix[np.intp]]: ...
