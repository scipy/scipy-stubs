from typing import Any, Final, Generic, Literal, Self, SupportsIndex, overload, override
from typing_extensions import TypeVar, deprecated

import numpy as np
import optype.numpy as onp

from ._ckdtree import cKDTree, cKDTreeNode

__all__ = ["KDTree", "Rectangle", "distance_matrix", "minkowski_distance", "minkowski_distance_p"]

###

type _Float1D = onp.Array1D[np.float64]
type _Float2D = onp.Array2D[np.float64]

_BoxSizeT_co = TypeVar("_BoxSizeT_co", bound=_Float2D | None, default=_Float2D | None, covariant=True)
_BoxSizeDataT_co = TypeVar("_BoxSizeDataT_co", bound=_Float1D | None, default=_Float1D | None, covariant=True)

###

class Rectangle:
    maxes: Final[onp.Array1D[np.float64]]
    mins: Final[onp.Array1D[np.float64]]
    m: Final[int]

    def __init__(self, /, maxes: onp.ToFloat1D, mins: onp.ToFloat1D) -> None: ...
    def volume(self, /) -> np.float64: ...
    def split(self, /, d: SupportsIndex, split: onp.ToFloat) -> tuple[Self, Self]: ...
    def min_distance_point(self, /, x: onp.ToFloat | onp.ToFloatND, p: onp.ToFloat = 2.0) -> np.float64: ...
    def max_distance_point(self, /, x: onp.ToFloat | onp.ToFloatND, p: onp.ToFloat = 2.0) -> np.float64: ...
    def min_distance_rectangle(self, /, other: Rectangle, p: onp.ToFloat = 2.0) -> np.float64: ...
    def max_distance_rectangle(self, /, other: Rectangle, p: onp.ToFloat = 2.0) -> np.float64: ...

class KDTree(cKDTree[_BoxSizeT_co, _BoxSizeDataT_co], Generic[_BoxSizeT_co, _BoxSizeDataT_co]):
    class node:
        @staticmethod
        def _create(ckdtree_node: cKDTreeNode | None = None) -> KDTree.leafnode | KDTree.innernode: ...
        def __init__(self, /, ckdtree_node: cKDTreeNode | None = None) -> None: ...
        def __lt__(self, other: object, /) -> bool: ...
        def __gt__(self, other: object, /) -> bool: ...
        def __le__(self, other: object, /) -> bool: ...
        def __ge__(self, other: object, /) -> bool: ...
        @override
        def __eq__(self, other: object, /) -> bool: ...

    class leafnode(node):
        @property
        def idx(self, /) -> onp.ArrayND[np.intp]: ...
        @property
        def children(self, /) -> int: ...

    class innernode(node):
        less: Final[KDTree.innernode | KDTree.leafnode]
        greater: Final[KDTree.innernode | KDTree.leafnode]

        def __init__(self, /, ckdtreenode: cKDTreeNode) -> None: ...
        @property
        def split_dim(self, /) -> int: ...
        @property
        def split(self, /) -> float: ...
        @property
        def children(self, /) -> int: ...

    @overload
    def __init__(
        self: KDTree[None, None],
        /,
        data: onp.ToComplexND,
        leafsize: onp.ToInt = 10,
        compact_nodes: bool = True,
        copy_data: bool = False,
        balanced_tree: bool = True,
        boxsize: None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: KDTree[_Float2D, _Float1D],
        /,
        data: onp.ToComplexND,
        leafsize: onp.ToInt,
        compact_nodes: bool,
        copy_data: bool,
        balanced_tree: bool,
        boxsize: onp.ToFloat2D,
    ) -> None: ...
    @overload
    def __init__(
        self: KDTree[_Float2D, _Float1D],
        /,
        data: onp.ToComplexND,
        leafsize: onp.ToInt = 10,
        compact_nodes: bool = True,
        copy_data: bool = False,
        balanced_tree: bool = True,
        *,
        boxsize: onp.ToFloat2D,
    ) -> None: ...

    #
    @override  # type:ignore[override]
    @overload  # 1d, k=1
    def query(  # pyrefly:ignore[bad-override]
        self,
        /,
        x: onp.ToFloatStrict1D,
        k: Literal[1] = 1,
        eps: onp.ToFloat = 0.0,
        p: onp.ToFloat = 2.0,
        distance_upper_bound: float = float("inf"),  # ruff: ignore[typed-argument-default-in-stub]
        workers: int | None = 1,
    ) -> tuple[float, np.intp]: ...
    @overload  # 1d
    def query(
        self,
        /,
        x: onp.ToFloatStrict1D,
        k: onp.ToInt | onp.ToInt1D,
        eps: onp.ToFloat = 0.0,
        p: onp.ToFloat = 2.0,
        distance_upper_bound: float = float("inf"),  # ruff: ignore[typed-argument-default-in-stub]
        workers: int | None = 1,
    ) -> tuple[onp.Array1D[np.float64], onp.Array1D[np.intp]] | Any: ...
    @overload  # 2d, k=1
    def query(
        self,
        /,
        x: onp.ToFloatStrict2D,
        k: Literal[1] = 1,
        eps: onp.ToFloat = 0.0,
        p: onp.ToFloat = 2.0,
        distance_upper_bound: float = float("inf"),  # ruff: ignore[typed-argument-default-in-stub]
        workers: int | None = 1,
    ) -> tuple[onp.Array1D[np.float64], onp.Array1D[np.intp]]: ...
    @overload  # 2d
    def query(  # pyright:ignore[reportIncompatibleMethodOverride]  # ty:ignore[invalid-method-override]
        self,
        /,
        x: onp.ToFloatStrict2D,
        k: onp.ToInt | onp.ToInt1D,
        eps: onp.ToFloat = 0.0,
        p: onp.ToFloat = 2.0,
        distance_upper_bound: float = float("inf"),  # ruff: ignore[typed-argument-default-in-stub]
        workers: int | None = 1,
    ) -> tuple[onp.Array2D[np.float64], onp.Array2D[np.intp]] | Any: ...

@deprecated("This function is deprecated in favor of `scipy.spatial.distance.minkowski` and will be removed in SciPy 2.1.0.")
def minkowski_distance_p(x: onp.ToComplexND, y: onp.ToComplexND, p: float = 2.0) -> onp.ArrayND[np.float64]: ...

#
@deprecated("This function is deprecated in favor of `scipy.spatial.distance.minkowski` and will be removed in SciPy 2.1.0.")
def minkowski_distance(x: onp.ToComplexND, y: onp.ToComplexND, p: float = 2.0) -> onp.ArrayND[np.float64]: ...

#
@deprecated("This function is deprecated in favor of `scipy.spatial.distance.cdist` and will be removed in SciPy 2.1.0.")
def distance_matrix(
    x: onp.ToComplexND, y: onp.ToComplexND, p: float = 2.0, threshold: int = 1_000_000
) -> onp.Array2D[np.float64]: ...
