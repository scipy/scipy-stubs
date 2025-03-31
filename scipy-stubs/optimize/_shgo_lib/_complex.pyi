from collections.abc import Callable, Generator, Sequence
from typing import Any, Concatenate, Final, Generic, TypeAlias
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from scipy.optimize._typing import Constraints
from ._vertex import VertexBase, VertexCacheBase

_Floats: TypeAlias = onp.ToFloat
_Float1D: TypeAlias = onp.Array1D[np.float64]
_FloatingND: TypeAlias = onp.ArrayND[np.floating[Any] | np.integer[Any]]

_Fun0D: TypeAlias = Callable[Concatenate[_Float1D, ...], onp.ToFloat]
_Fun1D: TypeAlias = Callable[Concatenate[_Float1D, ...], onp.ToFloat1D]

_Location: TypeAlias = _FloatingND | Sequence[float | int | bool]
_Bounds: TypeAlias = _FloatingND | Sequence[tuple[onp.ToFloat, onp.ToFloat]]
_Symmetry: TypeAlias = onp.ArrayND[np.integer[Any]] | op.CanGetitem[int | bool, op.CanIndex]

_HT = TypeVar("_HT", bound=VertexCacheBase, default=VertexCacheBase)

###

class Complex(Generic[_HT]):  # undocumented
    dim: Final[int | bool]
    domain: Final[_Bounds]
    bounds: Final[_Bounds]
    symmetry: Final[_Symmetry]
    sfield: _Fun0D
    sfield_args: tuple[object, ...]
    min_cons: Constraints  # only set if `constraints` is not None
    g_cons: Final[Sequence[_Fun1D]]
    g_args: Final[tuple[object, ...]]

    gen: int | bool
    perm_cycle: int | bool

    H: list[_HT]  # readonly
    V: Final[VertexCacheBase]
    V_non_symm: Final[list[VertexBase]]
    origin: list[float | int | bool]
    supremum: list[float | int | bool]
    triangulated_vectors: list[tuple[_Floats, _Floats]]

    cp: Generator[_Floats, None, _Floats]
    rls: Generator[VertexBase | _Floats]

    def __init__(
        self,
        /,
        dim: int | bool,
        domain: _Bounds | None = None,
        sfield: _Fun0D | None = None,
        sfield_args: tuple[object, ...] = (),
        symmetry: _Symmetry | None = None,
        constraints: Constraints | None = None,
        workers: int | bool = 1,
    ) -> None: ...
    def __call__(self, /) -> list[_HT]: ...
    def cyclic_product(
        self,
        /,
        bounds: _Bounds,
        origin: _Location,
        supremum: _Location,
        centroid: onp.ToFloat = True,
    ) -> Generator[_Floats, None, _Floats]: ...
    def triangulate(
        self,
        /,
        n: int | bool | None = None,
        symmetry: _Symmetry | None = None,
        centroid: onp.ToFloat = True,
        printout: onp.ToFloat = False,
    ) -> None: ...
    def refine(self, /, n: int | bool = 1) -> None: ...
    def refine_all(self, /, centroids: onp.ToBool = True) -> None: ...
    def refine_local_space(
        self,
        /,
        origin: _Location,
        supremum: _Location,
        bounds: _Bounds,
        centroid: onp.ToBool = 1,
    ) -> Generator[VertexBase | _Floats]: ...
    def refine_star(self, /, v: VertexBase) -> None: ...
    def split_edge(self, /, v1: VertexBase, v2: VertexBase) -> VertexBase: ...
    def vpool(self, /, origin: _Location, supremum: _Location) -> set[VertexBase]: ...
    def vf_to_vv(
        self,
        /,
        vertices: Sequence[VertexBase],
        simplices: Sequence[tuple[onp.ToFloat1D, onp.ToFloat1D]],
    ) -> None: ...
    def connect_vertex_non_symm(
        self,
        /,
        v_x: onp.ToFloat1D,
        near: set[VertexBase] | list[VertexBase] | None = None,
    ) -> bool | None: ...
    def in_simplex(self, /, S: onp.ToFloat1D, v_x: onp.ToFloat1D, A_j0: _FloatingND | None = None) -> bool: ...
    def deg_simplex(self, /, S: _FloatingND, proj: _FloatingND | None = None) -> bool: ...
