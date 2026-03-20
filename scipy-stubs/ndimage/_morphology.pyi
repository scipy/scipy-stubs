from collections.abc import Iterable
from typing import Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = [
    "binary_closing",
    "binary_dilation",
    "binary_erosion",
    "binary_fill_holes",
    "binary_hit_or_miss",
    "binary_opening",
    "binary_propagation",
    "black_tophat",
    "distance_transform_bf",
    "distance_transform_cdt",
    "distance_transform_edt",
    "generate_binary_structure",
    "grey_closing",
    "grey_dilation",
    "grey_erosion",
    "grey_opening",
    "iterate_structure",
    "morphological_gradient",
    "morphological_laplace",
    "white_tophat",
]

###

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])
_OriginScalarT = TypeVar("_OriginScalarT", bound=int | npc.integer)

_Mode: TypeAlias = Literal["reflect", "constant", "nearest", "mirror", "wrap"]
_MetricCDT: TypeAlias = Literal["chessboard", "taxicab"]
_MetricBF: TypeAlias = Literal["euclidean", _MetricCDT]

_Origin: TypeAlias = int | tuple[int, ...]

###

@overload  # known shape
def iterate_structure(
    structure: onp.ArrayND[np.bool_ | npc.integer, _ShapeT], iterations: int, origin: None = None
) -> onp.ArrayND[np.bool_, _ShapeT]: ...
@overload  # known shape, origin=<given>
def iterate_structure(
    structure: onp.ArrayND[np.bool_ | npc.integer, _ShapeT], iterations: int, origin: _OriginScalarT | Iterable[_OriginScalarT]
) -> tuple[onp.ArrayND[np.bool_, _ShapeT], list[_OriginScalarT]]: ...
@overload  # unknown shape
def iterate_structure(structure: onp.ToIntND, iterations: int, origin: None = None) -> onp.ArrayND[np.bool_]: ...
@overload  # unknown shape, origin=<given>
def iterate_structure(
    structure: onp.ToIntND, iterations: int, origin: _OriginScalarT | Iterable[_OriginScalarT]
) -> tuple[onp.ArrayND[np.bool_], list[_OriginScalarT]]: ...

#
@overload
def generate_binary_structure(rank: Literal[0, -1, -2, -3], connectivity: int) -> onp.Array0D[np.bool_]: ...
@overload
def generate_binary_structure(rank: Literal[1], connectivity: int) -> onp.Array1D[np.bool_]: ...
@overload
def generate_binary_structure(rank: Literal[2], connectivity: int) -> onp.Array2D[np.bool_]: ...
@overload
def generate_binary_structure(rank: Literal[3], connectivity: int) -> onp.Array3D[np.bool_]: ...
@overload
def generate_binary_structure(rank: int, connectivity: int) -> onp.ArrayND[np.bool_]: ...

#
def binary_erosion(
    input: onp.ToComplex | onp.ToComplexND,
    structure: onp.ToIntND | None = None,
    iterations: int = 1,
    mask: onp.ToInt | onp.ToIntND | None = None,
    output: onp.ArrayND[np.bool_] | type[bool | np.bool_] | None = None,
    border_value: onp.ToInt = 0,
    origin: _Origin = 0,
    brute_force: bool = False,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[np.bool_]: ...
def binary_dilation(
    input: onp.ToComplex | onp.ToComplexND,
    structure: onp.ToIntND | None = None,
    iterations: int = 1,
    mask: onp.ToInt | onp.ToIntND | None = None,
    output: onp.ArrayND[np.bool_] | type[bool | np.bool_] | None = None,
    border_value: onp.ToInt = 0,
    origin: _Origin = 0,
    brute_force: bool = False,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[np.bool_]: ...
def binary_opening(
    input: onp.ToComplex | onp.ToComplexND,
    structure: onp.ToIntND | None = None,
    iterations: int = 1,
    output: onp.ArrayND[np.bool_] | type[bool | np.bool_] | None = None,
    origin: _Origin = 0,
    mask: onp.ToInt | onp.ToIntND | None = None,
    border_value: onp.ToInt = 0,
    brute_force: bool = False,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[np.bool_]: ...
def binary_closing(
    input: onp.ToComplex | onp.ToComplexND,
    structure: onp.ToIntND | None = None,
    iterations: int = 1,
    output: onp.ArrayND[np.bool_] | type[bool | np.bool_] | None = None,
    origin: _Origin = 0,
    mask: onp.ToInt | onp.ToIntND | None = None,
    border_value: onp.ToInt = 0,
    brute_force: bool = False,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[np.bool_]: ...

#
def binary_hit_or_miss(
    input: onp.ToComplex | onp.ToComplexND,
    structure1: onp.ToInt | onp.ToIntND | None = None,
    structure2: onp.ToInt | onp.ToIntND | None = None,
    output: onp.ArrayND[np.bool_] | type[bool | np.bool_] | None = None,
    origin1: _Origin = 0,
    origin2: _Origin | None = None,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[np.bool_]: ...
def binary_propagation(
    input: onp.ToComplex | onp.ToComplexND,
    structure: onp.ToIntND | None = None,
    mask: onp.ToInt | onp.ToIntND | None = None,
    output: onp.ArrayND[np.bool_] | type[bool | np.bool_] | None = None,
    border_value: onp.ToInt = 0,
    origin: _Origin = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[np.bool_]: ...
def binary_fill_holes(
    input: onp.ToComplex | onp.ToComplexND,
    structure: onp.ToIntND | None = None,
    output: onp.ArrayND[np.bool_] | None = None,
    origin: _Origin = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[np.bool_]: ...

#
def grey_erosion(
    input: onp.ToComplex | onp.ToComplexND,
    size: tuple[int, ...] | None = None,
    footprint: onp.ToScalar | onp.ToArrayND | None = None,
    structure: onp.ToIntND | None = None,
    output: onp.ArrayND[npc.number | np.bool_] | None = None,
    mode: _Mode = "reflect",
    cval: onp.ToComplex = 0.0,
    origin: _Origin = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[npc.number | np.bool_]: ...
def grey_dilation(
    input: onp.ToComplex | onp.ToComplexND,
    size: tuple[int, ...] | None = None,
    footprint: onp.ToScalar | onp.ToArrayND | None = None,
    structure: onp.ToIntND | None = None,
    output: onp.ArrayND[npc.number | np.bool_] | None = None,
    mode: _Mode = "reflect",
    cval: onp.ToComplex = 0.0,
    origin: _Origin = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[npc.number | np.bool_]: ...
def grey_opening(
    input: onp.ToComplex | onp.ToComplexND,
    size: tuple[int, ...] | None = None,
    footprint: onp.ToScalar | onp.ToArrayND | None = None,
    structure: onp.ToIntND | None = None,
    output: onp.ArrayND[npc.number | np.bool_] | None = None,
    mode: _Mode = "reflect",
    cval: onp.ToComplex = 0.0,
    origin: _Origin = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[npc.number | np.bool_]: ...
def grey_closing(
    input: onp.ToComplex | onp.ToComplexND,
    size: tuple[int, ...] | None = None,
    footprint: onp.ToScalar | onp.ToArrayND | None = None,
    structure: onp.ToIntND | None = None,
    output: onp.ArrayND[npc.number | np.bool_] | None = None,
    mode: _Mode = "reflect",
    cval: onp.ToComplex = 0.0,
    origin: _Origin = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[npc.number | np.bool_]: ...

#
def morphological_gradient(
    input: onp.ToComplex | onp.ToComplexND,
    size: tuple[int, ...] | None = None,
    footprint: onp.ToScalar | onp.ToArrayND | None = None,
    structure: onp.ToIntND | None = None,
    output: onp.ArrayND[npc.number | np.bool_] | None = None,
    mode: _Mode = "reflect",
    cval: onp.ToComplex = 0.0,
    origin: _Origin = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[npc.number | np.bool_]: ...
def morphological_laplace(
    input: onp.ToComplex | onp.ToComplexND,
    size: tuple[int, ...] | None = None,
    footprint: onp.ToScalar | onp.ToArrayND | None = None,
    structure: onp.ToIntND | None = None,
    output: onp.ArrayND[npc.number | np.bool_] | None = None,
    mode: _Mode = "reflect",
    cval: onp.ToComplex = 0.0,
    origin: _Origin = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[npc.number | np.bool_]: ...

#
def white_tophat(
    input: onp.ToComplex | onp.ToComplexND,
    size: tuple[int, ...] | None = None,
    footprint: onp.ToScalar | onp.ToArrayND | None = None,
    structure: onp.ToIntND | None = None,
    output: onp.ArrayND[npc.number | np.bool_] | None = None,
    mode: _Mode = "reflect",
    cval: onp.ToComplex = 0.0,
    origin: _Origin = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[npc.number | np.bool_]: ...
def black_tophat(
    input: onp.ToComplex | onp.ToComplexND,
    size: tuple[int, ...] | None = None,
    footprint: onp.ToScalar | onp.ToArrayND | None = None,
    structure: onp.ToIntND | None = None,
    output: onp.ArrayND[npc.number | np.bool_] | None = None,
    mode: _Mode = "reflect",
    cval: onp.ToComplex = 0.0,
    origin: onp.ToComplex = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> onp.ArrayND[npc.number | np.bool_]: ...

#
def distance_transform_bf(
    input: onp.ToComplex | onp.ToComplexND,
    metric: _MetricBF = "euclidean",
    sampling: onp.ToFloat | onp.ToFloatND | None = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: onp.ArrayND[np.float64 | np.uint32] | None = None,
    indices: onp.ArrayND[np.int32] | None = None,
) -> (
    onp.ArrayND[npc.number | np.bool_] | onp.ArrayND[np.int32] | tuple[onp.ArrayND[npc.number | np.bool_], onp.ArrayND[np.int32]]
): ...

#
def distance_transform_cdt(
    input: onp.ToComplex | onp.ToComplexND,
    metric: _MetricCDT | onp.ToScalar | onp.ToArrayND = "chessboard",
    return_distances: bool = True,
    return_indices: bool = False,
    distances: onp.ArrayND[np.int32] | None = None,
    indices: onp.ArrayND[np.int32] | None = None,
) -> onp.ArrayND[np.int32] | tuple[onp.ArrayND[np.int32], onp.ArrayND[np.int32]]: ...

#
def distance_transform_edt(
    input: onp.ToComplex | onp.ToComplexND,
    sampling: onp.ToScalar | onp.ToArrayND | None = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: onp.ArrayND[np.float64] | None = None,
    indices: onp.ArrayND[np.int32] | None = None,
) -> onp.ArrayND[np.float64] | onp.ArrayND[np.int32] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]]: ...
