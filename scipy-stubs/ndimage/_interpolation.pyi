from collections.abc import Callable
from typing import Concatenate, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp

__all__ = [
    "affine_transform",
    "geometric_transform",
    "map_coordinates",
    "rotate",
    "shift",
    "spline_filter",
    "spline_filter1d",
    "zoom",
]

_ScalarT = TypeVar("_ScalarT", bound=np.generic)

_Order: TypeAlias = Literal[0, 1, 2, 3, 4, 5]
_Mode: TypeAlias = Literal["reflect", "grid-mirror", "constant", "grid-constant", "nearest", "mirror", "wrap", "grid-wrap"]
_MappingFunc: TypeAlias = Callable[Concatenate[tuple[int, ...], ...], tuple[onp.ToFloat, ...]]
_ArrayOrDType: TypeAlias = onp.ArrayND[_ScalarT] | type[_ScalarT] | np.dtype[_ScalarT]

_FloatArrayOut: TypeAlias = onp.ArrayND[np.float64 | np.float32]
_ComplexArrayOut: TypeAlias = onp.ArrayND[np.complex128 | np.float64 | np.complex64 | np.float32]

#
@overload
def spline_filter1d(
    input: onp.ToFloatND,
    order: _Order = 3,
    axis: int = -1,
    output: type[float | np.float64] = np.float64,  # noqa: PYI011
    mode: _Mode = "mirror",
) -> onp.ArrayND[np.float64]: ...
@overload
def spline_filter1d(
    input: onp.ToComplexND, order: _Order = 3, axis: int = -1, *, output: type[op.JustComplex], mode: _Mode = "mirror"
) -> onp.ArrayND[np.complex128]: ...
@overload
def spline_filter1d(
    input: onp.ToComplexND, order: _Order, axis: int, output: _ArrayOrDType[_ScalarT], mode: _Mode = "mirror"
) -> onp.ArrayND[_ScalarT]: ...
@overload
def spline_filter1d(
    input: onp.ToComplexND, order: _Order = 3, axis: int = -1, *, output: _ArrayOrDType[_ScalarT], mode: _Mode = "mirror"
) -> onp.ArrayND[_ScalarT]: ...

#
@overload
def spline_filter(
    input: onp.ToFloatND,
    order: _Order = 3,
    output: type[float | np.float64] = np.float64,  # noqa: PYI011
    mode: _Mode = "mirror",
) -> onp.ArrayND[np.float64]: ...
@overload
def spline_filter(
    input: onp.ToComplexND, order: _Order = 3, *, output: type[op.JustComplex], mode: _Mode = "mirror"
) -> onp.ArrayND[np.complex128]: ...
@overload
def spline_filter(
    input: onp.ToComplexND, order: _Order, output: _ArrayOrDType[_ScalarT], mode: _Mode = "mirror"
) -> onp.ArrayND[_ScalarT]: ...
@overload
def spline_filter(
    input: onp.ToComplexND, order: _Order = 3, *, output: _ArrayOrDType[_ScalarT], mode: _Mode = "mirror"
) -> onp.ArrayND[_ScalarT]: ...

#
@overload
def geometric_transform(
    input: onp.ToFloat | onp.ToFloatND,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None = None,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> _FloatArrayOut: ...
@overload
def geometric_transform(
    input: onp.ToComplex | onp.ToComplexND,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None = None,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> _ComplexArrayOut: ...
@overload
def geometric_transform(
    input: onp.ToScalar | onp.ToArrayND,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None,
    output: _ArrayOrDType[_ScalarT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> onp.ArrayND[_ScalarT]: ...
@overload
def geometric_transform(
    input: onp.ToScalar | onp.ToArrayND,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: _ArrayOrDType[_ScalarT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> onp.ArrayND[_ScalarT]: ...
@overload
def geometric_transform(
    input: onp.ToScalar | onp.ToArrayND,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: type[int],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> onp.ArrayND[np.int_]: ...
@overload
def geometric_transform(
    input: onp.ToScalar | onp.ToArrayND,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: type[float],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> onp.ArrayND[np.float64 | np.int_]: ...
@overload
def geometric_transform(
    input: onp.ToScalar | onp.ToArrayND,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: type[complex],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> onp.ArrayND[np.complex128 | np.float64 | np.int_]: ...

#
@overload
def map_coordinates(
    input: onp.ToFloat | onp.ToFloatND,
    coordinates: onp.ToFloat | onp.ToFloatND,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
) -> _FloatArrayOut: ...
@overload
def map_coordinates(
    input: onp.ToComplex | onp.ToComplexND,
    coordinates: onp.ToFloat | onp.ToFloatND,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> _ComplexArrayOut: ...
@overload
def map_coordinates(
    input: onp.ToScalar | onp.ToArrayND,
    coordinates: onp.ToFloat | onp.ToFloatND,
    output: _ArrayOrDType[_ScalarT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[_ScalarT]: ...
@overload
def map_coordinates(
    input: onp.ToScalar | onp.ToArrayND,
    coordinates: onp.ToFloat | onp.ToFloatND,
    output: type[bool],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[np.bool_]: ...
@overload
def map_coordinates(
    input: onp.ToScalar | onp.ToArrayND,
    coordinates: onp.ToFloat | onp.ToFloatND,
    output: type[int],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[np.int_ | np.bool_]: ...
@overload
def map_coordinates(
    input: onp.ToScalar | onp.ToArrayND,
    coordinates: onp.ToFloat | onp.ToFloatND,
    output: type[float],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[np.float64 | np.int_ | np.bool_]: ...
@overload
def map_coordinates(
    input: onp.ToScalar | onp.ToArrayND,
    coordinates: onp.ToFloat | onp.ToFloatND,
    output: type[complex],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[np.complex128 | np.float64 | np.int_ | np.bool_]: ...

#
@overload
def affine_transform(
    input: onp.ToFloat | onp.ToFloatND,
    matrix: onp.ToFloat | onp.ToFloat1D | onp.ToFloat2D,
    offset: onp.ToFloat | onp.ToFloat1D = 0.0,
    output_shape: tuple[int, ...] | None = None,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
) -> _FloatArrayOut: ...
@overload
def affine_transform(
    input: onp.ToComplex | onp.ToComplexND,
    matrix: onp.ToFloat | onp.ToFloat1D | onp.ToFloat2D,
    offset: onp.ToFloat | onp.ToFloat1D = 0.0,
    output_shape: tuple[int, ...] | None = None,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> _ComplexArrayOut: ...
@overload
def affine_transform(
    input: onp.ToScalar | onp.ToArrayND,
    matrix: onp.ToFloat | onp.ToFloat1D | onp.ToFloat2D,
    offset: onp.ToFloat | onp.ToFloat1D = 0.0,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: _ArrayOrDType[_ScalarT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[_ScalarT]: ...
@overload
def affine_transform(
    input: onp.ToScalar | onp.ToArrayND,
    matrix: onp.ToFloat | onp.ToFloat1D | onp.ToFloat2D,
    offset: onp.ToFloat | onp.ToFloat1D = 0.0,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: type[int],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[np.int_]: ...
@overload
def affine_transform(
    input: onp.ToScalar | onp.ToArrayND,
    matrix: onp.ToFloat | onp.ToFloat1D | onp.ToFloat2D,
    offset: onp.ToFloat | onp.ToFloat1D = 0.0,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: type[float],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[np.float64 | np.int_]: ...
@overload
def affine_transform(
    input: onp.ToScalar | onp.ToArrayND,
    matrix: onp.ToFloat | onp.ToFloat1D | onp.ToFloat2D,
    offset: onp.ToFloat | onp.ToFloat1D = 0.0,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: type[complex],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[np.complex128 | np.float64 | np.int_]: ...

#
@overload
def shift(
    input: onp.ToFloat | onp.ToFloatND,
    shift: onp.ToFloat | onp.ToFloatND,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
) -> _FloatArrayOut: ...
@overload
def shift(
    input: onp.ToComplex | onp.ToComplexND,
    shift: onp.ToFloat | onp.ToFloatND,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> _ComplexArrayOut: ...
@overload
def shift(
    input: onp.ToScalar | onp.ToArrayND,
    shift: onp.ToFloat | onp.ToFloatND,
    output: _ArrayOrDType[_ScalarT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[_ScalarT]: ...
@overload
def shift(
    input: onp.ToScalar | onp.ToArrayND,
    shift: onp.ToFloat | onp.ToFloatND,
    output: type[int],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[np.int_]: ...
@overload
def shift(
    input: onp.ToScalar | onp.ToArrayND,
    shift: onp.ToFloat | onp.ToFloatND,
    output: type[float],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[np.float64 | np.int_]: ...
@overload
def shift(
    input: onp.ToScalar | onp.ToArrayND,
    shift: onp.ToFloat | onp.ToFloatND,
    output: type[complex],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[np.complex128 | np.float64 | np.int_]: ...

#
@overload
def zoom(
    input: onp.ToFloat | onp.ToFloatND,
    zoom: onp.ToFloat | onp.ToFloatND,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
    *,
    grid_mode: bool = False,
) -> _FloatArrayOut: ...
@overload
def zoom(
    input: onp.ToComplex | onp.ToComplexND,
    zoom: onp.ToFloat | onp.ToFloatND,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
    *,
    grid_mode: bool = False,
) -> _ComplexArrayOut: ...
@overload
def zoom(
    input: onp.ToScalar | onp.ToArrayND,
    zoom: onp.ToFloat | onp.ToFloatND,
    output: _ArrayOrDType[_ScalarT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
    *,
    grid_mode: bool = False,
) -> onp.ArrayND[_ScalarT]: ...
@overload
def zoom(
    input: onp.ToScalar | onp.ToArrayND,
    zoom: onp.ToFloat | onp.ToFloatND,
    output: type[int],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
    *,
    grid_mode: bool = False,
) -> onp.ArrayND[np.int_]: ...
@overload
def zoom(
    input: onp.ToScalar | onp.ToArrayND,
    zoom: onp.ToFloat | onp.ToFloatND,
    output: type[float],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
    *,
    grid_mode: bool = False,
) -> onp.ArrayND[np.float64 | np.int_]: ...
@overload
def zoom(
    input: onp.ToScalar | onp.ToArrayND,
    zoom: onp.ToFloat | onp.ToFloatND,
    output: type[complex],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
    *,
    grid_mode: bool = False,
) -> onp.ArrayND[np.complex128 | np.float64 | np.int_]: ...

#
@overload
def rotate(
    input: onp.ToFloat | onp.ToFloatND,
    angle: onp.ToFloat,
    axes: tuple[onp.ToInt, onp.ToInt] = (1, 0),
    reshape: bool = True,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToFloat = 0.0,
    prefilter: onp.ToBool = True,
) -> _FloatArrayOut: ...
@overload
def rotate(
    input: onp.ToComplex | onp.ToComplexND,
    angle: onp.ToFloat,
    axes: tuple[onp.ToInt, onp.ToInt] = (1, 0),
    reshape: bool = True,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> _ComplexArrayOut: ...
@overload
def rotate(
    input: onp.ToScalar | onp.ToArrayND,
    angle: onp.ToFloat,
    axes: tuple[onp.ToInt, onp.ToInt] = (1, 0),
    reshape: bool = True,
    *,
    output: _ArrayOrDType[_ScalarT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[_ScalarT]: ...
@overload
def rotate(
    input: onp.ToScalar | onp.ToArrayND,
    angle: onp.ToFloat,
    axes: tuple[onp.ToInt, onp.ToInt] = (1, 0),
    reshape: bool = True,
    *,
    output: type[int],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[np.int_]: ...
@overload
def rotate(
    input: onp.ToScalar | onp.ToArrayND,
    angle: onp.ToFloat,
    axes: tuple[onp.ToInt, onp.ToInt] = (1, 0),
    reshape: bool = True,
    *,
    output: type[float],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[np.float64 | np.int_]: ...
@overload
def rotate(
    input: onp.ToScalar | onp.ToArrayND,
    angle: onp.ToFloat,
    axes: tuple[onp.ToInt, onp.ToInt] = (1, 0),
    reshape: bool = True,
    *,
    output: type[complex],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: onp.ToComplex = 0.0,
    prefilter: onp.ToBool = True,
) -> onp.ArrayND[np.complex128 | np.float64 | np.int_]: ...
