from collections.abc import Callable, Sequence
from typing import Literal, Never, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = [
    "braycurtis",
    "canberra",
    "cdist",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "directed_hausdorff",
    "euclidean",
    "hamming",
    "is_valid_dm",
    "is_valid_y",
    "jaccard",
    "jensenshannon",
    "mahalanobis",
    "minkowski",
    "num_obs_dm",
    "num_obs_y",
    "pdist",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalsneath",
    "sqeuclidean",
    "squareform",
    "yule",
]

_MetricName: TypeAlias = Literal[
    "braycurtis",
    "canberra",
    "chebychev",
    "chebyshev",
    "cheby",
    "cheb",
    "ch",
    "cityblock",
    "cblock",
    "cb",
    "c",
    "correlation",
    "co",
    "cosine",
    "cos",
    "dice",
    "euclidean",
    "euclid",
    "eu",
    "e",
    "hamming",
    "hamm",
    "ha",
    "h",
    "minkowski",
    "mi",
    "m",
    "pnorm",
    "jaccard",
    "jacc",
    "ja",
    "j",
    "jensenshannon",
    "js",
    "mahalanobis",
    "mahal",
    "mah",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "se",
    "s",
    "sokalsneath",
    "sqeuclidean",
    "sqe",
    "sqeuclid",
    "yule",
]

###

_NumberT = TypeVar("_NumberT", bound=npc.number)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

_MetricFunc: TypeAlias = Callable[[onp.Array1D[np.float64], onp.Array1D[np.float64]], onp.ToFloat | None]
_Metric: TypeAlias = _MetricName | _MetricFunc  # noqa: PYI047

_Force: TypeAlias = Literal["NO", "No", "no", "TOMATRIX", "ToMatrix", "tomatrix", "TOVECTOR", "ToVector", "tovector"]

# workaround for mypy & pyright's failure to conform to the overload typing specification
_JustAnyShape: TypeAlias = tuple[Never, Never, Never, Never]

_ToFloatStrictND: TypeAlias = onp.ArrayND[npc.floating | npc.integer, _JustAnyShape]

###

# NOTE: on numpy<2.1, both mypy and pyright reports false positive overload-overlap errors for overload 2 of `jensenshannon`
# pyright: reportOverlappingOverload=false
# mypy: disable-error-code=overload-overlap

# TODO(@jorenham): metric-specific overloads
# https://github.com/scipy/scipy-stubs/issues/404
@overload
def cdist(
    XA: onp.ToFloat2D,
    XB: onp.ToFloat2D,
    metric: _MetricName = "euclidean",
    *,
    out: onp.Array2D[np.float64] | None = None,
    p: float = 2,
    w: onp.ToFloat1D | None = None,
    V: onp.ToFloat2D | None = None,
    VI: onp.ToFloat2D | None = None,
) -> onp.Array2D[np.float64]: ...
@overload
def cdist(
    XA: onp.ToFloat2D, XB: onp.ToFloat2D, metric: _MetricFunc, *, out: onp.Array2D[np.float64] | None = None, **kwds: object
) -> onp.Array2D[np.float64]: ...

# TODO(@jorenham): metric-specific overloads
# https://github.com/scipy/scipy-stubs/issues/404
@overload
def pdist(
    X: onp.ToFloat2D,
    metric: _MetricName = "euclidean",
    *,
    out: onp.Array1D[np.float64] | None = None,
    p: float = 2,
    w: onp.ToFloat1D | None = None,
    V: onp.ToFloat2D | None = None,
    VI: onp.ToFloat2D | None = None,
) -> onp.Array1D[np.float64]: ...
@overload
def pdist(
    X: onp.ToFloat2D, metric: _MetricFunc, *, out: onp.Array1D[np.float64] | None = None, **kwargs: object
) -> onp.Array1D[np.float64]: ...

#
@overload  # ?d T@number
def squareform(X: onp.ArrayND[_NumberT, _JustAnyShape], force: _Force = "no", checks: bool = True) -> onp.ArrayND[_NumberT]: ...
@overload  # 1d +int
def squareform(X: Sequence[int], force: _Force = "no", checks: bool = True) -> onp.Array2D[np.int_]: ...
@overload  # 1d ~float
def squareform(X: list[float], force: _Force = "no", checks: bool = True) -> onp.Array2D[np.float64]: ...
@overload  # 1d ~complex
def squareform(X: list[complex], force: _Force = "no", checks: bool = True) -> onp.Array2D[np.complex128]: ...
@overload  # 1d T@number
def squareform(
    X: onp.ToArrayStrict1D[_NumberT, _NumberT], force: _Force = "no", checks: bool = True
) -> onp.Array2D[_NumberT]: ...
@overload  # 2d +int
def squareform(X: Sequence[Sequence[int]], force: _Force = "no", checks: bool = True) -> onp.Array1D[np.int_]: ...
@overload  # 2d ~float
def squareform(X: Sequence[list[float]], force: _Force = "no", checks: bool = True) -> onp.Array1D[np.float64]: ...
@overload  # 2d ~complex
def squareform(X: Sequence[list[complex]], force: _Force = "no", checks: bool = True) -> onp.Array1D[np.complex128]: ...
@overload  # 2d T@number
def squareform(
    X: onp.ToArrayStrict2D[_NumberT, _NumberT], force: _Force = "no", checks: bool = True
) -> onp.Array1D[_NumberT]: ...
@overload  # fallback
def squareform(X: onp.ToComplexND, force: _Force = "no", checks: bool = True) -> onp.ArrayND: ...

#
def correlation(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None, centered: bool = True) -> np.float64: ...

#
def cosine(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...

#
@overload
def mahalanobis(u: onp.ToFloat1D, v: onp.ToFloat1D, VI: onp.ToFloat2D) -> np.float64: ...
@overload
def mahalanobis(u: onp.ToJustComplex1D, v: onp.ToComplex1D, VI: onp.ToComplex2D) -> np.complex128: ...
@overload
def mahalanobis(u: onp.ToComplex1D, v: onp.ToJustComplex1D, VI: onp.ToComplex2D) -> np.complex128: ...
@overload
def mahalanobis(u: onp.ToComplex1D, v: onp.ToComplex1D, VI: onp.ToJustComplex2D) -> np.complex128: ...
@overload
def mahalanobis(u: onp.ToComplex1D, v: onp.ToComplex1D, VI: onp.ToComplex2D) -> np.float64 | np.complex128: ...

#
@overload
def sokalsneath(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...
@overload
def sokalsneath(u: onp.ToJustComplex1D, v: onp.ToComplex1D, w: onp.ToFloat1D | None = None) -> np.complex128: ...
@overload
def sokalsneath(u: onp.ToComplex1D, v: onp.ToJustComplex1D, w: onp.ToFloat1D | None = None) -> np.complex128: ...
@overload
def sokalsneath(u: onp.ToComplex1D, v: onp.ToComplex1D, w: onp.ToFloat1D | None = None) -> np.float64 | np.complex128: ...

#
@overload
def sqeuclidean(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...
@overload
def sqeuclidean(u: onp.ToJustComplex1D, v: onp.ToComplex1D, w: onp.ToFloat1D | None = None) -> np.complex128: ...
@overload
def sqeuclidean(u: onp.ToComplex1D, v: onp.ToJustComplex1D, w: onp.ToFloat1D | None = None) -> np.complex128: ...
@overload
def sqeuclidean(u: onp.ToComplex1D, v: onp.ToComplex1D, w: onp.ToFloat1D | None = None) -> np.float64 | np.complex128: ...

#
@overload  # ?d, keepdims=False
def jensenshannon(
    p: _ToFloatStrictND, q: _ToFloatStrictND, base: float | None = None, *, axis: int = 0, keepdims: Literal[False] = False
) -> np.float64 | onp.ArrayND[np.float64]: ...
@overload  # ?d, keepdims=True
def jensenshannon(
    p: onp.ArrayND[npc.floating | npc.integer, _ShapeT],
    q: onp.ArrayND[npc.floating | npc.integer, _ShapeT],
    base: float | None = None,
    *,
    axis: int = 0,
    keepdims: Literal[True],
) -> onp.ArrayND[np.float64, _ShapeT]: ...
@overload  # 1d, keepdims=False
def jensenshannon(
    p: onp.ToFloatStrict1D, q: onp.ToFloatStrict1D, base: float | None = None, *, axis: int = 0, keepdims: Literal[False] = False
) -> np.float64: ...
@overload  # 1d, keepdims=True
def jensenshannon(
    p: onp.ToFloatStrict1D, q: onp.ToFloatStrict1D, base: float | None = None, *, axis: int = 0, keepdims: Literal[True]
) -> onp.Array1D[np.float64]: ...
@overload  # 2d, keepdims=False
def jensenshannon(
    p: onp.ToFloatStrict2D, q: onp.ToFloatStrict2D, base: float | None = None, *, axis: int = 0, keepdims: Literal[False] = False
) -> onp.Array1D[np.float64]: ...
@overload  # 2d, keepdims=True
def jensenshannon(
    p: onp.ToFloatStrict2D, q: onp.ToFloatStrict2D, base: float | None = None, *, axis: int = 0, keepdims: Literal[True]
) -> onp.Array2D[np.float64]: ...
@overload  # nd, keepdims=False
def jensenshannon(
    p: onp.ToFloatND, q: onp.ToFloatND, base: float | None = None, *, axis: int = 0, keepdims: Literal[False] = False
) -> np.float64 | onp.ArrayND[np.float64]: ...
@overload  # nd, keepdims=True
def jensenshannon(
    p: onp.ToFloatND, q: onp.ToFloatND, base: float | None = None, *, axis: int = 0, keepdims: Literal[True]
) -> onp.ArrayND[np.float64]: ...

# NOTE: These technically also accept complex inputs, but will unsafely downcast to float and emit a warning, so we disallow it.
def braycurtis(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...
def canberra(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...
def chebyshev(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...
def cityblock(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...
def dice(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> float: ...
def directed_hausdorff(
    u: onp.ToFloat2D, v: onp.ToFloat2D, rng: onp.random.ToRNG | None = 0, *, seed: onp.random.ToRNG | None = None
) -> tuple[float, int, int]: ...
def hamming(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...
def euclidean(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> float: ...
def jaccard(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToBool1D | None = None) -> np.float64: ...
def minkowski(u: onp.ToFloat1D, v: onp.ToFloat1D, p: float = 2, w: onp.ToFloat1D | None = None) -> float: ...
def rogerstanimoto(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> float: ...
def russellrao(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> float: ...
def seuclidean(u: onp.ToFloat1D, v: onp.ToFloat1D, V: onp.ToFloat1D) -> float: ...
def yule(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> float: ...

#
def num_obs_dm(d: onp.ToArray2D) -> int: ...
def num_obs_y(Y: onp.ToArray1D) -> int: ...
def is_valid_dm(D: onp.ToArray2D, tol: float = 0.0, throw: bool = False, name: str = "D", warning: bool = False) -> bool: ...
def is_valid_y(y: onp.ToArray1D, warning: bool = False, throw: bool = False, name: str | None = None) -> bool: ...
