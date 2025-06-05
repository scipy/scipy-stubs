# NOTE: This private(!) module only exists in `if typing.TYPE_CHECKING: ...` and in `.pyi` stubs

from _typeshed import Incomplete
from collections.abc import Iterator, Sequence
from os import PathLike
from types import TracebackType
from typing import IO, Any, Literal, LiteralString, Protocol, Self, SupportsIndex, TypeAlias, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp

__all__ = [
    "RNG",
    "Alternative",
    "AnyBool",
    "AnyShape",
    "ByteOrder",
    "Casting",
    "ConvMode",
    "EnterNoneMixin",
    "EnterSelfMixin",
    "Falsy",
    "FileLike",
    "FileModeRW",
    "FileModeRWA",
    "FileName",
    "NanPolicy",
    "OrderCF",
    "OrderKACF",
    "SequenceNotStr",
    "ToRNG",
    "Truthy",
    "_FortranFunction",
]

# helper mixins
@type_check_only
class EnterSelfMixin:
    def __enter__(self, /) -> Self: ...
    def __exit__(self, /, type: type[BaseException] | None, value: BaseException | None, tb: TracebackType | None) -> None: ...

@type_check_only
class EnterNoneMixin:
    def __enter__(self, /) -> None: ...
    def __exit__(self, /, type: type[BaseException] | None, value: BaseException | None, tb: TracebackType | None) -> None: ...

# used in `scipy.linalg.blas` and `scipy.linalg.lapack`
@type_check_only
class _FortranFunction(Protocol):
    @property
    def dtype(self, /) -> np.dtype[Incomplete]: ...
    @property
    def int_dtype(self, /) -> np.dtype[np.integer[Any]]: ...
    @property
    def module_name(self, /) -> LiteralString: ...
    @property
    def prefix(self, /) -> LiteralString: ...
    @property
    def typecode(self, /) -> LiteralString: ...
    def __call__(self, /, *args: object, **kwargs: object) -> Incomplete: ...

_VT_co = TypeVar("_VT_co", covariant=True)

# A slightly modified variant from https://github.com/python/typing/issues/256#issuecomment-1442633430
# This works because `str.__contains__` does not accept object (either in typeshed or at runtime)
@type_check_only
class SequenceNotStr(Protocol[_VT_co]):
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> _VT_co: ...
    @overload
    def __getitem__(self, index: slice, /) -> Sequence[_VT_co]: ...
    def __iter__(self, /) -> Iterator[_VT_co]: ...
    def __reversed__(self, /) -> Iterator[_VT_co]: ...
    def __contains__(self, value: object, /) -> bool: ...  # <-- the trick
    def __len__(self, /) -> int: ...
    def index(self, value: object, start: int = 0, stop: int = ..., /) -> int: ...
    def count(self, value: object, /) -> int: ...

# I/O
_ByteSOrStr = TypeVar("_ByteSOrStr", bytes, str)
FileName: TypeAlias = str | PathLike[str]
FileLike: TypeAlias = FileName | IO[_ByteSOrStr]
FileModeRW: TypeAlias = Literal["r", "w"]
FileModeRWA: TypeAlias = Literal[FileModeRW, "a"]

# TODO(jorenham): Include `np.bool[L[False]]` once we have `numpy>=2.2`
Falsy: TypeAlias = Literal[False, 0]
Truthy: TypeAlias = Literal[True, 1]

# keep in sync with `numpy._typing._scalars`
AnyBool: TypeAlias = bool | np.bool_ | Literal[0, 1]

# equivalent to `numpy._typing._shape._ShapeLike`
AnyShape: TypeAlias = op.CanIndex | Sequence[op.CanIndex]

RNG: TypeAlias = np.random.Generator | np.random.RandomState
# NOTE: This is less incorrect and more accurate than the current `np.random.default_rng` `seed` param annotation.
ToRNG: TypeAlias = (
    int
    | np.integer[Any]
    | np.timedelta64
    | onp.ArrayND[np.integer[Any] | np.timedelta64 | np.flexible | np.object_]
    | np.random.SeedSequence
    | np.random.BitGenerator
    | RNG
    | None
)

# numpy literals
ByteOrder: TypeAlias = Literal["S", "<", "little", ">", "big", "=", "native", "|", "I"]
OrderCF: TypeAlias = Literal["C", "F"]
OrderKACF: TypeAlias = Literal["K", "A", OrderCF]
Casting: TypeAlias = Literal["no", "equiv", "safe", "same_kind", "unsafe"]
ConvMode: TypeAlias = Literal["valid", "same", "full"]

# scipy literals
NanPolicy: TypeAlias = Literal["raise", "propagate", "omit"]
Alternative: TypeAlias = Literal["two-sided", "less", "greater"]
NormalizationMode: TypeAlias = Literal["backward", "ortho", "forward"]
DCTType: TypeAlias = Literal[1, 2, 3, 4]
