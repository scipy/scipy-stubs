# NOTE(scipy-stubs): This private module should not be used outside of scipy-stubs

from collections.abc import Sequence
from types import TracebackType
from typing import Any, Protocol, SupportsIndex, TypeAlias, final, type_check_only
from typing_extensions import TypeVar

import numpy as np

__all__ = "AnyShape", "CanArrayND", "ExitMixin"

# helper mixins
@type_check_only
class ExitMixin:
    @final
    def __exit__(self, /, type: type[BaseException] | None, value: BaseException | None, tb: TracebackType | None) -> None: ...

# equivalent to `numpy._typing._shape._ShapeLike`
AnyShape: TypeAlias = SupportsIndex | Sequence[SupportsIndex]

# NOTE: For some reason, `onp.CanArrayND` isn't understood by pyright when running `uv run pyright tests`, even though it works
# fine when running `uv run pyright` in the root directory (same story for basedpyright). By copying the definition here, these
# Pyright won't report false positives (no idea why though), so this is but a workaround.
# https://github.com/jorenham/optype/blob/abf1758/optype/numpy/_array.py#L124-L133
# TODO(jorenham): Remove this workaround once the issue is fixed in Pyright.

_SCT_co = TypeVar("_SCT_co", bound=np.generic, covariant=True)
_NDT_co = TypeVar("_NDT_co", bound=tuple[int, ...], default=tuple[Any, ...], covariant=True)

@type_check_only
class CanArrayND(Protocol[_SCT_co, _NDT_co]):
    def __len__(self, /) -> int: ...
    def __array__(self, /) -> np.ndarray[_NDT_co, np.dtype[_SCT_co]]: ...
