from typing_extensions import Self

import numpy.typing as npt
import optype.numpy as onpt

__all__ = ["CensoredData"]

class CensoredData:
    def __init__(
        self,
        uncensored: npt.ArrayLike | None = None,
        *,
        left: npt.ArrayLike | None = None,
        right: npt.ArrayLike | None = None,
        interval: npt.ArrayLike | None = None,
    ) -> None: ...
    def __sub__(self, other: object, /) -> CensoredData: ...
    def __truediv__(self, other: object, /) -> CensoredData: ...
    def __len__(self) -> int: ...

    def num_censored(self) -> int: ...

    @classmethod
    def right_censored(cls, x: npt.ArrayLike, censored: onpt.AnyBoolArray) -> Self: ...
    @classmethod
    def left_censored(cls, x: npt.ArrayLike, censored: onpt.AnyBoolArray) -> Self: ...
    @classmethod
    def interval_censored(cls, low: npt.ArrayLike, high: npt.ArrayLike) -> Self: ...
