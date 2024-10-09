__all__ = ["aliases", "native_code", "swapped_code", "sys_is_le", "to_numpy_code"]

from typing import Final, Literal

sys_is_le: Final[bool] = ...
native_code: Final[Literal["<", ">"]] = ...
swapped_code: Final[Literal["<", ">"]] = ...
aliases: dict[str, tuple[str, ...]]

def to_numpy_code(code: str) -> Literal["<", ">", "="]: ...
