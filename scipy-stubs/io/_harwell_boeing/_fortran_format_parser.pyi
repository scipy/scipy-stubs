import abc
import re
from typing import Final, Generic, Literal, TypeAlias, TypeVar, type_check_only
from typing_extensions import Self

import optype.numpy as onp

__all__ = ["BadFortranFormat", "ExpFormat", "FortranFormatParser", "IntFormat"]

_NumberT = TypeVar("_NumberT", int | bool, float | int | bool)
_TokenType: TypeAlias = Literal["INT", "INT_ID", "EXP_ID", "DOT", "LPAR", "RPAR"]

TOKENS: Final[dict[_TokenType, str]]

class BadFortranFormat(SyntaxError): ...

@type_check_only
class _NumberFormat(Generic[_NumberT], metaclass=abc.ABCMeta):
    width: Final[int | bool]
    repeat: Final[int | bool | None]
    min: Final[int | bool | None]
    @property
    def fortran_format(self, /) -> str: ...
    @property
    def python_format(self, /) -> str: ...
    @classmethod
    def from_number(cls, n: _NumberT, min: int | bool | None = None) -> Self: ...

class IntFormat(_NumberFormat[int | bool]):
    def __init__(self, /, width: int | bool, min: int | bool | None = None, repeat: int | bool | None = None) -> None: ...

class ExpFormat(_NumberFormat[float | int | bool]):
    significand: Final[int | bool]
    def __init__(
        self, /, width: int | bool, significand: int | bool, min: int | bool | None = None, repeat: int | bool | None = None
    ) -> None: ...

class Token:
    type: Final[_TokenType]
    value: Final[str]
    pos: Final[int | bool]
    def __init__(self, /, type: _TokenType, value: str, pos: int | bool) -> None: ...

class Tokenizer:
    tokens: Final[list[_TokenType]]
    res: Final[list[re.Pattern[str]]]
    data: str
    curpos: int | bool
    len: int | bool
    def input(self, /, s: str) -> None: ...
    def next_token(self, /) -> Token: ...

class FortranFormatParser:
    tokenizer: Final[Tokenizer]
    def parse(self, /, s: str) -> IntFormat | ExpFormat: ...

def number_digits(n: onp.ToInt) -> int | bool: ...
