from typing import Final as _Final, Literal as _Literal
from typing_extensions import LiteralString as _LiteralString

version: _Final[_LiteralString] = ...
full_version: _Final[_LiteralString] = ...
short_version: _Final[_Literal["1.15.0rc2", "1.15.0"]] = ...
git_revision: _Final[_LiteralString] = ...
release: _Final[bool] = ...
