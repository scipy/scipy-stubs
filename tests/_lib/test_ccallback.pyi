import ctypes as ct
from ctypes import _NamedFuncPointer
from typing import assert_type

from scipy import LowLevelCallable

_cdll: ct.CDLL
_void_p: ct.c_void_p

###
# LowLevelCallable

assert_type(LowLevelCallable(_cdll.f), LowLevelCallable[_NamedFuncPointer, None])
assert_type(LowLevelCallable(_cdll.f, _void_p), LowLevelCallable[_NamedFuncPointer, ct.c_void_p])
