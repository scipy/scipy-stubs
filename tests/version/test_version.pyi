from typing import LiteralString, assert_type

import scipy.version

# this additional assignment avoids having to use the literal string type
v: str = scipy.version.short_version
assert_type(v, str)

assert_type(scipy.version.version, LiteralString)
assert_type(scipy.version.full_version, LiteralString)
assert_type(scipy.version.git_revision, LiteralString)

assert_type(scipy.version.release, bool)
