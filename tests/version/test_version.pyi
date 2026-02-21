from optype.test import assert_subtype

import scipy.version

assert_subtype[str](scipy.version.short_version)
assert_subtype[str](scipy.version.git_revision)
assert_subtype[bool](scipy.version.release)
