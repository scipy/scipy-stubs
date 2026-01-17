import io
from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.io.arff import MetaData, loadarff

###

str_literal: str
str_io: io.StringIO

###

assert_type(loadarff(str_literal), tuple[onp.Array1D[np.void], MetaData])
assert_type(loadarff(str_io), tuple[onp.Array1D[np.void], MetaData])
