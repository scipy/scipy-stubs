import numpy as np

from scipy.special import logsumexp

# https://github.com/scipy/scipy-stubs/issues/697
x = np.asarray([1, 2, 3], dtype=np.float64)
logsumexp(x)
