from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal.windows import (
    barthann,
    bartlett,
    blackman,
    blackmanharris,
    bohman,
    boxcar,
    chebwin,
    cosine,
    dpss,
    exponential,
    flattop,
    gaussian,
    general_cosine,
    general_gaussian,
    general_hamming,
    get_window,
    hamming,
    hann,
    kaiser,
    kaiser_bessel_derived,
    lanczos,
    nuttall,
    parzen,
    taylor,
    triang,
    tukey,
)

###

# get_window
assert_type(get_window("hann", 64), onp.Array1D[np.float64])
assert_type(get_window("hann", 64, xp=np), Any)

# barthann
assert_type(barthann(64), onp.Array1D[np.float64])
assert_type(barthann(64, xp=np), Any)

# bartlett
assert_type(bartlett(64), onp.Array1D[np.float64])
assert_type(bartlett(64, xp=np), Any)

# blackman
assert_type(blackman(64), onp.Array1D[np.float64])
assert_type(blackman(64, xp=np), Any)

# blackmanharris
assert_type(blackmanharris(64), onp.Array1D[np.float64])
assert_type(blackmanharris(64, xp=np), Any)

# bohman
assert_type(bohman(64), onp.Array1D[np.float64])
assert_type(bohman(64, xp=np), Any)

# boxcar
assert_type(boxcar(64), onp.Array1D[np.float64])
assert_type(boxcar(64, xp=np), Any)

# cosine
assert_type(cosine(64), onp.Array1D[np.float64])
assert_type(cosine(64, xp=np), Any)

# flattop
assert_type(flattop(64), onp.Array1D[np.float64])
assert_type(flattop(64, xp=np), Any)

# hamming
assert_type(hamming(64), onp.Array1D[np.float64])
assert_type(hamming(64, xp=np), Any)

# hann
assert_type(hann(64), onp.Array1D[np.float64])
assert_type(hann(64, xp=np), Any)

# lanczos
assert_type(lanczos(64), onp.Array1D[np.float64])
assert_type(lanczos(64, xp=np), Any)

# nuttall
assert_type(nuttall(64), onp.Array1D[np.float64])
assert_type(nuttall(64, xp=np), Any)

# parzen
assert_type(parzen(64), onp.Array1D[np.float64])
assert_type(parzen(64, xp=np), Any)

# triang
assert_type(triang(64), onp.Array1D[np.float64])
assert_type(triang(64, xp=np), Any)

# chebwin
assert_type(chebwin(64, 100.0), onp.Array1D[np.float64])
assert_type(chebwin(64, 100.0, xp=np), Any)

# gaussian
assert_type(gaussian(64, 3.0), onp.Array1D[np.float64])
assert_type(gaussian(64, 3.0, xp=np), Any)

# general_hamming
assert_type(general_hamming(64, 0.5), onp.Array1D[np.float64])
assert_type(general_hamming(64, 0.5, xp=np), Any)

# kaiser
assert_type(kaiser(64, 3.0), onp.Array1D[np.float64])
assert_type(kaiser(64, 3.0, xp=np), Any)

# kaiser_bessel_derived
assert_type(kaiser_bessel_derived(64, 3.0), onp.Array1D[np.float64])
assert_type(kaiser_bessel_derived(64, 3.0, xp=np), Any)

# tukey
assert_type(tukey(64), onp.Array1D[np.float64])
assert_type(tukey(64, alpha=0.25, xp=np), Any)

# general_cosine
assert_type(general_cosine(64, np.ones(3)), onp.Array1D[np.float64])

# exponential
assert_type(exponential(64), onp.Array1D[np.float64])
assert_type(exponential(64, center=0.0, tau=2.0), onp.Array1D[np.float64])
assert_type(exponential(64, xp=np), Any)

# general_gaussian
assert_type(general_gaussian(64, 1.5, 2.0), onp.Array1D[np.float64])
assert_type(general_gaussian(64, 1.5, 2.0, xp=np), Any)

# taylor
assert_type(taylor(64), onp.Array1D[np.float64])
assert_type(taylor(64, nbar=6, sll=25, norm=False, sym=False), onp.Array1D[np.float64])
assert_type(taylor(64, xp=np), Any)

# ... other window functions ...

# dpss
assert_type(dpss(64, 3), onp.Array1D[np.float64])
assert_type(dpss(64, 3, 2), onp.Array2D[np.float64])
assert_type(dpss(64, 3, return_ratios=True), tuple[onp.Array1D[np.float64], np.float64])
assert_type(dpss(64, 3, 2, return_ratios=True), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64]])
assert_type(dpss(64, 3, 2, xp=np), tuple[Any, Any])
