# type-tests for `stats/_qmc.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.stats import qmc
from scipy.stats.qmc import Halton, LatinHypercube, MultinomialQMC, MultivariateNormalQMC, PoissonDisk, QMCEngine, Sobol

_f64_nd: onp.ArrayND[np.float64]
_f64_2d: onp.Array2D[np.float64]

_engine: QMCEngine[np.float64]

###

qmc.scale(_f64_nd, 0, 1)
qmc.scale(_f64_2d, 0, 1)

assert_type(qmc.discrepancy(_f64_2d), float | np.float64)
assert_type(qmc.discrepancy(_f64_2d, iterative=True, method="MD", workers=2), float | np.float64)

assert_type(qmc.geometric_discrepancy(_f64_2d), float | np.float64)
assert_type(qmc.geometric_discrepancy(_f64_2d, method="mst", metric="cityblock"), float | np.float64)

assert_type(qmc.update_discrepancy(_f64_2d[0], _f64_2d, 0.5), float)

###
# QMCEngine

assert_type(_engine.d, int | np.int32 | np.int64)
assert_type(_engine.rng_seed, np.random.Generator)
assert_type(_engine.num_generated, int)

assert_type(_engine.random(), onp.Array2D[np.float64])
assert_type(_engine.random(8, workers=2), onp.Array2D[np.float64])
assert_type(_engine.integers(0), onp.Array2D[np.int64])
assert_type(_engine.integers(0, u_bounds=10, n=8, endpoint=True, workers=2), onp.Array2D[np.int64])
assert_type(_engine.reset(), QMCEngine[np.float64])
assert_type(_engine.fast_forward(8), QMCEngine[np.float64])

###
# Halton

assert_type(Halton(4), Halton)

assert_type(Halton(d=4), Halton)
assert_type(Halton(4, scramble=False), Halton)
assert_type(Halton(4, rng=0), Halton)
assert_type(Halton(4, seed=0), Halton)

assert_type(Halton(4).base, list[int])
assert_type(Halton(4).scramble, bool)

###
# LatinHypercube

assert_type(LatinHypercube(4), LatinHypercube)

assert_type(LatinHypercube(d=4), LatinHypercube)
assert_type(LatinHypercube(4, scramble=False, strength=2), LatinHypercube)
assert_type(LatinHypercube(4, optimization="random-cd"), LatinHypercube)
assert_type(LatinHypercube(4, rng=0), LatinHypercube)
assert_type(LatinHypercube(4, seed=0), LatinHypercube)

assert_type(LatinHypercube(4).scramble, bool)
assert_type(LatinHypercube(4).random(8), onp.Array2D[np.float64])

###
# Sobol

assert_type(Sobol(4), Sobol)

assert_type(Sobol(d=4), Sobol)
assert_type(Sobol(4, scramble=False, bits=30), Sobol)
assert_type(Sobol(4, optimization="lloyd"), Sobol)
assert_type(Sobol(4, rng=0), Sobol)
assert_type(Sobol(4, seed=0), Sobol)

assert_type(Sobol.MAXDIM, int)
assert_type(Sobol(4).dtype_i, type[np.uint32 | np.uint64])
assert_type(Sobol(4).scramble, bool)
assert_type(Sobol(4).bits, int | npc.integer)
assert_type(Sobol(4).random(8), onp.Array2D[np.float64])
assert_type(Sobol(4).random_base2(3), onp.Array2D[np.float64])

###
# PoissonDisk

assert_type(PoissonDisk(2), PoissonDisk)

assert_type(PoissonDisk(d=2), PoissonDisk)
assert_type(PoissonDisk(2, radius=0.1, hypersphere="surface", ncandidates=10), PoissonDisk)
assert_type(PoissonDisk(2, l_bounds=[0.0, 0.0], u_bounds=[1.0, 1.0]), PoissonDisk)
assert_type(PoissonDisk(2, rng=0), PoissonDisk)
assert_type(PoissonDisk(2, seed=0), PoissonDisk)

assert_type(PoissonDisk(2).radius_factor, float)
assert_type(PoissonDisk(2).l_bounds, onp.Array1D[np.float64])
assert_type(PoissonDisk(2).cell_size, np.float64)
assert_type(PoissonDisk(2).random(8), onp.Array2D[np.float64])
assert_type(PoissonDisk(2).fill_space(), onp.Array2D[np.float64])

###
# MultivariateNormalQMC

assert_type(MultivariateNormalQMC([0.0, 0.0]), MultivariateNormalQMC[Sobol])
assert_type(MultivariateNormalQMC([0.0, 0.0], _f64_2d), MultivariateNormalQMC[Sobol])
assert_type(MultivariateNormalQMC([0.0, 0.0], cov_root=_f64_2d, inv_transform=False), MultivariateNormalQMC[Sobol])
assert_type(MultivariateNormalQMC([0.0, 0.0], rng=0), MultivariateNormalQMC[Sobol])
assert_type(MultivariateNormalQMC([0.0, 0.0], seed=0), MultivariateNormalQMC[Sobol])
assert_type(MultivariateNormalQMC([0.0, 0.0], engine=Halton(2)), MultivariateNormalQMC[Halton])
assert_type(MultivariateNormalQMC([0.0, 0.0], engine=Halton(2), seed=0), MultivariateNormalQMC[Halton])

assert_type(MultivariateNormalQMC([0.0, 0.0]).engine, Sobol)
assert_type(MultivariateNormalQMC([0.0, 0.0]).random(8), onp.Array2D[np.float64])

###
# MultinomialQMC

assert_type(MultinomialQMC([0.5, 0.5], 10), MultinomialQMC[Sobol])
assert_type(MultinomialQMC(0.5, 10), MultinomialQMC[Sobol])
assert_type(MultinomialQMC([0.5, 0.5], 10, rng=0), MultinomialQMC[Sobol])
assert_type(MultinomialQMC([0.5, 0.5], 10, seed=0), MultinomialQMC[Sobol])
assert_type(MultinomialQMC([0.5, 0.5], 10, engine=Halton(2)), MultinomialQMC[Halton])
assert_type(MultinomialQMC([0.5, 0.5], 10, engine=Halton(2), seed=0), MultinomialQMC[Halton])

assert_type(MultinomialQMC([0.5, 0.5], 10).pvals, onp.Array1D[np.float32 | np.float64])
assert_type(MultinomialQMC([0.5, 0.5], 10).n_trials, int | npc.integer)
assert_type(MultinomialQMC([0.5, 0.5], 10).engine, Sobol)
assert_type(MultinomialQMC([0.5, 0.5], 10).random(8), onp.Array2D[np.float64])
