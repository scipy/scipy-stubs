# type-tests for `spatial/transform/_rotation.pyi`

from typing import Any, Literal, assert_type

import numpy as np
import optype.numpy as onp

from scipy.spatial.transform import Rotation, Slerp

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_1d: tuple[int]
_2d: tuple[int, int]
_3d: tuple[int, int, int]

###
# Rotation

_rot_nd: Rotation
_rot_0d: Rotation[tuple[()]]
_rot_1d: Rotation[tuple[int]]
_rot_2d: Rotation[tuple[int, int]]

# __init__

assert_type(Rotation(_f64_1d), Rotation[tuple[()]])
assert_type(Rotation(_f64_2d), Rotation[tuple[int]])
assert_type(Rotation(_f64_nd), Rotation[tuple[Any, ...]])

# from_*

assert_type(Rotation.from_quat(_f64_1d), Rotation[tuple[()]])
assert_type(Rotation.from_quat(_f64_2d), Rotation[tuple[int]])
assert_type(Rotation.from_quat(_f64_nd), Rotation)

assert_type(Rotation.from_matrix(_f64_2d), Rotation[tuple[()]])
assert_type(Rotation.from_matrix(_f64_3d), Rotation[tuple[int]])

assert_type(Rotation.from_rotvec(_f64_1d), Rotation[tuple[()]])
assert_type(Rotation.from_rotvec(_f64_2d), Rotation[tuple[int]])
assert_type(Rotation.from_rotvec(_f64_nd), Rotation)

assert_type(Rotation.from_mrp(_f64_1d), Rotation[tuple[()]])
assert_type(Rotation.from_mrp(_f64_2d), Rotation[tuple[int]])
assert_type(Rotation.from_mrp(_f64_nd), Rotation)

assert_type(Rotation.from_euler("xyz", 0.0), Rotation[tuple[()]])
assert_type(Rotation.from_euler("xyz", _f64_1d), Rotation[tuple[int]])
assert_type(Rotation.from_euler("xyz", _f64_2d), Rotation[tuple[int, int]])
assert_type(Rotation.from_euler("xyz", _f64_nd), Rotation)

assert_type(Rotation.from_davenport(_f64_1d, "extrinsic", 0.0), Rotation[tuple[()]])
assert_type(Rotation.from_davenport(_f64_2d, "intrinsic", _f64_1d), Rotation[tuple[int]])
assert_type(Rotation.from_davenport(_f64_2d, "intrinsic", _f64_2d), Rotation[tuple[int, int]])
assert_type(Rotation.from_davenport(_f64_2d, "intrinsic", _f64_nd), Rotation)

# identity

assert_type(Rotation.identity(), Rotation[tuple[()]])
assert_type(Rotation.identity(4), Rotation[tuple[int]])
assert_type(Rotation.identity(shape=_1d), Rotation[tuple[int]])
assert_type(Rotation.identity(shape=_2d), Rotation[tuple[int, int]])
assert_type(Rotation.identity(shape=_3d), Rotation[tuple[int, int, int]])

# random

assert_type(Rotation.random(), Rotation[tuple[()]])
assert_type(Rotation.random(4), Rotation[tuple[int]])
assert_type(Rotation.random(shape=_1d), Rotation[tuple[int]])
assert_type(Rotation.random(shape=_2d), Rotation[tuple[int, int]])
assert_type(Rotation.random(shape=_3d), Rotation[tuple[int, int, int]])

# concatenate

assert_type(Rotation.concatenate(_rot_nd), Rotation)
assert_type(Rotation.concatenate([_rot_0d, _rot_0d]), Rotation[tuple[int]])
assert_type(Rotation.concatenate([_rot_1d, _rot_1d]), Rotation[tuple[int, int]])
assert_type(Rotation.concatenate([_rot_nd, _rot_nd]), Rotation)  # type: ignore[assert-type]

# create_group

assert_type(Rotation.create_group("I"), Rotation[tuple[()]])
assert_type(Rotation.create_group("Dn", "X"), Rotation[tuple[()]])

# properties

assert_type(_rot_nd.single, bool)
assert_type(_rot_nd.shape, tuple[Any, ...])

# dunders

assert_type(_rot_nd.__bool__(), Literal[True])  # noqa: PLC2801
assert_type(len(_rot_nd), int)
assert_type(_rot_nd * _rot_nd, Rotation)
assert_type(_rot_nd**2.0, Rotation)
assert_type(_rot_nd[0], Rotation)
assert_type(_rot_nd[0:2], Rotation)
assert_type(next(iter(_rot_nd)), Rotation[tuple[()]])  # type: ignore[assert-type]

# inv

assert_type(_rot_0d.inv(), Rotation[tuple[()]])
assert_type(_rot_1d.inv(), Rotation[tuple[int]])
assert_type(_rot_nd.inv(), Rotation)

# as_quat

assert_type(_rot_0d.as_quat(), onp.Array1D[np.float64])
assert_type(_rot_1d.as_quat(), onp.Array2D[np.float64])
assert_type(_rot_nd.as_quat(), onp.ArrayND[np.float64])

# as_matrix

assert_type(_rot_0d.as_matrix(), onp.Array2D[np.float64])
assert_type(_rot_1d.as_matrix(), onp.Array3D[np.float64])
assert_type(_rot_nd.as_matrix(), onp.ArrayND[np.float64])

# as_rotvec

assert_type(_rot_0d.as_rotvec(), onp.Array1D[np.float64])
assert_type(_rot_1d.as_rotvec(), onp.Array2D[np.float64])
assert_type(_rot_nd.as_rotvec(), onp.ArrayND[np.float64])

# as_mrp

assert_type(_rot_0d.as_mrp(), onp.Array1D[np.float64])
assert_type(_rot_1d.as_mrp(), onp.Array2D[np.float64])
assert_type(_rot_nd.as_mrp(), onp.ArrayND[np.float64])

# as_euler

assert_type(_rot_0d.as_euler("xyz"), onp.Array1D[np.float64])
assert_type(_rot_1d.as_euler("xyz"), onp.Array2D[np.float64])
assert_type(_rot_nd.as_euler("xyz"), onp.ArrayND[np.float64])

# mean

assert_type(_rot_nd.mean(), Rotation[tuple[()]])
assert_type(_rot_nd.mean(axis=0), Rotation)

# magnitude

assert_type(_rot_0d.magnitude(), np.float64)
assert_type(_rot_1d.magnitude(), onp.Array1D[np.float64])
assert_type(_rot_2d.magnitude(), onp.Array2D[np.float64])
assert_type(_rot_nd.magnitude(), np.float64 | onp.ArrayND[np.float64])

# apply

assert_type(_rot_0d.apply(_f64_1d), onp.Array1D[np.float64])
assert_type(_rot_1d.apply(_f64_2d), onp.Array2D[np.float64])
assert_type(_rot_nd.apply(_f64_nd), onp.ArrayND[np.float64])

# approx_equal

assert_type(_rot_0d.approx_equal(_rot_0d), onp.ArrayND[np.bool_])
assert_type(_rot_1d.approx_equal(_rot_1d), onp.ArrayND[np.bool_])
assert_type(_rot_nd.approx_equal(_rot_nd), onp.ArrayND[np.bool_])

# as_* with known shapes

assert_type(_rot_0d.as_quat(), onp.Array1D[np.float64])
assert_type(_rot_0d.as_matrix(), onp.Array2D[np.float64])
assert_type(_rot_0d.as_rotvec(), onp.Array1D[np.float64])
assert_type(_rot_0d.as_mrp(), onp.Array1D[np.float64])
assert_type(_rot_0d.as_euler("xyz"), onp.Array1D[np.float64])
assert_type(_rot_0d.as_davenport(_f64_1d, "extrinsic"), onp.Array1D[np.float64])

assert_type(_rot_1d.as_quat(), onp.Array2D[np.float64])
assert_type(_rot_1d.as_matrix(), onp.Array3D[np.float64])
assert_type(_rot_1d.as_rotvec(), onp.Array2D[np.float64])
assert_type(_rot_1d.as_mrp(), onp.Array2D[np.float64])
assert_type(_rot_1d.as_euler("xyz"), onp.Array2D[np.float64])
assert_type(_rot_1d.as_davenport(_f64_1d, "extrinsic"), onp.Array2D[np.float64])

# magnitude with known shapes

assert_type(_rot_0d.magnitude(), np.float64)
assert_type(_rot_1d.magnitude(), onp.Array1D[np.float64])
assert_type(_rot_2d.magnitude(), onp.Array2D[np.float64])

# apply with known shapes

assert_type(_rot_0d.apply(_f64_1d), onp.Array1D[np.float64])
assert_type(_rot_1d.apply(_f64_2d), onp.Array2D[np.float64])

# mean with axis

assert_type(_rot_nd.mean(axis=0), Rotation)

# reduce

assert_type(_rot_nd.reduce(), Rotation)
assert_type(_rot_nd.reduce(left=_rot_nd), Rotation)

_reduced_with_idx = _rot_nd.reduce(return_indices=True)
assert_type(_reduced_with_idx[0], Rotation)
assert_type(_reduced_with_idx[1], onp.ArrayND[np.int32 | np.int64])
assert_type(_reduced_with_idx[2], onp.ArrayND[np.int32 | np.int64])

# align_vectors

_aligned = Rotation.align_vectors(_f64_2d, _f64_2d)
assert_type(_aligned[0], Rotation[tuple[()]])
assert_type(_aligned[1], float)

_aligned_sens = Rotation.align_vectors(_f64_2d, _f64_2d, return_sensitivity=True)
assert_type(_aligned_sens[0], Rotation[tuple[()]])
assert_type(_aligned_sens[1], float)
assert_type(_aligned_sens[2], onp.Array2D[np.float64])

###
# Slerp

_slerp = Slerp(_f64_1d, _rot_nd)
assert_type(_slerp.times, onp.Array1D[np.int32 | np.int64 | np.float32 | np.float64])
assert_type(_slerp.timedelta, onp.Array1D[np.int32 | np.int64 | np.float32 | np.float64])
assert_type(_slerp.rotations, Rotation)
assert_type(_slerp.rotvecs, onp.Array2D[np.float64])
assert_type(_slerp(_f64_1d), Rotation)
