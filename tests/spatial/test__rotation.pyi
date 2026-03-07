# type-tests for `spatial/transform/_rotation.pyi`

from typing import Literal, assert_type

import numpy as np
import optype.numpy as onp

from scipy.spatial.transform import Rotation, Slerp

###
# Rotation

_rot: Rotation
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]

assert_type(Rotation(_f64_1d), Rotation)
assert_type(Rotation(_f64_2d, normalize=False, copy=False, scalar_first=True), Rotation)

assert_type(Rotation.from_quat(_f64_1d), Rotation)
assert_type(Rotation.from_quat(_f64_2d, scalar_first=True), Rotation)
assert_type(Rotation.from_matrix(_f64_2d), Rotation)
assert_type(Rotation.from_matrix(_f64_3d, assume_valid=True), Rotation)
assert_type(Rotation.from_rotvec(_f64_1d), Rotation)
assert_type(Rotation.from_rotvec(_f64_2d, degrees=True), Rotation)
assert_type(Rotation.from_mrp(_f64_1d), Rotation)
assert_type(Rotation.from_mrp(_f64_2d), Rotation)
assert_type(Rotation.from_euler("xyz", 0.0), Rotation)
assert_type(Rotation.from_euler("xyz", _f64_1d, degrees=True), Rotation)
assert_type(Rotation.from_euler("xyz", _f64_2d), Rotation)
assert_type(Rotation.from_davenport(_f64_1d, "extrinsic", 0.0), Rotation)
assert_type(Rotation.from_davenport(_f64_2d, "intrinsic", _f64_1d, degrees=True), Rotation)

assert_type(Rotation.identity(), Rotation)
assert_type(Rotation.identity(4), Rotation)
assert_type(Rotation.identity(shape=(3,)), Rotation)
assert_type(Rotation.random(), Rotation)
assert_type(Rotation.random(4), Rotation)
assert_type(Rotation.random(shape=(2, 3)), Rotation)
assert_type(Rotation.concatenate(_rot), Rotation)
assert_type(Rotation.concatenate([_rot, _rot]), Rotation)
assert_type(Rotation.create_group("I"), Rotation)
assert_type(Rotation.create_group("Dn", "X"), Rotation)

assert_type(_rot.single, bool)
assert_type(_rot.shape, tuple[int, ...])

assert_type(_rot.__bool__(), Literal[True])  # noqa: PLC2801
assert_type(len(_rot), int)
assert_type(_rot * _rot, Rotation)
assert_type(_rot**2.0, Rotation)
assert_type(_rot[0], Rotation)
assert_type(_rot[0:2], Rotation)
assert_type(next(iter(_rot)), Rotation)
assert_type(_rot.inv(), Rotation)

assert_type(_rot.as_quat(), onp.Array1D[np.float64] | onp.Array2D[np.float64])
assert_type(_rot.as_quat(canonical=True, scalar_first=True), onp.Array1D[np.float64] | onp.Array2D[np.float64])
assert_type(_rot.as_matrix(), onp.Array2D[np.float64] | onp.Array3D[np.float64])
assert_type(_rot.as_rotvec(), onp.Array1D[np.float64] | onp.Array2D[np.float64])
assert_type(_rot.as_rotvec(degrees=True), onp.Array1D[np.float64] | onp.Array2D[np.float64])
assert_type(_rot.as_mrp(), onp.Array1D[np.float64] | onp.Array2D[np.float64])
assert_type(_rot.as_euler("xyz"), onp.Array1D[np.float64] | onp.Array2D[np.float64])
assert_type(_rot.as_euler("ZYX", degrees=True), onp.Array1D[np.float64] | onp.Array2D[np.float64])
assert_type(_rot.as_davenport(_f64_1d, "extrinsic"), onp.Array1D[np.float64] | onp.Array2D[np.float64])

assert_type(_rot.mean(), Rotation)
assert_type(_rot.mean(weights=_f64_1d), Rotation)
assert_type(_rot.magnitude(), float | np.float64 | onp.Array1D[np.float64])
assert_type(_rot.apply(_f64_1d), onp.Array1D[np.float64] | onp.Array2D[np.float64])
assert_type(_rot.apply(_f64_2d, inverse=True), onp.Array1D[np.float64] | onp.Array2D[np.float64])
assert_type(_rot.approx_equal(_rot), bool | np.bool_ | onp.Array1D[np.bool_])
assert_type(_rot.approx_equal(_rot, atol=1e-6, degrees=True), bool | np.bool_ | onp.Array1D[np.bool_])

assert_type(_rot.reduce(), Rotation)
assert_type(_rot.reduce(left=_rot), Rotation)
_reduced_with_idx = _rot.reduce(return_indices=True)
assert_type(_reduced_with_idx[0], Rotation)
assert_type(_reduced_with_idx[1], onp.ArrayND[np.int32 | np.int64])
assert_type(_reduced_with_idx[2], onp.ArrayND[np.int32 | np.int64])

_aligned = Rotation.align_vectors(_f64_2d, _f64_2d)
assert_type(_aligned[0], Rotation)
assert_type(_aligned[1], float)

_aligned_sens = Rotation.align_vectors(_f64_2d, _f64_2d, return_sensitivity=True)
assert_type(_aligned_sens[0], Rotation)
assert_type(_aligned_sens[1], float)
assert_type(_aligned_sens[2], onp.Array2D[np.float64])

###
# Slerp

_slerp = Slerp(_f64_1d, _rot)
assert_type(_slerp.times, onp.Array1D[np.int32 | np.int64 | np.float32 | np.float64])
assert_type(_slerp.timedelta, onp.Array1D[np.int32 | np.int64 | np.float32 | np.float64])
assert_type(_slerp.rotations, Rotation)
assert_type(_slerp.rotvecs, onp.Array2D[np.float64])
assert_type(_slerp(_f64_1d), Rotation)
