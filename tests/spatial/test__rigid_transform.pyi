# type-tests for `spatial/transform/_rigid_transform.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.spatial.transform import RigidTransform, Rotation

###

tf: RigidTransform
rot: Rotation
f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]

###
# construction

assert_type(RigidTransform(f64_2d), RigidTransform)
assert_type(RigidTransform(f64_3d, normalize=False, copy=False), RigidTransform)

assert_type(RigidTransform.from_matrix(f64_2d), RigidTransform)
assert_type(RigidTransform.from_matrix(f64_3d), RigidTransform)
assert_type(RigidTransform.from_rotation(rot), RigidTransform)
assert_type(RigidTransform.from_translation(f64_1d), RigidTransform)
assert_type(RigidTransform.from_translation(f64_2d), RigidTransform)
assert_type(RigidTransform.from_components(f64_1d, rot), RigidTransform)
assert_type(RigidTransform.from_components(f64_2d, rot), RigidTransform)
assert_type(RigidTransform.from_exp_coords(f64_1d), RigidTransform)
assert_type(RigidTransform.from_exp_coords(f64_2d), RigidTransform)
assert_type(RigidTransform.from_dual_quat(f64_1d), RigidTransform)
assert_type(RigidTransform.from_dual_quat(f64_2d, scalar_first=True), RigidTransform)
assert_type(RigidTransform.identity(), RigidTransform)
assert_type(RigidTransform.identity(4), RigidTransform)
assert_type(RigidTransform.identity(shape=(2, 3)), RigidTransform)
assert_type(RigidTransform.concatenate([tf, tf]), RigidTransform)

###
# properties

assert_type(tf.rotation, Rotation)
assert_type(tf.translation, onp.ArrayND[np.float64, tuple[int] | tuple[int, int]])
assert_type(tf.single, bool)
assert_type(tf.shape, tuple[int, ...])

###
# operators

assert_type(len(tf), int)
assert_type(tf * tf, RigidTransform)
assert_type(tf**2.0, RigidTransform)
assert_type(tf[0], RigidTransform)
assert_type(tf[0:2], RigidTransform)
assert_type(next(iter(tf)), RigidTransform)
assert_type(tf.inv(), RigidTransform)

###
# conversion methods

assert_type(tf.as_matrix(), onp.ArrayND[np.float64, tuple[int, int] | tuple[int, int, int]])
assert_type(tf.as_exp_coords(), onp.ArrayND[np.float64, tuple[int] | tuple[int, int]])
assert_type(tf.as_dual_quat(), onp.ArrayND[np.float64, tuple[int] | tuple[int, int]])
assert_type(tf.as_dual_quat(scalar_first=True), onp.ArrayND[np.float64, tuple[int] | tuple[int, int]])

_components = tf.as_components()
assert_type(_components[0], onp.ArrayND[np.float64, tuple[int] | tuple[int, int]])
assert_type(_components[1], Rotation)

###
# other methods

assert_type(tf.mean(), RigidTransform)
assert_type(tf.mean(weights=f64_1d), RigidTransform)
assert_type(tf.apply(f64_1d), onp.Array1D[np.float64])
assert_type(tf.apply(f64_2d), onp.Array2D[np.float64])
assert_type(tf.apply(f64_1d, inverse=True), onp.Array1D[np.float64])
