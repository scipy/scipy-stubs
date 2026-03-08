# type-tests for `spatial/transform/_rigid_transform.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.spatial.transform import RigidTransform, Rotation

###

_tf_0d: RigidTransform[tuple[()]]
_tf_1d: RigidTransform[tuple[int]]
_tf_2d: RigidTransform[tuple[int, int]]
_tf_nd: RigidTransform

_rot_0d: Rotation[tuple[()]]
_rot_1d: Rotation[tuple[int]]
_rot_nd: Rotation

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

# __init__
assert_type(RigidTransform(_f64_2d), RigidTransform[tuple[()]])
assert_type(RigidTransform(_f64_3d), RigidTransform[tuple[int]])
assert_type(RigidTransform(_f64_nd), RigidTransform[tuple[Any, ...]])

# __len__
assert_type(len(_tf_1d), int)
assert_type(len(_tf_2d), int)
assert_type(len(_tf_nd), int)

# __mul__
assert_type(_tf_0d * _tf_0d, RigidTransform[tuple[()]])
assert_type(_tf_0d * _tf_1d, RigidTransform[tuple[int]])
assert_type(_tf_0d * _tf_2d, RigidTransform[tuple[int, int]])
assert_type(_tf_1d * _tf_0d, RigidTransform[tuple[int]])
assert_type(_tf_1d * _tf_1d, RigidTransform[tuple[int]])
assert_type(_tf_1d * _tf_2d, RigidTransform)
assert_type(_tf_2d * _tf_1d, RigidTransform)
assert_type(_tf_2d * _tf_2d, RigidTransform)
assert_type(_tf_nd * _tf_nd, RigidTransform)  # type: ignore[assert-type]

# __pow__
assert_type(_tf_0d**2, RigidTransform[tuple[()]])
assert_type(_tf_1d**2, RigidTransform[tuple[int]])
assert_type(_tf_nd**2, RigidTransform)

# __getitem__
assert_type(_tf_1d[0], RigidTransform[tuple[()]])
assert_type(_tf_2d[0], RigidTransform[tuple[int]])
assert_type(_tf_nd[0], RigidTransform)
assert_type(_tf_1d[:1], RigidTransform[tuple[int]])
assert_type(_tf_2d[:1], RigidTransform[tuple[int, int]])
assert_type(_tf_nd[:1], RigidTransform)

# __iter__
assert_type(next(iter(_tf_1d)), RigidTransform[tuple[()]])  # pyrefly: ignore[assert-type]
assert_type(next(iter(_tf_2d)), RigidTransform[tuple[int]])  # pyrefly: ignore[assert-type]
assert_type(next(iter(_tf_nd)), RigidTransform)

# rotation
assert_type(_tf_0d.rotation, Rotation[tuple[()]])
assert_type(_tf_1d.rotation, Rotation[tuple[int]])
assert_type(_tf_nd.rotation, Rotation)

# translation
assert_type(_tf_0d.translation, onp.ArrayND[np.float64])
assert_type(_tf_1d.translation, onp.ArrayND[np.float64])
assert_type(_tf_nd.translation, onp.ArrayND[np.float64])

# single
assert_type(_tf_0d.single, bool)
assert_type(_tf_1d.single, bool)
assert_type(_tf_nd.single, bool)

# shape
assert_type(_tf_0d.shape, tuple[()])
assert_type(_tf_1d.shape, tuple[int])
assert_type(_tf_nd.shape, tuple[Any, ...])

# from_matrix
assert_type(RigidTransform.from_matrix(_f64_2d), RigidTransform[tuple[()]])
assert_type(RigidTransform.from_matrix(_f64_3d), RigidTransform[tuple[int]])
assert_type(RigidTransform.from_matrix(_f64_nd), RigidTransform)

# from_rotation
assert_type(RigidTransform.from_rotation(_rot_0d), RigidTransform[tuple[()]])
assert_type(RigidTransform.from_rotation(_rot_1d), RigidTransform[tuple[int]])
assert_type(RigidTransform.from_rotation(_rot_nd), RigidTransform)

# from_translation
assert_type(RigidTransform.from_translation(_f64_1d), RigidTransform[tuple[()]])
assert_type(RigidTransform.from_translation(_f64_2d), RigidTransform[tuple[int]])
assert_type(RigidTransform.from_translation(_f64_nd), RigidTransform)

# from_components
assert_type(RigidTransform.from_components(_f64_1d, _rot_0d), RigidTransform[tuple[()]])
assert_type(RigidTransform.from_components(_f64_1d, _rot_1d), RigidTransform[tuple[int]])
assert_type(RigidTransform.from_components(_f64_1d, _rot_nd), RigidTransform)
assert_type(RigidTransform.from_components(_f64_2d, _rot_0d), RigidTransform[tuple[int]])
assert_type(RigidTransform.from_components(_f64_2d, _rot_1d), RigidTransform[tuple[int]])
assert_type(RigidTransform.from_components(_f64_2d, _rot_nd), RigidTransform[tuple[int]])
assert_type(RigidTransform.from_components(_f64_nd, _rot_0d), RigidTransform)
assert_type(RigidTransform.from_components(_f64_nd, _rot_1d), RigidTransform)
assert_type(RigidTransform.from_components(_f64_nd, _rot_nd), RigidTransform)

# from_exp_coords
assert_type(RigidTransform.from_exp_coords(_f64_1d), RigidTransform[tuple[()]])
assert_type(RigidTransform.from_exp_coords(_f64_2d), RigidTransform[tuple[int]])
assert_type(RigidTransform.from_exp_coords(_f64_nd), RigidTransform)

# from_dual_quat
assert_type(RigidTransform.from_dual_quat(_f64_1d), RigidTransform[tuple[()]])
assert_type(RigidTransform.from_dual_quat(_f64_2d, scalar_first=True), RigidTransform[tuple[int]])
assert_type(RigidTransform.from_dual_quat(_f64_nd), RigidTransform)

# identity
assert_type(RigidTransform.identity(), RigidTransform[tuple[()]])
assert_type(RigidTransform.identity(4), RigidTransform[tuple[int]])
assert_type(RigidTransform.identity(shape=4), RigidTransform[tuple[int]])

# concatenate
assert_type(RigidTransform.concatenate(_tf_nd), RigidTransform)
assert_type(RigidTransform.concatenate([_tf_0d, _tf_0d]), RigidTransform[tuple[int]])
assert_type(RigidTransform.concatenate([_tf_1d, _tf_1d]), RigidTransform[tuple[int, int]])
assert_type(RigidTransform.concatenate([_tf_nd, _tf_nd]), RigidTransform)  # type: ignore[assert-type]

# inv
assert_type(_tf_0d.inv(), RigidTransform[tuple[()]])
assert_type(_tf_1d.inv(), RigidTransform[tuple[int]])
assert_type(_tf_nd.inv(), RigidTransform)

# as_matrix
assert_type(_tf_0d.as_matrix(), onp.Array2D[np.float64])
assert_type(_tf_1d.as_matrix(), onp.Array3D[np.float64])
assert_type(_tf_nd.as_matrix(), onp.ArrayND[np.float64])

# as_exp_coords
assert_type(_tf_0d.as_exp_coords(), onp.Array1D[np.float64])
assert_type(_tf_1d.as_exp_coords(), onp.Array2D[np.float64])
assert_type(_tf_nd.as_exp_coords(), onp.ArrayND[np.float64])

# as_dual_quat
assert_type(_tf_0d.as_dual_quat(), onp.Array1D[np.float64])
assert_type(_tf_1d.as_dual_quat(), onp.Array2D[np.float64])
assert_type(_tf_nd.as_dual_quat(), onp.ArrayND[np.float64])

# as_components
assert_type(_tf_0d.as_components(), tuple[onp.Array1D[np.float64], Rotation[tuple[()]]])
assert_type(_tf_1d.as_components(), tuple[onp.Array2D[np.float64], Rotation[tuple[int]]])
assert_type(_tf_nd.as_components(), tuple[onp.ArrayND[np.float64], Rotation])

# mean
assert_type(_tf_0d.mean(), RigidTransform[tuple[()]])
assert_type(_tf_1d.mean(), RigidTransform[tuple[()]])
assert_type(_tf_nd.mean(), RigidTransform[tuple[()]])
assert_type(_tf_nd.mean(axis=0), RigidTransform)

# apply
assert_type(_tf_0d.apply(_f64_1d), onp.Array1D[np.float64])
assert_type(_tf_1d.apply(_f64_2d), onp.Array2D[np.float64])
assert_type(_tf_nd.apply(_f64_nd), onp.ArrayND[np.float64])
