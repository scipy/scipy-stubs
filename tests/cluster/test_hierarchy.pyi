from collections.abc import Iterator
from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.cluster.hierarchy import (
    DisjointSet,
    average,
    centroid,
    complete,
    cophenet,
    dendrogram,
    fcluster,
    fclusterdata,
    from_mlab_linkage,
    inconsistent,
    leaders,
    linkage,
    maxRstat,
    maxdists,
    maxinconsts,
    median,
    set_link_color_palette,
    single,
    to_mlab_linkage,
    ward,
    weighted,
)

###

py_str_1d: list[str]
py_int_1d: list[int]
py_float_1d: list[float]
py_complex_1d: list[complex]

i32_1d: onp.Array1D[np.int32]
i64_1d: onp.Array1D[np.int64]
f32_1d: onp.Array1D[np.float32]
f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
f80_1d: onp.Array1D[npc.floating80]
c128_1d: onp.Array1D[np.complex128]
c160_1d: onp.Array1D[npc.complexfloating160]

###

# fcluster
assert_type(fcluster(f64_2d, 1.5), onp.Array1D[np.int32])
assert_type(fcluster(f64_2d, t=1.5), onp.Array1D[np.int32])
assert_type(fcluster(f64_2d, 1.5, "inconsistent", R=f64_2d), onp.Array1D[np.int32])
# fclusterdata
assert_type(fclusterdata(f64_2d, 1.5), onp.Array1D[np.int32])
assert_type(fclusterdata(f64_2d, t=1.5), onp.Array1D[np.int32])
assert_type(fclusterdata(f64_2d, 1.5, "inconsistent", R=f64_2d), onp.Array1D[np.int32])
# leaders
assert_type(leaders(f64_2d, i32_1d), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])

###

# linkage
assert_type(linkage(f64_1d), onp.Array2D[np.float64])
assert_type(linkage(f64_2d), onp.Array2D[np.float64])
# linkage aliases
assert_type(single(f64_2d), onp.Array2D[np.float64])
assert_type(complete(f64_2d), onp.Array2D[np.float64])
assert_type(average(f64_2d), onp.Array2D[np.float64])
assert_type(weighted(f64_2d), onp.Array2D[np.float64])
assert_type(centroid(f64_2d), onp.Array2D[np.float64])
assert_type(median(f64_2d), onp.Array2D[np.float64])
assert_type(ward(f64_2d), onp.Array2D[np.float64])

###

# cophenet
assert_type(cophenet(f64_2d), onp.Array1D[np.float64])
assert_type(cophenet(f64_2d, i64_1d), tuple[np.float64, onp.Array1D[np.float64]])
assert_type(cophenet(f64_2d, f32_1d), tuple[np.float64, onp.Array1D[np.float64]])
assert_type(cophenet(f64_2d, f64_1d), tuple[np.float64, onp.Array1D[np.float64]])
assert_type(cophenet(f64_2d, f80_1d), tuple[np.longdouble, onp.Array1D[np.float64]])
assert_type(cophenet(f64_2d, c128_1d), tuple[np.complex128, onp.Array1D[np.float64]])
assert_type(cophenet(f64_2d, c160_1d), tuple[np.clongdouble, onp.Array1D[np.float64]])
assert_type(cophenet(f64_2d, py_float_1d), tuple[np.float64, onp.Array1D[np.float64]])
assert_type(cophenet(f64_2d, py_complex_1d), tuple[np.complex128, onp.Array1D[np.float64]])
# {from,to}_mlab_linkage
assert_type(from_mlab_linkage(f64_2d), onp.Array2D[np.float64])
assert_type(to_mlab_linkage(f64_2d), onp.Array2D[np.float64])
# inconsistent
assert_type(inconsistent(f64_2d), onp.Array2D[np.float64])
# maxinconsts
assert_type(maxinconsts(f64_2d, f64_2d), onp.Array1D[np.float64])
# maxdists
assert_type(maxdists(f64_2d), onp.Array1D[np.float64])
# maxRstat
assert_type(maxRstat(f64_2d, f64_2d, 0), onp.Array1D[np.float64])

###

# dendrogram
assert_type(dendrogram(f64_2d)["color_list"], list[str])
assert_type(dendrogram(f64_2d)["icoord"], list[list[int]])
assert_type(dendrogram(f64_2d)["dcoord"], list[list[int]])
assert_type(dendrogram(f64_2d)["ivl"], list[str])
assert_type(dendrogram(f64_2d)["leaves"], list[int] | None)
assert_type(dendrogram(f64_2d)["leaves_color_list"], list[str])
# set_link_color_palette
assert_type(set_link_color_palette(["red", "green", "blue"]), None)

###

# DisjointSet

# DisjointSet(Iterable[T]) produces a DisjointSet[T] with universal set of type T.
assert_type(DisjointSet(py_str_1d), DisjointSet[str])
assert_type(DisjointSet(py_int_1d), DisjointSet[int])
# NOTE: Directly using assert_type fails with numpy arrays for all numpy<=2.0. Instead, use assignment statements.
_10: DisjointSet[np.int32] = DisjointSet(i32_1d)
_11: DisjointSet[np.int64] = DisjointSet(i64_1d)
# DisjointSet() produces a DisjointSet[Any] because T is unbound.
assert_type(DisjointSet(), DisjointSet[Any])

disjoint_set_str: DisjointSet[str]
disjoint_set_i64: DisjointSet[np.int64]

# __iter__ produces an iterator over the universal set.
assert_type(iter(disjoint_set_str), Iterator[str])
assert_type(iter(disjoint_set_i64), Iterator[np.int64])

# __len__ returns the length of the universal set
assert_type(len(disjoint_set_str), int)

# __contains__ accepts an element of the universal set and returns a boolean
assert_type("a" in disjoint_set_str, bool)
assert_type(np.int64(2) in disjoint_set_i64, bool)

# __getitem__ returns an element of the universal set
assert_type(disjoint_set_str["a"], str)
disjoint_set_str[1]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
assert_type(disjoint_set_i64[np.int64(1)], np.int64)

# add accepts an element of type T and adds it to the data structure (i.e. returns None)
assert_type(disjoint_set_str.add("a"), None)
disjoint_set_str.add(1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
assert_type(disjoint_set_i64.add(np.int64(1)), None)

# merge accepts two elements of type T and returns a boolean indicating if they belonged to the same subset
assert_type(disjoint_set_str.merge("a", "b"), bool)
disjoint_set_str.merge(1, 2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
assert_type(disjoint_set_i64.merge(np.int64(1), np.int64(2)), bool)

# connected accepts two elements of type T and returns a boolean indicating if they belonged to the same subset
assert_type(disjoint_set_str.connected("a", "b"), bool)
disjoint_set_str.connected(1, 2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
assert_type(disjoint_set_i64.connected(np.int64(1), np.int64(2)), bool)

# subset accepts one element of type T and returns its containing subset.
assert_type(disjoint_set_str.subset("a"), set[str])
disjoint_set_str.subset(1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
assert_type(disjoint_set_i64.subset(np.int64(1)), set[np.int64])

# subset_size accepts one element of type T and returns the *size* of its subset.
assert_type(disjoint_set_str.subset_size("a"), int)
disjoint_set_str.subset_size(1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
assert_type(disjoint_set_i64.subset_size(np.int64(1)), int)

# subsets returns a list of all subsets of type T
assert_type(disjoint_set_str.subsets(), list[set[str]])
assert_type(disjoint_set_i64.subsets(), list[set[np.int64]])
