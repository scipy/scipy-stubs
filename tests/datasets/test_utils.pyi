from typing import assert_type

from scipy import datasets

# all
assert_type(datasets.clear_cache(), None)

# ascent
assert_type(datasets.clear_cache(datasets.ascent), None)
assert_type(datasets.clear_cache([datasets.ascent]), None)
assert_type(datasets.clear_cache((datasets.ascent,)), None)

# electrocardiogram
assert_type(datasets.clear_cache(datasets.electrocardiogram), None)
assert_type(datasets.clear_cache([datasets.electrocardiogram]), None)
assert_type(datasets.clear_cache((datasets.electrocardiogram,)), None)

# face
assert_type(datasets.clear_cache(datasets.face), None)
assert_type(datasets.clear_cache([datasets.face]), None)
assert_type(datasets.clear_cache((datasets.face,)), None)

# combined
assert_type(datasets.clear_cache([datasets.ascent, datasets.electrocardiogram, datasets.face]), None)
assert_type(datasets.clear_cache((datasets.ascent, datasets.electrocardiogram, datasets.face)), None)
