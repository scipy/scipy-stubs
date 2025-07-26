from typing import assert_type

from scipy.datasets import ascent, clear_cache, electrocardiogram, face

# all
assert_type(clear_cache(), None)

# ascent
assert_type(clear_cache(ascent), None)
assert_type(clear_cache([ascent]), None)
assert_type(clear_cache((ascent,)), None)

# electrocardiogram
assert_type(clear_cache(electrocardiogram), None)
assert_type(clear_cache([electrocardiogram]), None)
assert_type(clear_cache((electrocardiogram,)), None)

# face
assert_type(clear_cache(face), None)
assert_type(clear_cache([face]), None)
assert_type(clear_cache((face,)), None)

# combined
assert_type(clear_cache([ascent, electrocardiogram, face]), None)
assert_type(clear_cache((ascent, electrocardiogram, face)), None)
