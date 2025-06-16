from pathlib import Path
from typing import assert_type

from scipy import datasets

assert_type(datasets.download_all(), None)
assert_type(datasets.download_all(None), None)
assert_type(datasets.download_all("."), None)
assert_type(datasets.download_all(Path.cwd()), None)
