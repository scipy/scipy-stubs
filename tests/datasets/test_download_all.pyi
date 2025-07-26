from pathlib import Path
from typing import assert_type

from scipy.datasets import download_all

assert_type(download_all(), None)
assert_type(download_all(None), None)
assert_type(download_all("."), None)
assert_type(download_all(Path.cwd()), None)
