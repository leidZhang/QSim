# python imports
import os
import sys

from tests.performance_environment import prepare_test_environment, destroy_map
from tests.sep_algs.test_sep_algs import test_sep_algs

# pytest
python_path: str = sys.executable
test_command: str = f"{python_path} -m pytest tests/unit_test/"
os.system(test_command)
