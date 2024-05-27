# python imports
import os
import sys

# pytest
python_path: str = sys.executable
test_command: str = f"{python_path} -m pytest tests/unit_test/"
os.system(test_command)
