# python imports
import os
import sys

from tests.control.test_throttle_pid import test_throttle_pid_1

# pytest
# python_path: str = sys.executable
# test_command: str = f"{python_path} -m pytest tests"
# os.system(test_command)

if __name__ == "__main__":
    test_throttle_pid_1()