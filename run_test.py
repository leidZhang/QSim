import os
import sys

from core.utils.ipc_utils import EventQueue


if __name__ == "__main__":
    # pytest for unit tests
    python_path: str = sys.executable
    test_command: str = f"{python_path} -m pytest tests/"
    os.system(test_command)