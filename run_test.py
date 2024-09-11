import os
import sys

if __name__ == "__main__":
    python_path: str = sys.executable
    test_command: str = f"{python_path} -m pytest tests/"
    os.system(test_command)