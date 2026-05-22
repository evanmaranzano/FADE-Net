"""Deprecated compatibility wrapper for the current SWA implementation."""

import os
import runpy
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))


if __name__ == "__main__":
    print("average_swa.py is deprecated; delegating to scripts/swa_average.py")
    runpy.run_path(os.path.join(SCRIPT_DIR, "swa_average.py"), run_name="__main__")
