import os
import subprocess

__version__ = '0.4.0'

current_path = os.getcwd()
try:
    os.chdir(os.path.dirname(__file__))
    GIT_VERSION = subprocess.check_output(["git", "describe", "--always"]).strip().decode('utf-8')
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    GIT_VERSION = "Unknown"
os.chdir(current_path)