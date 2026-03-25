"""Pin Python to 3.12 to avoid 3.14 host-header leakage with NDK clang."""
from pythonforandroid.recipes.python3 import Python3Recipe as _Python3Recipe


class Python3Recipe(_Python3Recipe):
    version = '3.12.7'
    url = 'https://www.python.org/ftp/python/{version}/Python-{version}.tgz'


recipe = Python3Recipe()
