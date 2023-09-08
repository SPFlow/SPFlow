import pytest
import tensorly as tl
import sys

def pytest_configure():
    print("hello world")
    tl.set_backend("pytorch")
    print("Backend: "+ tl.get_backend())

