import pytest
import numpy as np

from py21cmcast import core

def test_compare():
    assert core.compare_arrays(np.array([1, 1, 3.14]), np.array([1, 1, 3.14]), 1e-6) is True