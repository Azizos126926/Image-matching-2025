import numpy as np

from imc2025.submission import array_to_str, none_to_str


def test_none_to_str():
    assert none_to_str(3) == "nan;nan;nan"


def test_array_to_str():
    value = np.array([1.0, 2.25, 3.5])
    assert array_to_str(value) == "1.000000000;2.250000000;3.500000000"
