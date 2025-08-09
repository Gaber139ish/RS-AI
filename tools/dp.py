import numpy as np
from typing import List


def add_dp_noise(arr, sigma: float = 1e-3):
    a = np.asarray(arr, dtype=np.float32)
    return a + np.random.normal(0, sigma, size=a.shape).astype(np.float32)


def secure_aggregate(chunks: List):
    # Placeholder: sum of chunks
    return sum(chunks)