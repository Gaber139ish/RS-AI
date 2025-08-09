import numpy as np
from typing import Any, Tuple


class TensorNetworkCompressor:
    def __init__(self, backend: Any, rank: int = 8):
        self.backend = backend
        self.rank = int(rank)

    @property
    def sponge_size(self) -> Tuple[int, int, int]:
        return self.backend.sponge_size

    def _compress(self, block: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Flatten 3D to 2D for SVD proxy
        mat = block.reshape(block.shape[0], -1)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        k = min(self.rank, S.shape[0])
        return U[:, :k], S[:k], Vt[:k, :]

    def _decompress(self, U, S, Vt, shape) -> np.ndarray:
        mat = (U * S) @ Vt
        return mat.reshape(shape)

    def store(self, key: str, vector):
        # Delegate to backend (could compress blocks internally if backend exposes hooks)
        self.backend.store(key, vector)

    def load(self, key: str):
        return self.backend.load(key)

    def distance_to_nearest(self, state_vector):
        return self.backend.distance_to_nearest(state_vector)