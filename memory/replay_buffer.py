import threading
import random
from typing import List, Tuple
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = int(capacity)
        self.storage: List[np.ndarray] = []
        self.lock = threading.Lock()
        self._idx = 0

    def add(self, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        with self.lock:
            if len(self.storage) < self.capacity:
                self.storage.append(vector)
            else:
                self.storage[self._idx] = vector
                self._idx = (self._idx + 1) % self.capacity

    def size(self) -> int:
        with self.lock:
            return len(self.storage)

    def sample(self, batch_size: int) -> np.ndarray:
        batch_size = min(batch_size, self.size())
        with self.lock:
            if batch_size <= 0:
                return np.empty((0,), dtype=np.float32)
            idxs = random.sample(range(len(self.storage)), k=batch_size)
            batch = [self.storage[i] for i in idxs]
        # Stack into (batch, dim)
        return np.stack(batch, axis=0)