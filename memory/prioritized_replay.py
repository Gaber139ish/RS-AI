import threading
import random
from typing import List, Tuple
import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, epsilon: float = 1e-3):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.storage: List[np.ndarray] = []
        self.priorities: List[float] = []
        self.lock = threading.Lock()
        self._idx = 0

    def add(self, vector: np.ndarray, priority: float) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        p = float(priority) + self.epsilon
        with self.lock:
            if len(self.storage) < self.capacity:
                self.storage.append(vector)
                self.priorities.append(p)
            else:
                self.storage[self._idx] = vector
                self.priorities[self._idx] = p
                self._idx = (self._idx + 1) % self.capacity

    def size(self) -> int:
        with self.lock:
            return len(self.storage)

    def sample(self, batch_size: int) -> np.ndarray:
        with self.lock:
            n = len(self.storage)
            if n == 0:
                return np.empty((0,), dtype=np.float32)
            batch_size = min(batch_size, n)
            probs = np.asarray(self.priorities, dtype=np.float64) ** self.alpha
            probs = probs / probs.sum()
            idxs = np.random.choice(n, size=batch_size, p=probs, replace=False)
            batch = [self.storage[i] for i in idxs]
        return np.stack(batch, axis=0)