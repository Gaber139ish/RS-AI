from typing import Any
import numpy as np


class AutoMemory:
    def __init__(self, backends: list[Any]):
        self.backends = backends
        self.active_idx = 0 if backends else -1

    @property
    def sponge_size(self):
        return self.backends[self.active_idx].sponge_size

    def _choose(self, vector) -> int:
        # Simple heuristic: pick backend with largest distance -> treats it as more novel
        distances = [b.distance_to_nearest(vector) for b in self.backends]
        return int(np.argmax(distances))

    def store(self, key: str, vector):
        idx = self._choose(vector)
        self.active_idx = idx
        self.backends[idx].store(key, vector)

    def load(self, key: str):
        # Read from current active
        return self.backends[self.active_idx].load(key)

    def distance_to_nearest(self, state_vector) -> float:
        return max(b.distance_to_nearest(state_vector) for b in self.backends)