# spine/curiosity_engine.py

import random
import math
import time
from typing import Any, Optional

class CuriosityEngine:
    def __init__(self, memory: Optional[Any] = None, threshold: float = 0.1, memory_size: int = 100, curiosity_factor: float = 0.8, recursion_depth: int = 2, hook=None):
        self.short_term_memory = []
        self.memory_size = memory_size
        self.curiosity_factor = curiosity_factor
        self.recursion_depth = recursion_depth
        self.hook = hook or (lambda info: None)  # Default to no-op
        self.memory_backend = memory
        self.threshold = threshold

    def remember(self, data):
        if len(self.short_term_memory) >= self.memory_size:
            self.short_term_memory.pop(0)
        self.short_term_memory.append(list(data) if not isinstance(data, list) else data)

    def is_novel(self, data):
        # Basic novelty check based on cosine-like difference
        return all(self._difference(data, m) > self.curiosity_factor for m in self.short_term_memory)

    def _difference(self, a, b):
        try:
            a, b = list(map(float, a)), list(map(float, b))
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x ** 2 for x in a))
            norm_b = math.sqrt(sum(y ** 2 for y in b))
            return 1 - dot / (norm_a * norm_b + 1e-9)
        except Exception:
            return 1.0  # Default to max difference if non-numeric

    def explore(self, inputs, depth=0):
        self.hook({
            "stage": "explore",
            "depth": depth,
            "inputs": inputs,
            "memory": self.short_term_memory[-3:],  # last 3 for trace
        })

        if self.is_novel(inputs):
            self.remember(inputs)
            if depth < self.recursion_depth:
                mutated = self.mutate(inputs)
                return self.explore(mutated, depth + 1)
            return inputs
        else:
            return None  # Skip known data

    def mutate(self, data):
        # Slightly mutate input data
        try:
            return [x + random.uniform(-0.1, 0.1) for x in data]
        except Exception:
            return data

    def reward(self, vector) -> float:
        """Compute a novelty reward for a vector.
        - If a persistent memory backend is provided, use distance to nearest stored vector.
        - Otherwise, use cosine-difference to last short-term memory entry.
        """
        try:
            if self.memory_backend is not None and hasattr(self.memory_backend, "distance_to_nearest"):
                distance = float(self.memory_backend.distance_to_nearest(vector))
                # Normalize via simple logistic to (0,1)
                score = 1.0 / (1.0 + math.exp(-(distance - self.threshold)))
                return max(0.0, min(1.0, score))
            # Fallback to short-term memory comparison
            if self.short_term_memory:
                diff = self._difference(vector, self.short_term_memory[-1])
                return float(max(0.0, min(1.0, diff)))
            return 0.0
        except Exception:
            return 0.0


def create():
    return CuriosityEngine(
        memory=None,
        threshold=0.1,
        memory_size=200,
        curiosity_factor=0.85,
        recursion_depth=3,
        hook=lambda info: print(f"[HOOK] Curiosity: {info}")
    )
