# spine/curiosity_engine.py

import random
import math
import time

class CuriosityEngine:
    def __init__(self, memory_size=100, curiosity_factor=0.8, recursion_depth=2, hook=None):
        self.memory = []
        self.memory_size = memory_size
        self.curiosity_factor = curiosity_factor
        self.recursion_depth = recursion_depth
        self.hook = hook or (lambda info: None)  # Default to no-op

    def remember(self, data):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append(data)

    def is_novel(self, data):
        # Basic novelty check based on cosine-like difference
        return all(self._difference(data, m) > self.curiosity_factor for m in self.memory)

    def _difference(self, a, b):
        try:
            a, b = list(map(float, a)), list(map(float, b))
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x ** 2 for x in a))
            norm_b = math.sqrt(sum(y ** 2 for y in b))
            return 1 - dot / (norm_a * norm_b + 1e-9)
        except:
            return 1.0  # Default to max difference if non-numeric

    def explore(self, inputs, depth=0):
        self.hook({
            "stage": "explore",
            "depth": depth,
            "inputs": inputs,
            "memory": self.memory[-3:],  # last 3 for trace
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
        except:
            return data

def create():
    return CuriosityEngine(
        memory_size=200,
        curiosity_factor=0.85,
        recursion_depth=3,
        hook=lambda info: print(f"[HOOK] Curiosity: {info}")
    )
