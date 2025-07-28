# memory/memory_hffs.py

"""
Holographic Fractal File System (HFFS) memory backend.
"""
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HFFSMemory:
    def __init__(self, base_path, sponge_size=(27,27,27)):
        self.base_path = base_path
        try:
            os.makedirs(self.base_path, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create memory directory: {e}")
        self.sponge_size = tuple(sponge_size)

    def recall_last(self):
        if not self.memory:
            return None
        last_key = sorted(self.memory.keys())[-1]
        return self.memory[last_key]


    def store(self, key, vector):
        """Persist state vector into fractal map"""
        try:
            path = os.path.join(self.base_path, f"{key}.npy")
            arr = np.array(vector).reshape(self.sponge_size)
            np.save(path, arr)
        except Exception as e:
            logger.warning(f"Failed storing memory '{key}': {e}")

    def load(self, key):
        """Load stored state"""
        try:
            path = os.path.join(self.base_path, f"{key}.npy")
            return np.load(path)
        except FileNotFoundError:
            logger.warning(f"Memory '{key}' not found.")
            return np.zeros(self.sponge_size)

    def distance_to_nearest(self, state_vector):
        """Compute euclidean distance to nearest stored vector"""
        min_dist = float('inf')
        for fn in os.listdir(self.base_path):
            try:
                stored = np.load(os.path.join(self.base_path, fn)).flatten()
                d = np.linalg.norm(np.array(state_vector).flatten() - stored)
                if d < min_dist:
                    min_dist = d
            except Exception:
                continue
        return min_dist if min_dist != float('inf') else 0.0

# Factory for plugin system (not loaded via spine)
def create(config=None):
    path = config.get("filepaths", {}).get("memory_base", "data/sponge") if config else "data/sponge"
    size = tuple(config.get("filepaths", {}).get("sponge_size", [27,27,27])) if config else (27,27,27)
    return HFFSMemory(path, sponge_size=size)
