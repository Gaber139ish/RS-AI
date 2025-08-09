from typing import Any, Dict
import numpy as np


class InterpretabilityProbe:
    def __init__(self):
        self.last = {}

    def probe(self, tensor) -> Dict[str, Any]:
        arr = np.asarray(tensor).reshape(-1)
        info = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "max": float(arr.max()),
            "min": float(arr.min()),
        }
        self.last = info
        return info