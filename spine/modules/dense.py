import math
import numpy as np
from typing import Any, Dict


class DenseModule:
    def __init__(self, input_dim: int, output_dim: int | None = None, lr: float = 0.01, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        # Lightweight elementwise affine parameters (same-size mapping)
        self.W = rng.uniform(0.9, 1.1, size=(self.output_dim,)).astype(np.float32)
        self.b = np.zeros((self.output_dim,), dtype=np.float32)
        self.lr = lr

    def _match_dim(self, v: np.ndarray, dim: int) -> np.ndarray:
        v = v.reshape(-1)
        if v.shape[0] < dim:
            return np.pad(v, (0, dim - v.shape[0]))
        if v.shape[0] > dim:
            return v[:dim]
        return v

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        x = self._match_dim(x, self.output_dim)
        return (self.W * x + self.b).astype(np.float32)

    def train_step(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        x = np.asarray(inputs, dtype=np.float32).reshape(-1)
        y = np.asarray(targets, dtype=np.float32).reshape(-1)
        x = self._match_dim(x, self.output_dim)
        y = self._match_dim(y, self.output_dim)
        # Forward
        z = self.W * x + self.b
        # MSE loss and gradients
        diff = (z - y)
        loss = float(np.mean(diff ** 2))
        grad_z = 2.0 * diff / diff.size
        grad_W = grad_z * x
        grad_b = grad_z
        # SGD update
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        return loss


def create(config: Dict[str, Any] | None = None) -> DenseModule:
    if config is None:
        shape = (27, 27, 27)
        lr = 0.01
    else:
        shape = tuple(config.get("filepaths", {}).get("sponge_size", [27, 27, 27]))
        lr = float(config.get("training", {}).get("inner_lr", 0.01))
    input_dim = int(np.prod(shape))
    return DenseModule(input_dim=input_dim, output_dim=input_dim, lr=lr)