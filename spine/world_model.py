import numpy as np
from typing import Any, Dict


class WorldModel:
    def __init__(self, dim: int, lr: float = 1e-3):
        self.dim = int(dim)
        self.W = np.zeros((dim, dim), dtype=np.float32)
        self.lr = float(lr)
        self.prev = np.zeros(dim, dtype=np.float32)

    def process(self, x):
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        v = v[: self.dim] if v.shape[0] >= self.dim else np.pad(v, (0, self.dim - v.shape[0]))
        y = (self.W @ self.prev)
        self.prev = v
        return y

    def train_step(self, inputs, targets) -> float:
        y = np.asarray(targets, dtype=np.float32).reshape(-1)
        y = y[: self.dim] if y.shape[0] >= self.dim else np.pad(y, (0, self.dim - y.shape[0]))
        y_hat = self.process(inputs)
        diff = y_hat - y
        # Simple gradient update on W with outer product
        self.W -= self.lr * np.outer(diff, self.prev)
        return float(np.mean(diff ** 2))


def create(config: Dict[str, Any] | None = None) -> WorldModel:
    if config is None:
        shape = (27, 27, 27)
        lr = 1e-3
    else:
        shape = tuple(config.get('filepaths', {}).get('sponge_size', [27, 27, 27]))
        lr = float(config.get('training', {}).get('world_lr', 1e-3))
    dim = int(np.prod(shape))
    return WorldModel(dim=dim, lr=lr)