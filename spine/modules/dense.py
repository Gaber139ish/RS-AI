import math
import numpy as np
from typing import Any, Dict


class DenseModule:
    def __init__(self, input_dim: int, output_dim: int | None = None, lr: float = 0.01, seed: int = 42, momentum: float = 0.9, clip_norm: float = 1.0):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        # Lightweight elementwise affine parameters (same-size mapping)
        self.W = rng.uniform(0.9, 1.1, size=(self.output_dim,)).astype(np.float32)
        self.b = np.zeros((self.output_dim,), dtype=np.float32)
        self.lr = lr
        self.momentum = float(momentum)
        self.clip_norm = float(clip_norm)
        # Velocity terms for momentum
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

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

    def _clip(self, g: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(g) + 1e-12)
        if n > self.clip_norm:
            g = g * (self.clip_norm / n)
        return g

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
        # Clip
        grad_W = self._clip(grad_W)
        grad_b = self._clip(grad_b)
        # Momentum SGD update
        self.vW = self.momentum * self.vW + (1.0 - self.momentum) * grad_W
        self.vb = self.momentum * self.vb + (1.0 - self.momentum) * grad_b
        self.W -= self.lr * self.vW
        self.b -= self.lr * self.vb
        return loss


def create(config: Dict[str, Any] | None = None) -> DenseModule:
    if config is None:
        shape = (27, 27, 27)
        lr = 0.01
        momentum = 0.9
        clip_norm = 1.0
    else:
        shape = tuple(config.get("filepaths", {}).get("sponge_size", [27, 27, 27]))
        tcfg = config.get("training", {})
        lr = float(tcfg.get("inner_lr", 0.01))
        momentum = float(tcfg.get("momentum", 0.9))
        clip_norm = float(tcfg.get("clip_norm", 1.0))
    input_dim = int(np.prod(shape))
    return DenseModule(input_dim=input_dim, output_dim=input_dim, lr=lr, momentum=momentum, clip_norm=clip_norm)