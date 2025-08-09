import numpy as np
from typing import Any, Dict


class HopfieldModule:
    def __init__(self, dim: int, slots: int = 64, beta: float = 5.0, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.dim = int(dim)
        self.slots = int(slots)
        self.beta = float(beta)
        self.memory = rng.normal(0, 0.1, size=(self.slots, self.dim)).astype(np.float32)
        self._renorm_memory()

    def _renorm_memory(self):
        norms = np.linalg.norm(self.memory, axis=1, keepdims=True) + 1e-6
        self.memory = (self.memory / norms).astype(np.float32)

    def process(self, x):
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        v = v[: self.dim] if v.shape[0] >= self.dim else np.pad(v, (0, self.dim - v.shape[0]))
        v_norm = np.linalg.norm(v) + 1e-6
        # Stable attention scores
        scores = (self.memory @ v) / v_norm
        scores = scores - float(scores.max())
        attn = np.exp(np.clip(self.beta * scores, -50.0, 50.0))
        denom = attn.sum() + 1e-9
        attn = attn / denom
        if not np.isfinite(attn).all():
            attn = np.ones_like(attn) / attn.size
        y = (attn[:, None] * self.memory).sum(axis=0)
        # Hebbian write-back (fast weights) with small step
        self.memory = 0.999 * self.memory + 0.001 * np.outer(attn, v / v_norm)
        self._renorm_memory()
        return y.astype(np.float32)

    def train_step(self, inputs, targets) -> float:
        out = self.process(inputs)
        y = np.asarray(targets, dtype=np.float32).reshape(-1)
        y = y[: self.dim] if y.shape[0] >= self.dim else np.pad(y, (0, self.dim - y.shape[0]))
        diff = out - y
        return float(np.mean(diff ** 2))


def create(config: Dict[str, Any] | None = None) -> HopfieldModule:
    if config is None:
        shape = (27, 27, 27)
        beta = 5.0
    else:
        shape = tuple(config.get('filepaths', {}).get('sponge_size', [27, 27, 27]))
        beta = float(config.get('training', {}).get('hopfield_beta', 5.0))
    dim = int(np.prod(shape))
    return HopfieldModule(dim=dim, beta=beta)