import numpy as np
from typing import Any, Dict


class HRRModule:
    def __init__(self, dim: int, num_keys: int = 16, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.dim = int(dim)
        self.keys = [self._unit(rng.normal(size=self.dim).astype(np.float32)) for _ in range(num_keys)]
        self.memory = np.zeros(self.dim, dtype=np.float32)

    def _unit(self, v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v) + 1e-9)
        return (v / n).astype(np.float32)

    def _bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ra = np.fft.rfft(a, n=self.dim)
        rb = np.fft.rfft(b, n=self.dim)
        return np.fft.irfft(ra * rb, n=self.dim).astype(np.float32)

    def _unbind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ra = np.fft.rfft(a, n=self.dim)
        rb = np.fft.rfft(b, n=self.dim)
        # Avoid divide-by-zero
        denom = rb.copy()
        denom[np.abs(denom) < 1e-12] = 1e-12
        return np.fft.irfft(ra / denom, n=self.dim).astype(np.float32)

    def process(self, x):
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        if v.shape[0] < self.dim:
            v = np.pad(v, (0, self.dim - v.shape[0]))
        else:
            v = v[: self.dim]
        v = self._unit(v)
        # Bind with a key based on a hash of v to route
        key = self.keys[int(abs(float(v.sum()))) % len(self.keys)]
        bound = self._bind(v, key)
        # Update memory (EMA)
        self.memory = (0.99 * self.memory + 0.01 * bound).astype(np.float32)
        # Retrieve approximate v
        recon = self._unbind(self.memory, key)
        return recon.astype(np.float32)

    def train_step(self, inputs, targets) -> float:
        out = self.process(inputs)
        y = np.asarray(targets, dtype=np.float32).reshape(-1)
        if y.shape[0] < self.dim:
            y = np.pad(y, (0, self.dim - y.shape[0]))
        else:
            y = y[: self.dim]
        diff = out - y
        return float(np.mean(diff ** 2))


def create(config: Dict[str, Any] | None = None) -> HRRModule:
    if config is None:
        shape = (27, 27, 27)
    else:
        shape = tuple(config.get('filepaths', {}).get('sponge_size', [27, 27, 27]))
    dim = int(np.prod(shape))
    return HRRModule(dim=dim)