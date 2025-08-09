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
        return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b)).astype(np.float32)

    def _unbind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        conj = np.conj(np.fft.rfft(b))
        return np.fft.irfft(np.fft.rfft(a) * conj).astype(np.float32)

    def process(self, x):
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        v = self._unit(v[: self.dim]) if v.shape[0] >= self.dim else self._unit(np.pad(v, (0, self.dim - v.shape[0])))
        # Bind with a key based on a hash of v to route
        key = self.keys[int(abs(float(v.sum()))) % len(self.keys)]
        bound = self._bind(v, key)
        # Update memory (EMA)
        self.memory = 0.99 * self.memory + 0.01 * bound
        # Retrieve approximate v
        recon = self._unbind(self.memory, key)
        return recon.astype(np.float32)

    def train_step(self, inputs, targets) -> float:
        out = self.process(inputs)
        y = np.asarray(targets, dtype=np.float32).reshape(-1)
        y = y[: self.dim] if y.shape[0] >= self.dim else np.pad(y, (0, self.dim - y.shape[0]))
        diff = out - y
        return float(np.mean(diff ** 2))


def create(config: Dict[str, Any] | None = None) -> HRRModule:
    if config is None:
        shape = (27, 27, 27)
    else:
        shape = tuple(config.get('filepaths', {}).get('sponge_size', [27, 27, 27]))
    dim = int(np.prod(shape))
    return HRRModule(dim=dim)