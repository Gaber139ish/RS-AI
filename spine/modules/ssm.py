import numpy as np
from typing import Any, Dict


class SSMModule:
    def __init__(self, dim: int, hidden: int = 256, lr: float = 1e-3, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.lr = float(lr)
        # State x_t in R^hidden; observation y_t in R^dim
        self.A = (0.95 * np.eye(self.hidden) + 0.05 * rng.normal(0, 0.1, (self.hidden, self.hidden))).astype(np.float32)
        self.C = rng.normal(0, 0.1, (self.dim, self.hidden)).astype(np.float32)
        self.h = np.zeros(self.hidden, dtype=np.float32)

    def process(self, x):
        y = np.asarray(x, dtype=np.float32).reshape(-1)
        y = y[: self.dim] if y.shape[0] >= self.dim else np.pad(y, (0, self.dim - y.shape[0]))
        # Predict next hidden and observation
        self.h = (self.A @ self.h).astype(np.float32)
        y_hat = (self.C @ self.h).astype(np.float32)
        # Correct hidden via simple residual mapping
        resid = y - y_hat
        # Project residual into hidden with C^T
        self.h = self.h + 0.01 * (self.C.T @ resid)
        return y_hat.astype(np.float32)

    def train_step(self, inputs, targets) -> float:
        y = np.asarray(targets, dtype=np.float32).reshape(-1)
        y = y[: self.dim] if y.shape[0] >= self.dim else np.pad(y, (0, self.dim - y.shape[0]))
        y_hat = self.process(inputs)
        diff = y_hat - y
        # Least-squares gradient step on C only (keep A stable)
        grad_C = np.outer(diff, self.h)
        self.C -= self.lr * grad_C
        return float(np.mean(diff ** 2))


def create(config: Dict[str, Any] | None = None) -> SSMModule:
    if config is None:
        shape = (27, 27, 27)
        hidden = 256
        lr = 1e-3
    else:
        shape = tuple(config.get('filepaths', {}).get('sponge_size', [27, 27, 27]))
        hidden = int(config.get('training', {}).get('ssm_hidden', 256))
        lr = float(config.get('training', {}).get('ssm_lr', 1e-3))
    dim = int(np.prod(shape))
    return SSMModule(dim=dim, hidden=hidden, lr=lr)