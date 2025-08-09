import numpy as np
import logging
from typing import Any, Dict


class SSMModule:
    def __init__(self, dim: int, hidden: int = 256, lr: float = 1e-3, seed: int = 42, clip_norm: float = 1.0, state_clip_value: float = 10.0):
        self._logger = logging.getLogger(__name__)
        rng = np.random.default_rng(seed)
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.clip_norm = float(clip_norm)
        self.state_clip_value = float(state_clip_value)
        # State x_t in R^hidden; observation y_t in R^dim
        self.A = (0.95 * np.eye(self.hidden) + 0.05 * rng.normal(0, 0.1, (self.hidden, self.hidden))).astype(np.float32)
        self.C = rng.normal(0, 0.1, (self.dim, self.hidden)).astype(np.float32)
        self.h = np.zeros(self.hidden, dtype=np.float32)

    def process(self, x):
        y = np.asarray(x, dtype=np.float32).reshape(-1)
        y = y[: self.dim] if y.shape[0] >= self.dim else np.pad(y, (0, self.dim - y.shape[0]))
        # Predict next hidden and observation
        self.h = (self.A @ self.h).astype(np.float32)
        # Clip hidden state to avoid blow-up
        if self.state_clip_value > 0:
            np.clip(self.h, -self.state_clip_value, self.state_clip_value, out=self.h)
        y_hat = (self.C @ self.h).astype(np.float32)
        # Correct hidden via simple residual mapping
        resid = y - y_hat
        # Residual clipping by norm
        resid_norm = float(np.linalg.norm(resid)) + 1e-9
        if resid_norm > self.clip_norm:
            scale = self.clip_norm / resid_norm
            resid = (resid * scale).astype(np.float32)
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug("SSM resid clipped: norm=%.4f -> clip_norm=%.4f", resid_norm, self.clip_norm)
        # Project residual into hidden with C^T
        dh = (self.C.T @ resid).astype(np.float32)
        # Clip hidden delta
        dh_norm = float(np.linalg.norm(dh)) + 1e-9
        if dh_norm > self.clip_norm:
            dh = (dh * (self.clip_norm / dh_norm)).astype(np.float32)
        self.h = self.h + 0.01 * dh
        if self.state_clip_value > 0:
            np.clip(self.h, -self.state_clip_value, self.state_clip_value, out=self.h)
        return y_hat.astype(np.float32)

    def train_step(self, inputs, targets) -> float:
        y = np.asarray(targets, dtype=np.float32).reshape(-1)
        y = y[: self.dim] if y.shape[0] >= self.dim else np.pad(y, (0, self.dim - y.shape[0]))
        y_hat = self.process(inputs)
        diff = y_hat - y
        # Clip diff to stabilize gradient
        diff_norm = float(np.linalg.norm(diff)) + 1e-9
        if diff_norm > self.clip_norm:
            diff = (diff * (self.clip_norm / diff_norm)).astype(np.float32)
        # Least-squares gradient step on C only (keep A stable)
        grad_C = np.outer(diff, self.h).astype(np.float32)
        # Frobenius norm clip
        gnorm = float(np.linalg.norm(grad_C)) + 1e-9
        if gnorm > self.clip_norm:
            grad_C *= (self.clip_norm / gnorm)
        self.C -= self.lr * grad_C
        return float(np.mean(diff ** 2))


def create(config: Dict[str, Any] | None = None) -> SSMModule:
    if config is None:
        shape = (27, 27, 27)
        hidden = 256
        lr = 1e-3
        clip_norm = 1.0
    else:
        shape = tuple(config.get('filepaths', {}).get('sponge_size', [27, 27, 27]))
        hidden = int(config.get('training', {}).get('ssm_hidden', 256))
        lr = float(config.get('training', {}).get('ssm_lr', 1e-3))
        clip_norm = float(config.get('training', {}).get('clip_norm', 1.0))
    dim = int(np.prod(shape))
    return SSMModule(dim=dim, hidden=hidden, lr=lr, clip_norm=clip_norm)