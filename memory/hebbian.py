import numpy as np
from typing import Tuple


class HebbianUpdater:
    def __init__(self, lr: float = 0.01, decay: float = 0.99):
        self.lr = float(lr)
        self.decay = float(decay)
        self.trace = None

    def update(self, block: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        # Ensure shapes align (broadcast to block size)
        pre = np.asarray(pre, dtype=np.float32)
        post = np.asarray(post, dtype=np.float32)
        # Flatten to match aggregated activity
        pre_m = pre.mean() if pre.ndim > 0 else float(pre)
        post_m = post.mean() if post.ndim > 0 else float(post)
        hebb = pre_m * post_m
        if self.trace is None:
            self.trace = np.zeros_like(block, dtype=np.float32)
        self.trace = self.decay * self.trace + (1.0 - self.decay) * hebb
        return block + self.lr * self.trace