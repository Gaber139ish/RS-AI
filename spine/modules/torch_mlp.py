from typing import Any, Dict
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = object  # type: ignore


class TorchMLP(nn.Module if torch else object):  # type: ignore
    def __init__(self, input_dim: int, hidden_dim: int = 256, lr: float = 1e-3):
        if torch is None:
            raise ImportError("PyTorch not available")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.input_dim = input_dim

    def process(self, x):
        if torch is None:
            return x
        x_t = torch.tensor(np.asarray(x, dtype=np.float32).reshape(-1))
        if x_t.shape[0] != self.input_dim:
            if x_t.shape[0] < self.input_dim:
                pad = torch.zeros(self.input_dim - x_t.shape[0])
                x_t = torch.cat([x_t, pad], dim=0)
            else:
                x_t = x_t[: self.input_dim]
        with torch.no_grad():
            y = self.net(x_t)
        return y.detach().cpu().numpy()

    def train_step(self, inputs, targets) -> float:
        x = torch.tensor(np.asarray(inputs, dtype=np.float32).reshape(-1))
        y = torch.tensor(np.asarray(targets, dtype=np.float32).reshape(-1))
        if x.shape[0] != self.input_dim:
            if x.shape[0] < self.input_dim:
                x = torch.cat([x, torch.zeros(self.input_dim - x.shape[0])], dim=0)
            else:
                x = x[: self.input_dim]
        if y.shape[0] != self.input_dim:
            if y.shape[0] < self.input_dim:
                y = torch.cat([y, torch.zeros(self.input_dim - y.shape[0])], dim=0)
            else:
                y = y[: self.input_dim]
        self.optimizer.zero_grad()
        out = self.net(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())


def create(config: Dict[str, Any] | None = None) -> TorchMLP:
    if config is None:
        shape = (27, 27, 27)
        lr = 1e-3
        hidden = 128
    else:
        shape = tuple(config.get("filepaths", {}).get("sponge_size", [27, 27, 27]))
        tcfg = config.get("training", {})
        lr = float(tcfg.get("torch_lr", 1e-3))
        hidden = int(tcfg.get("torch_hidden", 128))
    input_dim = int(np.prod(shape))
    return TorchMLP(input_dim=input_dim, hidden_dim=hidden, lr=lr)