import os
import json
import numpy as np
from typing import Any, Dict


def save_checkpoint(spine, path: str, extra: Dict[str, Any] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state: Dict[str, Any] = {"modules": {}, "extra": extra or {}}
    for name, module in spine.modules.items():
        mod_state = {}
        if hasattr(module, "W"):
            mod_state["W"] = module.W.tolist() if not isinstance(module.W, np.ndarray) else module.W.tolist()
        if hasattr(module, "b"):
            mod_state["b"] = module.b.tolist() if not isinstance(module.b, np.ndarray) else module.b.tolist()
        if hasattr(module, "lr"):
            mod_state["lr"] = float(module.lr)
        state["modules"][name] = mod_state
    with open(path, "w") as f:
        json.dump(state, f)


def load_checkpoint(spine, path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        state = json.load(f)
    for name, mod_state in state.get("modules", {}).items():
        module = spine.modules.get(name)
        if module is None:
            continue
        if hasattr(module, "W") and "W" in mod_state:
            module.W = np.asarray(mod_state["W"], dtype=np.float32)
        if hasattr(module, "b") and "b" in mod_state:
            module.b = np.asarray(mod_state["b"], dtype=np.float32)
        if hasattr(module, "lr") and "lr" in mod_state:
            module.lr = float(mod_state["lr"])
    return state.get("extra", {})