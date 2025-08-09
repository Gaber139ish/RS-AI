"""
Central neural architecture with plugin/module system.
"""
import importlib
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class NeuralSpine:
    def __init__(self, config: Dict[str, Any]):
        """Load core modules based on config"""
        self.config = config
        self.modules: Dict[str, Any] = {}
        self._load_modules()

    def _load_modules(self) -> None:
        for name, path in self.config.get("modules", {}).items():
            try:
                module = importlib.import_module(path)
                factory = getattr(module, "create")
                try:
                    # Prefer passing full config if factory supports it
                    self.modules[name] = factory(self.config)
                except TypeError:
                    # Fallback to no-arg factory
                    self.modules[name] = factory()
            except ImportError:
                logger.warning(f"Module '{name}' at '{path}' not found, skipping.")
            except AttributeError:
                logger.warning(f"Module '{name}' loaded but has no 'create()' method, skipping.")

    def forward(self, x):
        """Pass data through active modules in sequence. Modules without 'process' are skipped."""
        for name, module in self.modules.items():
            if hasattr(module, "process"):
                x = module.process(x)
            else:
                logger.debug(f"Module '{name}' has no 'process', skipping in forward().")
        return x

    def add_module(self, name, module_obj) -> None:
        """Dynamically add a new module"""
        self.modules[name] = module_obj

    def remove_module(self, name) -> None:
        """Remove an existing module"""
        self.modules.pop(name, None)

    def prioritize(self, name: str) -> None:
        if name not in self.modules:
            return
        # Reinsert module at the beginning to process first
        module = self.modules.pop(name)
        self.modules = {name: module, **self.modules}

    def train_step(self, inputs, targets) -> float:
        """Run a single training step across modules that implement 'train_step'.
        Returns the average loss across participating modules.
        """
        losses = []
        for name, module in self.modules.items():
            if hasattr(module, "train_step"):
                try:
                    loss = float(module.train_step(inputs, targets))
                    losses.append(loss)
                except Exception as e:
                    logger.warning(f"Module '{name}' train_step failed: {e}")
        return float(sum(losses) / len(losses)) if losses else 0.0
