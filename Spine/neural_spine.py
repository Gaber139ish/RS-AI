"""
Central neural architecture with plugin/module system.
"""
import importlib
import logging

logger = logging.getLogger(__name__)

class NeuralSpine:
    def __init__(self, config):
        """Load core modules based on config"""
        self.config = config
        self.modules = {}
        self._load_modules()

    def _load_modules(self):
        for name, path in self.config.get("modules", {}).items():
            try:
                module = importlib.import_module(path)
                # Expect each module to have a `create()` factory function
                self.modules[name] = module.create()
            except ImportError:
                logger.warning(f"Module '{name}' at '{path}' not found, skipping.")
            except AttributeError:
                logger.warning(f"Module '{name}' loaded but has no 'create()' method, skipping.")

    def forward(self, x):
        """Pass data through active modules in sequence"""
        for name, module in self.modules.items():
            x = module.process(x)
        return x

    def add_module(self, name, module_obj):
        """Dynamically add a new module"""
        self.modules[name] = module_obj

    def remove_module(self, name):
        """Remove an existing module"""
        self.modules.pop(name, None)
