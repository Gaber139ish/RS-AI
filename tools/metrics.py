import threading
from typing import Dict, Any
import time


class MetricsRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: Dict[str, Any] = {
            "timestamp": time.time(),
        }

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._metrics[key] = value
            self._metrics["timestamp"] = time.time()

    def inc(self, key: str, value: float = 1.0) -> None:
        with self._lock:
            self._metrics[key] = float(self._metrics.get(key, 0.0)) + float(value)
            self._metrics["timestamp"] = time.time()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._metrics)