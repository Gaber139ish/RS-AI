from typing import Any, Dict
import time


class AutoSelector:
    def __init__(self, spine, metrics, config: Dict[str, Any]):
        self.spine = spine
        self.metrics = metrics
        self.config = config
        self.last_switch = 0.0
        self.cooldown_s = float(config.get('auto', {}).get('cooldown_s', 10.0))

    def maybe_switch(self):
        now = time.time()
        if now - self.last_switch < self.cooldown_s:
            return
        snap = self.metrics.snapshot()
        avg_loss = float(snap.get('avg_loss', 1.0))
        # Simple heuristic: if loss high, prefer hopfield; otherwise dense
        if avg_loss > 0.01 and 'hopfield' in self.spine.modules:
            # Move hopfield to front of pipeline
            self.spine.prioritize('hopfield')
            self.last_switch = now
        elif avg_loss <= 0.01 and 'dense' in self.spine.modules:
            self.spine.prioritize('dense')
            self.last_switch = now