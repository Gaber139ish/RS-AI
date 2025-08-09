import random
from typing import List


class ModuleGater:
    def __init__(self, spine, explore_prob: float = 0.05):
        self.spine = spine
        self.explore_prob = float(explore_prob)
        self.last_bucket = None

    def _bucket(self, avg_loss: float) -> str:
        if avg_loss > 0.1:
            return 'high'
        if avg_loss > 0.01:
            return 'mid'
        return 'low'

    def step(self, avg_loss: float) -> None:
        names = list(self.spine.modules.keys())
        if not names:
            return
        bucket = self._bucket(avg_loss)
        if random.random() < self.explore_prob:
            # Move a random module to front
            choice = random.choice(names)
            self.spine.prioritize(choice)
            self.last_bucket = bucket
            return
        # Deterministic preference order per bucket
        pref: List[str] = []
        if bucket == 'high':
            pref = ['hopfield', 'hrr', 'dense']
        elif bucket == 'mid':
            pref = ['hrr', 'hopfield', 'dense']
        else:
            pref = ['dense', 'hrr', 'hopfield']
        # Find first available and prioritize
        for name in pref:
            if name in self.spine.modules:
                self.spine.prioritize(name)
                break
        self.last_bucket = bucket