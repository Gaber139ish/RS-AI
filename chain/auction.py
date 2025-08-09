from typing import List, Dict, Any
import random


class JobAuction:
    def __init__(self, base_reward: float = 0.5):
        self.base_reward = float(base_reward)

    def run(self, bidders: List[Dict[str, Any]], job: Dict[str, Any]) -> Dict[str, Any]:
        # Score bidders by stake*ai_score and randomness
        best = None
        best_score = -1.0
        for b in bidders:
            score = float(b.get('stake', 0.0)) * float(b.get('ai_score', 0.0)) * (0.9 + 0.2 * random.random())
            if score > best_score:
                best = b
                best_score = score
        payout = self.base_reward * (1.0 + min(best.get('ai_score', 0.0), 1.0)) if best else 0.0
        return {"winner": best, "payout": float(payout)}