from typing import Dict


class ChainState:
    def __init__(self):
        self.stake: Dict[str, float] = {}
        self.slashed: Dict[str, float] = {}

    def set_stake(self, node_id: str, amount: float):
        self.stake[node_id] = float(amount)

    def get_stake(self, node_id: str) -> float:
        return float(self.stake.get(node_id, 0.0))

    def total_stake(self) -> float:
        return float(sum(self.stake.values()))

    def apply_slash(self, node_id: str, fraction: float) -> float:
        fraction = max(0.0, min(1.0, float(fraction)))
        before = self.get_stake(node_id)
        penalty = before * fraction
        after = max(0.0, before - penalty)
        self.stake[node_id] = after
        self.slashed[node_id] = self.slashed.get(node_id, 0.0) + penalty
        return penalty