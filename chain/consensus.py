from typing import List, Tuple
import random
import time
from chain.block import Block, Transaction


def select_winner(proposals: List[Tuple[Block, bytes]]) -> int:
    # Each proposal: (block, proposer_pub)
    # Winner index by max stake*ai_score; ties by earliest timestamp
    best_idx = -1
    best_score = -1.0
    best_time = 0.0
    for i, (blk, _pub) in enumerate(proposals):
        score = float(blk.stake) * float(blk.ai_score)
        if score > best_score or (score == best_score and blk.timestamp < best_time):
            best_idx = i
            best_score = score
            best_time = blk.timestamp
    return best_idx


def select_committee(candidates: List[Tuple[Block, bytes]], k: int = 3) -> List[int]:
    weights = [max(1e-6, float(blk.stake) * float(blk.ai_score)) for blk, _ in candidates]
    total = sum(weights)
    probs = [w / total for w in weights]
    idxs = list(range(len(candidates)))
    # Sample without replacement by weighted reservoir approximation
    chosen = set()
    while len(chosen) < min(k, len(idxs)):
        r = random.random()
        cdf = 0.0
        for i, p in enumerate(probs):
            cdf += p
            if r <= cdf:
                chosen.add(i)
                break
    return list(chosen)


def verify_block(blk: Block) -> bool:
    # For now, recompute hash and ensure it matches
    return blk.compute_hash() == blk.hash_hex


def mint_reward(ai_score: float, base_reward: float = 1.0) -> float:
    # Reward increases with ai_score but caps to 2x
    return float(base_reward) * (1.0 + min(ai_score, 1.0))


def slash_fraction(misbehavior: str) -> float:
    if misbehavior == 'invalid_hash':
        return 0.05
    if misbehavior == 'double_propose':
        return 0.10
    return 0.0