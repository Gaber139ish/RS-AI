from typing import List, Tuple
from dataclasses import dataclass
from chain.block import Block


@dataclass
class Vote:
    voter: str
    block_hash: str


def bft_commit(blocks: List[Tuple[Block, bytes]], quorum: int) -> int:
    # Returns index of committed block by majority hash vote, or -1 if none
    counts = {}
    for i, (blk, _pub) in enumerate(blocks):
        h = blk.hash_hex
        counts[h] = counts.get(h, 0) + 1
        if counts[h] >= quorum:
            # commit first block with this hash
            for j, (blk2, _pub2) in enumerate(blocks):
                if blk2.hash_hex == h:
                    return j
    return -1