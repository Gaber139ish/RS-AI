import os
import json
from typing import List, Optional
from chain.block import Block


class Ledger:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.chain_file = os.path.join(base_path, "chain.jsonl")
        os.makedirs(base_path, exist_ok=True)
        self.blocks: List[Block] = []
        self._load()

    def _load(self):
        if not os.path.isfile(self.chain_file):
            return
        with open(self.chain_file, 'r') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    blk = Block(
                        index=d['index'], prev_hash=d['prev_hash'], timestamp=d['timestamp'],
                        proposer=d['proposer'], stake=d['stake'], ai_score=d['ai_score'],
                        txs=[], signature_hex=d.get('signature_hex', ''), hash_hex=d.get('hash_hex', '')
                    )
                    self.blocks.append(blk)
                except Exception:
                    continue

    def last_hash(self) -> str:
        return self.blocks[-1].hash_hex if self.blocks else "genesis"

    def height(self) -> int:
        return len(self.blocks)

    def append(self, blk: Block) -> bool:
        # Minimal consistency check
        if blk.prev_hash != self.last_hash():
            return False
        # Persist
        with open(self.chain_file, 'a') as f:
            f.write(blk.to_json() + "\n")
        self.blocks.append(blk)
        return True