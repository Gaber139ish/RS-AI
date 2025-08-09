from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import time
import json
import hashlib


@dataclass
class Transaction:
    kind: str
    data: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, "data": self.data}, sort_keys=True)


@dataclass
class Block:
    index: int
    prev_hash: str
    timestamp: float
    proposer: str  # public id hex
    stake: float
    ai_score: float
    txs: List[Transaction]
    signature_hex: str = ""
    hash_hex: str = ""

    def compute_hash(self) -> str:
        payload = {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "proposer": self.proposer,
            "stake": round(self.stake, 8),
            "ai_score": round(self.ai_score, 8),
            "txs": [json.loads(tx.to_json()) for tx in self.txs],
            "signature_hex": self.signature_hex,
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def to_json(self) -> str:
        d = {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "proposer": self.proposer,
            "stake": self.stake,
            "ai_score": self.ai_score,
            "txs": [json.loads(tx.to_json()) for tx in self.txs],
            "signature_hex": self.signature_hex,
            "hash_hex": self.hash_hex,
        }
        return json.dumps(d, sort_keys=True)