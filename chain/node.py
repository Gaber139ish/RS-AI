import os
import time
import json
import threading
from typing import Any, Dict, List, Tuple
import numpy as np

from chain.crypto import generate_keypair, sign, verify_with_secret, pub_hex
from chain.wallet import Wallet
from chain.block import Block, Transaction
from chain.ledger import Ledger
from chain.consensus import select_winner, verify_block, mint_reward
from spine.neural_spine import NeuralSpine
from spine.curiosity_engine import CuriosityEngine
from memory.sponge_memory import create as create_entangled
from memory.memory_hffs import HFFSMemory


class Node:
    def __init__(self, config: Dict[str, Any], node_dir: str):
        self.config = config
        self.node_dir = node_dir
        os.makedirs(node_dir, exist_ok=True)
        # Identity
        self.pub, self.secret = generate_keypair()
        self.node_id = pub_hex(self.pub)
        # Wallet
        self.wallet = Wallet(initial_balance=float(config.get('chain', {}).get('initial_balance', 10.0)))
        self.wallet.stake(float(config.get('chain', {}).get('initial_stake', 5.0)))
        # AI core per node
        self.spine = NeuralSpine(config)
        mem_backend = (config.get('memory', {}).get('backend', 'hffs') or 'hffs').lower()
        if mem_backend == 'entangled':
            self.memory = create_entangled(config)
        else:
            self.memory = HFFSMemory(
                base_path=config['filepaths']['memory_base'],
                sponge_size=tuple(config['filepaths']['sponge_size'])
            )
        self.curiosity = CuriosityEngine(memory=self.memory, threshold=config['training']['threshold'])
        self._stop = threading.Event()

    def train_and_score(self, steps: int = 20) -> Tuple[float, Dict[str, Any]]:
        shape = tuple(self.memory.sponge_size)
        x = np.random.rand(*shape).reshape(-1).astype(np.float32)
        last = None
        for _ in range(steps):
            last = self.spine.train_step(x, x)
        # AI score: inverse of loss with logistic squashing
        loss = float(last or 1.0)
        ai_score = 1.0 / (1.0 + np.exp(5.0 * (loss - 0.01)))
        return float(ai_score), {"loss": loss}

    def propose_block(self, ledger: Ledger) -> Block:
        ai_score, info = self.train_and_score(steps=int(self.config.get('chain', {}).get('train_steps', 20)))
        stake = float(self.wallet.staked())
        txs = [Transaction(kind="train", data={"node": self.node_id, "info": info})]
        blk = Block(
            index=ledger.height() + 1,
            prev_hash=ledger.last_hash(),
            timestamp=time.time(),
            proposer=self.node_id,
            stake=stake,
            ai_score=ai_score,
            txs=txs,
        )
        blk.signature_hex = sign(self.secret, blk.compute_hash().encode('utf-8')).hex()
        blk.hash_hex = blk.compute_hash()
        return blk


class FederatedChain:
    def __init__(self, config: Dict[str, Any], base_dir: str, num_nodes: int = 3):
        self.config = config
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.ledger = Ledger(os.path.join(base_dir, 'ledger'))
        self.nodes: List[Node] = [Node(config, os.path.join(base_dir, f'node_{i}')) for i in range(num_nodes)]
        self.round_time = float(config.get('chain', {}).get('round_time', 2.0))
        self.base_reward = float(config.get('chain', {}).get('base_reward', 1.0))

    def run_round(self) -> Dict[str, Any]:
        proposals: List[Tuple[Block, bytes]] = []
        for node in self.nodes:
            blk = node.propose_block(self.ledger)
            proposals.append((blk, node.pub))
        winner_idx = select_winner(proposals)
        blk, pub = proposals[winner_idx]
        # Verify and append
        if not verify_block(blk):
            return {"status": "reject", "reason": "invalid_hash"}
        # Reward: mint to winner wallet
        reward = mint_reward(blk.ai_score, base_reward=self.base_reward)
        self.nodes[winner_idx].wallet.deposit(reward)
        appended = self.ledger.append(blk)
        return {"status": "ok" if appended else "reject", "winner": blk.proposer, "reward": reward, "height": self.ledger.height()}

    def run(self, rounds: int = 5) -> None:
        for _ in range(rounds):
            res = self.run_round()
            time.sleep(self.round_time)