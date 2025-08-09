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
from chain.consensus import select_winner, verify_block, mint_reward, select_committee, slash_fraction
from chain.consensus_bft import bft_commit
from chain.contracts import ContractEngine
from chain.p2p import InProcBus
from chain.state import ChainState
from spine.neural_spine import NeuralSpine
from spine.curiosity_engine import CuriosityEngine
from memory.sponge_memory import create as create_entangled
from memory.memory_hffs import HFFSMemory
from tools.policy import PolicyEnforcer
from tools.why import WhyEngine
from chain.auction import JobAuction
from chain.oracles import red_team_oracle, eval_oracle
from chain.federated_dp import aggregate_metrics
from tools.zkml import generate_proof, verify_proof


class Node:
    BUS = InProcBus()  # simple shared bus for simulation

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
        self.contracts = ContractEngine.from_config(config.get('contracts', {}))
        self.policies = config.get('policies', {})
        self.policy = PolicyEnforcer(self.policies)
        self.why = WhyEngine(self.policies)
        self._stop = threading.Event()
        self._last_proposer_loss = None

    def train_and_score(self, steps: int = 20) -> Tuple[float, Dict[str, Any]]:
        shape = tuple(self.memory.sponge_size)
        x = np.random.rand(*shape).reshape(-1).astype(np.float32)
        last = None
        for _ in range(steps):
            last = self.spine.train_step(x, x)
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
        # Record proposer proof-of-improvement if applicable
        if self._last_proposer_loss is not None and bool(self.config.get('zkml', {}).get('enabled', False)):
            proof = generate_proof(self._last_proposer_loss, float(info.get('loss', 1.0)), {"proposer": self.node_id})
            if verify_proof(proof):
                txs.append(Transaction(kind="zkml", data=proof))
        self._last_proposer_loss = float(info.get('loss', 1.0))
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
        self.state = ChainState()
        for n in self.nodes:
            self.state.set_stake(n.node_id, n.wallet.staked())
        self.auction = JobAuction(base_reward=float(config.get('chain', {}).get('auction_reward', 0.25)))

    def run_round(self) -> Dict[str, Any]:
        proposals: List[Tuple[Block, bytes]] = []
        for node in self.nodes:
            blk = node.propose_block(self.ledger)
            proposals.append((blk, node.pub))
        committee = select_committee(proposals, k=max(1, len(self.nodes) // 2 + 1))
        committee_blocks = [proposals[i] for i in committee]
        commit_idx_local = bft_commit(committee_blocks, quorum=max(1, len(committee) // 2 + 1))
        if commit_idx_local < 0:
            return {"status": "no_commit"}
        commit_idx = committee[commit_idx_local]
        blk, pub = proposals[commit_idx]
        if not verify_block(blk):
            frac = slash_fraction('invalid_hash')
            for n in self.nodes:
                if n.node_id == blk.proposer:
                    penalty = n.wallet.staked() * frac
                    n.wallet.unstake(penalty)
                    self.state.apply_slash(n.node_id, frac)
                    break
            return {"status": "reject", "reason": "invalid_hash"}
        # Oracles and PoUW auction
        evals = eval_oracle({"ai_score": blk.ai_score, "stake": blk.stake})
        bidders = [{"id": n.node_id, "stake": n.wallet.staked(), "ai_score": blk.ai_score} for n in self.nodes]
        auction_res = self.auction.run(bidders, job={"intent": "useful_training"})
        if auction_res.get('winner'):
            win_id = auction_res['winner']['id']
            for n in self.nodes:
                if n.node_id == win_id:
                    n.wallet.deposit(auction_res['payout'])
        # DP aggregation of proposer loss (demo)
        dp_metrics = aggregate_metrics([{"loss": tx.data.get('info', {}).get('loss', 0.0)} for tx in blk.txs if tx.kind == 'train'])
        # Contracts + policy checks
        context = {"metrics": {"ai_score": blk.ai_score, "stake": blk.stake}, "proposer": blk.proposer}
        actions = self.nodes[0].contracts.evaluate(context) if self.nodes else []
        for act in actions:
            if act['type'] == 'mint':
                to = act.get('to', blk.proposer)
                allowed, msg = self.nodes[0].policy.check('mint', {"amount": act.get('amount', 0.0)})
                if allowed:
                    for n in self.nodes:
                        if n.node_id == to:
                            n.wallet.deposit(float(act['amount']))
            elif act['type'] == 'transfer':
                frm = act.get('from'); to = act.get('to'); amt = float(act.get('amount', 0.0))
                allowed, msg = self.nodes[0].policy.check('transfer_funds', {"amount": amt})
                if allowed:
                    src = next((n for n in self.nodes if n.node_id == frm), None)
                    dst = next((n for n in self.nodes if n.node_id == to), None)
                    if src and dst:
                        src.wallet.transfer_to(dst.wallet, amt)
        winner_idx = commit_idx
        reward = mint_reward(blk.ai_score, base_reward=self.base_reward)
        self.nodes[winner_idx].wallet.deposit(reward)
        self.state.set_stake(self.nodes[winner_idx].node_id, self.nodes[winner_idx].wallet.staked())
        appended = self.ledger.append(blk)
        return {"status": "ok" if appended else "reject", "winner": blk.proposer, "reward": reward, "height": self.ledger.height(), "actions": actions, "committee": committee, "auction": auction_res, "dp": dp_metrics}

    def run(self, rounds: int = 5) -> None:
        for _ in range(rounds):
            res = self.run_round()
            time.sleep(self.round_time)