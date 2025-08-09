import os
import time
import threading
import queue
import numpy as np
from typing import Any, Dict

from spine.neural_spine import NeuralSpine
from spine.curiosity_engine import CuriosityEngine
from spine.introspection import Introspection
from spine.checkpoint import save_checkpoint, load_checkpoint
from memory.memory_hffs import HFFSMemory
from memory.replay_buffer import ReplayBuffer
from tools.reflection_logbook import ReflectionLogbook


class Orchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stop_event = threading.Event()

        # IO
        self.logbook = ReflectionLogbook(config['filepaths']['logbook'])
        self.memory = HFFSMemory(
            base_path=config['filepaths']['memory_base'],
            sponge_size=tuple(config['filepaths']['sponge_size'])
        )
        # Core
        self.spine = NeuralSpine(config)
        self.curiosity = CuriosityEngine(memory=self.memory, threshold=config['training']['threshold'])
        self.introspect = Introspection(spine=self.spine, logbook=self.logbook)
        self.replay = ReplayBuffer(capacity=int(config.get('training', {}).get('replay_capacity', 5000)))

        # Shared queues
        self.sample_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)

        # Threads
        self.explorer_thread = threading.Thread(target=self._explorer_loop, name="Explorer", daemon=True)
        self.learner_thread = threading.Thread(target=self._learner_loop, name="Learner", daemon=True)
        self.meta_thread = threading.Thread(target=self._meta_loop, name="Meta", daemon=True)

        # Checkpoint
        self.ckpt_path = self.config.get('filepaths', {}).get('checkpoint', 'data/checkpoints/ckpt.json')

    def start(self):
        os.makedirs('data/checkpoints', exist_ok=True)
        # Try to load a checkpoint
        load_checkpoint(self.spine, self.ckpt_path)
        self.explorer_thread.start()
        self.learner_thread.start()
        self.meta_thread.start()

    def stop(self):
        self.stop_event.set()
        for t in (self.explorer_thread, self.learner_thread, self.meta_thread):
            t.join(timeout=2.0)
        # Save final checkpoint
        save_checkpoint(self.spine, self.ckpt_path, extra={"timestamp": time.time()})

    # Threads
    def _explorer_loop(self):
        rng = np.random.default_rng()
        sponge_shape = tuple(self.memory.sponge_size)
        while not self.stop_event.is_set():
            # Generate candidate input and assess novelty
            candidate = rng.random(sponge_shape).reshape(-1).astype(np.float32)
            reward = self.curiosity.reward(candidate)
            if reward > 0.5:
                # Store novel sample
                try:
                    self.sample_queue.put(candidate, timeout=0.1)
                except queue.Full:
                    pass
            time.sleep(0.01)

    def _learner_loop(self):
        batch_size = int(self.config.get('training', {}).get('batch_size', 8))
        train_steps = int(self.config.get('training', {}).get('max_steps', 500))
        steps = 0
        last_log = time.time()
        while not self.stop_event.is_set() and steps < train_steps:
            # Accumulate into replay
            try:
                sample = self.sample_queue.get(timeout=0.1)
                self.replay.add(sample)
            except queue.Empty:
                pass

            # Train if we have enough data
            if self.replay.size() >= batch_size:
                batch = self.replay.sample(batch_size)
                # Do simple self-supervised identity training
                losses = []
                for i in range(batch.shape[0]):
                    x = batch[i]
                    loss = self.spine.train_step(x, x)
                    losses.append(loss)
                steps += 1
                # Periodic logging and checkpoint
                if time.time() - last_log > 1.0:
                    avg_loss = float(np.mean(losses)) if losses else 0.0
                    self.logbook.record(f"Trainer step={steps} avg_loss={avg_loss:.6f}")
                    save_checkpoint(self.spine, self.ckpt_path, extra={"step": steps})
                    last_log = time.time()

        # Signal completion
        self.stop_event.set()

    def _meta_loop(self):
        # Periodically run introspection and tune lr dynamically
        while not self.stop_event.is_set():
            report = self.introspect.assess(np.zeros(int(np.prod(self.memory.sponge_size))), None)
            # Simple lr annealing
            for module in self.spine.modules.values():
                if hasattr(module, 'lr'):
                    module.lr = max(1e-5, float(module.lr) * 0.999)
            # Log
            self.logbook.record(f"Meta report: {report}")
            time.sleep(2.0)


def run_multithreaded_training(config: Dict[str, Any], duration_seconds: int = 10) -> None:
    orch = Orchestrator(config)
    orch.start()
    try:
        time.sleep(duration_seconds)
    finally:
        orch.stop()