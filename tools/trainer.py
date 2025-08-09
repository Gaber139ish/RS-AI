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
from memory.prioritized_replay import PrioritizedReplayBuffer
from tools.reflection_logbook import ReflectionLogbook
from tools.metrics import MetricsRegistry
from tools.dashboard import start_dashboard, ControlBridge
from memory.sponge_memory import create as create_entangled


class Orchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()  # when set, training pauses

        # IO
        self.logbook = ReflectionLogbook(config['filepaths']['logbook'])
        mem_backend = (config.get('memory', {}).get('backend', 'hffs') or 'hffs').lower()
        if mem_backend == 'entangled':
            self.memory = create_entangled(config)
        else:
            self.memory = HFFSMemory(
                base_path=config['filepaths']['memory_base'],
                sponge_size=tuple(config['filepaths']['sponge_size'])
            )
        # Core
        self.spine = NeuralSpine(config)
        self.curiosity = CuriosityEngine(memory=self.memory, threshold=config['training']['threshold'])
        self.introspect = Introspection(spine=self.spine, logbook=self.logbook)
        self.replay = PrioritizedReplayBuffer(
            capacity=int(config.get('training', {}).get('replay_capacity', 5000)),
            alpha=float(config.get('training', {}).get('replay_alpha', 0.6))
        )

        # Metrics + Dashboard
        self.metrics = MetricsRegistry()
        self.bridge = ControlBridge()
        self.bridge.set_orchestrator(self)
        host = config.get('dashboard', {}).get('host', '127.0.0.1')
        port = int(config.get('dashboard', {}).get('port', 8080))
        self.http_server = start_dashboard(host, port, self.metrics, self.bridge)
        self.http_thread = threading.Thread(target=self.http_server.serve_forever, name="Dashboard", daemon=True)

        # Shared queues
        self.sample_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)

        # Threads
        self.explorer_thread = threading.Thread(target=self._explorer_loop, name="Explorer", daemon=True)
        self.learner_thread = threading.Thread(target=self._learner_loop, name="Learner", daemon=True)
        self.meta_thread = threading.Thread(target=self._meta_loop, name="Meta", daemon=True)

        # Checkpoint
        self.ckpt_path = self.config.get('filepaths', {}).get('checkpoint', 'data/checkpoints/ckpt.json')

    # Control API
    def pause(self):
        self.pause_event.set()

    def resume(self):
        self.pause_event.clear()

    def set_param(self, key: str, value: Any):
        # Allow adjusting some parameters at runtime
        if key == 'lr':
            for module in self.spine.modules.values():
                if hasattr(module, 'lr'):
                    module.lr = float(value)
        elif key == 'threshold':
            self.curiosity.threshold = float(value)
        else:
            raise ValueError(f"Unsupported param: {key}")

    def start(self):
        os.makedirs('data/checkpoints', exist_ok=True)
        load_checkpoint(self.spine, self.ckpt_path)
        self.http_thread.start()
        self.explorer_thread.start()
        self.learner_thread.start()
        self.meta_thread.start()

    def stop(self):
        self.stop_event.set()
        self.http_server.shutdown()
        for t in (self.explorer_thread, self.learner_thread, self.meta_thread, self.http_thread):
            t.join(timeout=2.0)
        save_checkpoint(self.spine, self.ckpt_path, extra={"timestamp": time.time()})

    # Threads
    def _explorer_loop(self):
        rng = np.random.default_rng()
        sponge_shape = tuple(self.memory.sponge_size)
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.05)
                continue
            # Generate candidate input and assess novelty
            candidate = rng.random(sponge_shape).reshape(-1).astype(np.float32)
            reward = self.curiosity.reward(candidate)
            if reward > 0.5:
                try:
                    self.sample_queue.put(candidate, timeout=0.1)
                    self.metrics.inc('explorer_accepted', 1)
                except queue.Full:
                    self.metrics.inc('explorer_dropped', 1)
            else:
                self.metrics.inc('explorer_rejected', 1)
            time.sleep(0.005)

    def _learner_loop(self):
        batch_size = int(self.config.get('training', {}).get('batch_size', 8))
        train_steps = int(self.config.get('training', {}).get('max_steps', 500))
        steps = 0
        last_log = time.time()
        while not self.stop_event.is_set() and steps < train_steps:
            if self.pause_event.is_set():
                time.sleep(0.05)
                continue
            # Accumulate into prioritized replay
            try:
                sample = self.sample_queue.get(timeout=0.1)
                # Use novelty as priority
                priority = float(self.curiosity.reward(sample))
                self.replay.add(sample, priority=priority)
                self.metrics.inc('replay_size', 1)
            except queue.Empty:
                pass

            # Train if we have enough data
            if self.replay.size() >= batch_size:
                batch = self.replay.sample(batch_size)
                losses = []
                for i in range(batch.shape[0]):
                    x = batch[i]
                    loss = self.spine.train_step(x, x)
                    losses.append(loss)
                steps += 1
                self.metrics.set('trainer_steps', steps)
                if time.time() - last_log > 1.0:
                    avg_loss = float(np.mean(losses)) if losses else 0.0
                    self.metrics.set('avg_loss', avg_loss)
                    self.logbook.record(f"Trainer step={steps} avg_loss={avg_loss:.6f}")
                    save_checkpoint(self.spine, self.ckpt_path, extra={"step": steps})
                    last_log = time.time()

        self.stop_event.set()

    def _meta_loop(self):
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue
            report = self.introspect.assess(np.zeros(int(np.prod(self.memory.sponge_size))), None)
            for module in self.spine.modules.values():
                if hasattr(module, 'lr'):
                    module.lr = max(1e-5, float(module.lr) * 0.999)
            self.logbook.record(f"Meta report: {report}")
            self.metrics.set('meta_last_report_ok', True)
            time.sleep(2.0)


def run_multithreaded_training(config: Dict[str, Any], duration_seconds: int = 10) -> None:
    orch = Orchestrator(config)
    orch.start()
    try:
        time.sleep(duration_seconds)
    finally:
        orch.stop()