# test_run.py
# Driver script that wires together spine, memory, curiosity, introspection, and logging
import os
import time
import numpy as np
import toml

from spine.neural_spine import NeuralSpine
from spine.curiosity_engine import CuriosityEngine
from spine.introspection import Introspection
from tools.reflection_logbook import ReflectionLogbook
from memory.memory_hffs import HFFSMemory


def main():
    # Load configuration
    config = toml.load('configs/rs-config.toml')

    # Prepare directories
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/sponge', exist_ok=True)

    # Initialize subsystems
    logbook = ReflectionLogbook(config['filepaths']['logbook'])
    memory = HFFSMemory(
        base_path=config['filepaths']['memory_base'],
        sponge_size=tuple(config['filepaths']['sponge_size'])
    )
    spine = NeuralSpine(config)
    curiosity = CuriosityEngine(
        memory=memory,
        threshold=config['training']['threshold']
    )
    introspect = Introspection(spine=spine, logbook=logbook)

    # Dummy input vector
    vec = np.random.rand(*memory.sponge_size)
    flat = vec.flatten()

    # 1) Forward pass through spine
    output = spine.forward(flat)
    print(f"[+] Spine output vector of length {len(output)}")

    # 2) Store output in memory
    key = f"state_{int(time.time())}"
    memory.store(key, output)
    print(f"[+] Stored state as '{key}'")

    # 3) Introspection report
    report = introspect.assess(flat, flat)
    print(f"[*] Introspection report: {report}")

    # 4) Compute novelty reward
    reward = curiosity.reward(output)
    print(f"[!] Novelty reward: {reward:.4f}")

    # 5) Log run summary
    entry = (
        f"**Run summary**\n"
        f"- Stored key: `{key}`\n"
        f"- Novelty reward: `{reward:.4f}`\n"
        f"- Module losses: `{report}`\n"
    )
    logbook.record(entry)
    print("[✓] Run logged to reflections.md")


if __name__ == '__main__':
    main()
