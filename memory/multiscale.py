from typing import Any, Dict, List, Tuple
import numpy as np

from .sponge_memory import EntangledSpongeMemory, create as create_entangled


class MultiScaleMemory:
    def __init__(self, configs: List[Dict[str, Any]]):
        # Each config describes one scale (e.g., different block_size)
        self.scales: List[EntangledSpongeMemory] = [create_entangled(cfg) for cfg in configs]
        self._size = self.scales[0].sponge_size

    @property
    def sponge_size(self) -> Tuple[int, int, int]:
        return self._size

    def store(self, key: str, vector):
        for mem in self.scales:
            mem.store(key, vector)

    def load(self, key: str):
        # Blend reconstructions from coarse to fine
        recon = None
        for i, mem in enumerate(self.scales):
            part = mem.load(key)
            if recon is None:
                recon = part
            else:
                alpha = (i + 1) / len(self.scales)
                recon = (1.0 - alpha) * recon + alpha * part
        return recon if recon is not None else np.zeros(self.sponge_size, dtype=np.float32)

    def distance_to_nearest(self, state_vector) -> float:
        # Take min distance across scales
        return min(mem.distance_to_nearest(state_vector) for mem in self.scales)


def create_multiscale(base_config: Dict[str, Any]) -> MultiScaleMemory:
    mc = base_config.copy()
    memcfg = mc.get('memory', {}).copy()
    # Generate 3 scales: coarse, mid, fine
    grids = [(13, 13, 13), (9, 9, 9), (7, 7, 7)]
    configs = []
    for bs in grids:
        cfg = base_config.copy()
        m = cfg.get('memory', {}).copy()
        m['block_size'] = list(bs)
        cfg['memory'] = m
        configs.append(cfg)
    return MultiScaleMemory(configs)