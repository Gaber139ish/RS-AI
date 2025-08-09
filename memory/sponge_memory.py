import os
import json
import math
import threading
from typing import Any, Dict, List, Tuple

import numpy as np

from .hebbian import HebbianUpdater


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _gaussian_kernel(distance: float, sigma: float) -> float:
    return math.exp(- (distance ** 2) / (2 * (sigma ** 2)))


class SpongeTopology:
    def __init__(self, sponge_size: Tuple[int, int, int], block_size: Tuple[int, int, int]):
        self.sponge_size = tuple(int(x) for x in sponge_size)
        self.block_size = tuple(int(x) for x in block_size)
        self.grid = tuple(int(math.ceil(s / b)) for s, b in zip(self.sponge_size, self.block_size))

    def iter_blocks(self):
        gx, gy, gz = self.grid
        for bx in range(gx):
            for by in range(gy):
                for bz in range(gz):
                    yield (bx, by, bz)

    def block_bounds(self, block_idx: Tuple[int, int, int]) -> Tuple[slice, slice, slice]:
        bx, by, bz = block_idx
        bsx, bsy, bsz = self.block_size
        sx = slice(bx * bsx, min((bx + 1) * bsx, self.sponge_size[0]))
        sy = slice(by * bsy, min((by + 1) * bsy, self.sponge_size[1]))
        sz = slice(bz * bsz, min((bz + 1) * bsz, self.sponge_size[2]))
        return sx, sy, sz

    def neighbors(self, block_idx: Tuple[int, int, int], radius: int = 1):
        gx, gy, gz = self.grid
        bx, by, bz = block_idx
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = bx + dx, by + dy, bz + dz
                    if 0 <= nx < gx and 0 <= ny < gy and 0 <= nz < gz:
                        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                        yield (nx, ny, nz), dist


class EntangledSpongeMemory:
    def __init__(self, base_path: str, sponge_size: Tuple[int, int, int], block_size: Tuple[int, int, int] = (9, 9, 9), entanglement_strength: float = 0.15, neighbor_radius: int = 1, holographic_dim: int = 1024, seed: int = 42, hebbian: bool = True, hebbian_lr: float = 0.001, hebbian_decay: float = 0.995):
        self.base_path = base_path
        self.sponge_size = tuple(sponge_size)
        self.block_size = tuple(block_size)
        self.entanglement_strength = float(entanglement_strength)
        self.neighbor_radius = int(neighbor_radius)
        self.holo_dim = int(holographic_dim)
        self.topology = SpongeTopology(self.sponge_size, self.block_size)

        # Paths
        self.blocks_path = os.path.join(self.base_path, "blocks")
        self.index_path = os.path.join(self.base_path, "index.json")
        self.signatures_path = os.path.join(self.base_path, "signatures")
        self.global_path = os.path.join(self.base_path, "global_latent.npy")

        # Init
        _ensure_dir(self.base_path)
        _ensure_dir(self.blocks_path)
        _ensure_dir(self.signatures_path)
        if not os.path.isfile(self.index_path):
            with open(self.index_path, 'w') as f:
                json.dump({}, f)
        if not os.path.isfile(self.global_path):
            np.save(self.global_path, np.zeros((self.holo_dim,), dtype=np.float32))

        # Holographic random projection matrix (fixed)
        rng = np.random.default_rng(seed)
        self.holo_proj = rng.normal(0, 1.0 / math.sqrt(self.holo_dim), size=(self.holo_dim, int(np.prod(self.sponge_size)))).astype(np.float32)
        self._lock = threading.Lock()

        # Hebbian updater
        self._hebbian = HebbianUpdater(lr=hebbian_lr, decay=hebbian_decay) if hebbian else None

    # Internal IO
    def _block_file(self, idx: Tuple[int, int, int]) -> str:
        return os.path.join(self.blocks_path, f"{idx[0]}_{idx[1]}_{idx[2]}.npy")

    def _read_block(self, idx: Tuple[int, int, int]) -> np.ndarray:
        path = self._block_file(idx)
        if os.path.isfile(path):
            return np.load(path)
        sx, sy, sz = self.topology.block_bounds(idx)
        shape = (sx.stop - sx.start, sy.stop - sy.start, sz.stop - sz.start)
        return np.zeros(shape, dtype=np.float32)

    def _write_block(self, idx: Tuple[int, int, int], data: np.ndarray) -> None:
        path = self._block_file(idx)
        np.save(path, data.astype(np.float32))

    def _load_index(self) -> Dict[str, Any]:
        with open(self.index_path, 'r') as f:
            return json.load(f)

    def _save_index(self, idx: Dict[str, Any]) -> None:
        tmp = self.index_path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(idx, f)
        os.replace(tmp, self.index_path)

    def _signature_file(self, key: str) -> str:
        return os.path.join(self.signatures_path, f"{key}.npy")

    def _compute_signature(self, flat_vector: np.ndarray) -> np.ndarray:
        # Holographic random projection as a global signature
        return (self.holo_proj @ flat_vector.astype(np.float32)).astype(np.float32)

    # Public API
    def store(self, key: str, vector: np.ndarray) -> None:
        with self._lock:
            sponge = np.asarray(vector, dtype=np.float32).reshape(self.sponge_size)
            flat = sponge.reshape(-1)
            # Update global latent
            global_latent = np.load(self.global_path)
            global_latent = 0.99 * global_latent + 0.01 * self._compute_signature(flat)
            np.save(self.global_path, global_latent)

            # Write blocks and entangle neighbors
            index = self._load_index()
            touched_blocks: List[Tuple[int, int, int]] = []
            for bidx in self.topology.iter_blocks():
                sx, sy, sz = self.topology.block_bounds(bidx)
                local = sponge[sx, sy, sz]
                current = self._read_block(bidx)
                # Optional Hebbian modulation
                if self._hebbian is not None:
                    current = self._hebbian.update(current, pre=local, post=current)
                # Direct write as EMA
                updated = 0.9 * current + 0.1 * local
                # Spread to neighbors
                for nidx, dist in self.topology.neighbors(bidx, radius=self.neighbor_radius):
                    nsx, nsy, nsz = self.topology.block_bounds(nidx)
                    nblock = self._read_block(nidx)
                    # Determine overlap slice size to blend; use min shapes
                    blend = local[:nblock.shape[0], :nblock.shape[1], :nblock.shape[2]]
                    strength = self.entanglement_strength * _gaussian_kernel(dist, sigma=max(1e-6, self.neighbor_radius / 2))
                    nblock = (1.0 - strength) * nblock + strength * blend
                    self._write_block(nidx, nblock)
                self._write_block(bidx, updated)
                touched_blocks.append(bidx)

            # Update index and signature
            index[key] = {
                "blocks": touched_blocks,
                "timestamp": float(time_now()),
            }
            self._save_index(index)
            sig = self._compute_signature(flat)
            np.save(self._signature_file(key), sig)

    def load(self, key: str) -> np.ndarray:
        with self._lock:
            index = self._load_index()
            meta = index.get(key)
            if meta is None:
                return np.zeros(self.sponge_size, dtype=np.float32)
            sponge = np.zeros(self.sponge_size, dtype=np.float32)
            weight = np.zeros(self.sponge_size, dtype=np.float32)
            for bidx in meta.get("blocks", []):
                bidx = tuple(bidx)
                sx, sy, sz = self.topology.block_bounds(bidx)
                local = self._read_block(bidx)
                sponge[sx, sy, sz] += local
                weight[sx, sy, sz] += 1.0
            sponge = np.divide(sponge, np.maximum(weight, 1.0))
            # Refine with global latent by projecting back via pseudo-inverse
            try:
                flat = sponge.reshape(-1)
                global_latent = np.load(self.global_path)
                correction = self.holo_proj.T @ global_latent
                flat = 0.99 * flat + 0.01 * correction
                sponge = flat.reshape(self.sponge_size)
            except Exception:
                pass
            return sponge

    def distance_to_nearest(self, state_vector: np.ndarray) -> float:
        with self._lock:
            flat = np.asarray(state_vector, dtype=np.float32).reshape(-1)
            sig = self._compute_signature(flat)
            best = float('inf')
            # Compare to signatures for speed
            for fname in os.listdir(self.signatures_path):
                if not fname.endswith('.npy'):
                    continue
                other = np.load(os.path.join(self.signatures_path, fname))
                # Cosine distance
                denom = float(np.linalg.norm(sig) * np.linalg.norm(other) + 1e-9)
                cos = float(np.dot(sig, other) / denom)
                dist = 1.0 - cos
                if dist < best:
                    best = dist
            return best if best != float('inf') else 0.0

    @property
    def sponge_size(self) -> Tuple[int, int, int]:
        return self._sponge_size

    @sponge_size.setter
    def sponge_size(self, val):
        self._sponge_size = tuple(val)


def time_now() -> float:
    import time
    return time.time()


def create(config: Dict[str, Any] | None = None) -> EntangledSpongeMemory:
    if config is None:
        base = "data/sponge"
        sponge_size = (27, 27, 27)
        block_size = (9, 9, 9)
        entanglement_strength = 0.15
        neighbor_radius = 1
        holo_dim = 1024
        hebbian = True
        hebbian_lr = 0.001
        hebbian_decay = 0.995
    else:
        fp = config.get("filepaths", {})
        base = fp.get("memory_base", "data/sponge")
        sponge_size = tuple(fp.get("sponge_size", [27, 27, 27]))
        memcfg = config.get("memory", {})
        block_size = tuple(memcfg.get("block_size", [9, 9, 9]))
        entanglement_strength = float(memcfg.get("entanglement_strength", 0.15))
        neighbor_radius = int(memcfg.get("neighbor_radius", 1))
        holo_dim = int(memcfg.get("holographic_dim", 1024))
        hebbian = bool(memcfg.get("hebbian", True))
        hebbian_lr = float(memcfg.get("hebbian_lr", 0.001))
        hebbian_decay = float(memcfg.get("hebbian_decay", 0.995))
    return EntangledSpongeMemory(
        base_path=base,
        sponge_size=sponge_size,
        block_size=block_size,
        entanglement_strength=entanglement_strength,
        neighbor_radius=neighbor_radius,
        holographic_dim=holo_dim,
        hebbian=hebbian,
        hebbian_lr=hebbian_lr,
        hebbian_decay=hebbian_decay,
    )