from typing import Any, Tuple, Optional

from tools.policy import PolicyEnforcer


class GuardedMemory:
    def __init__(self, backend: Any, policy: PolicyEnforcer):
        self._backend = backend
        self._policy = policy

    @property
    def sponge_size(self) -> Tuple[int, int, int]:
        return getattr(self._backend, 'sponge_size')

    def store(self, key: str, vector, context: Optional[dict] = None) -> bool:
        ctx = context or {}
        allowed, _ = self._policy.check('store_memory', ctx)
        if not allowed:
            return False
        self._backend.store(key, vector)
        return True

    def load(self, key: str):
        return self._backend.load(key)

    def distance_to_nearest(self, state_vector) -> float:
        return self._backend.distance_to_nearest(state_vector)