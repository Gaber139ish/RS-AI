from typing import Dict, Any, Tuple


class PolicyEnforcer:
    def __init__(self, policies: Dict[str, Any]):
        self.policies = policies or {}

    def check(self, action: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        # Minimal rule stubs; extend with real checks
        if action == 'store_memory' and context.get('contains_sensitive', False):
            return False, 'Privacy: sensitive data must not be stored'
        if action == 'transfer_funds' and float(context.get('amount', 0.0)) > float(self.policies.get('max_transfer', 1000.0)):
            return False, 'Security: transfer exceeds policy limit'
        return True, 'OK'