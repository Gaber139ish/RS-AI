from typing import Any, Dict


class SafetyChecker:
    def __init__(self, policies: Dict[str, Any] | None = None):
        self.policies = policies or {}

    def check_output(self, vector) -> Dict[str, Any]:
        # Placeholder: always safe
        return {"safe": True, "reason": "ok"}